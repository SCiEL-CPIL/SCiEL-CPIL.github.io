import numpy as np
import time
from scipy.optimize import minimize

N = 50
DELTA = 1.0 / N

def _autoconv_sup(h):
    """Compute sup norm of autoconvolution of step function with heights h."""
    n = len(h)
    num_points = 2 * n - 1
    ff_vals = np.zeros(num_points)
    
    for k in range(num_points):
        total = 0.0
        i_min = max(0, k - (n - 1))
        i_max = min(n - 1, k)
        for i in range(i_min, i_max + 1):
            j_min = max(0, k - i - 1)
            j_max = min(n - 1, k - i + 1)
            if j_min > j_max:
                continue
            js = np.arange(j_min, j_max + 1)
            lo = np.maximum(i, k - js)
            hi = np.minimum(i + 1, k - js + 1)
            overlaps = np.maximum(0, hi - lo)
            total += np.sum(h[i] * h[js] * overlaps)
        ff_vals[k] = total * DELTA
    
    return np.max(ff_vals)

def _ratio(h):
    """Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf)."""
    h = np.maximum(h, 0)
    sum_h2 = np.sum(h**2)
    sum_h = np.sum(h)
    if sum_h < 1e-15:
        return 0.0
    sup_ff = _autoconv_sup(h)
    if sup_ff < 1e-15:
        return 0.0
    return sum_h2 / (sum_h * sup_ff)

def _grad_peak_conv(h, k_star):
    """
    Compute gradient of (h*h)[k_star] with respect to h.
    conv[k] = sum_{i} h[i] * h[k-i] (with h=0 outside [0, N-1])
    d conv[k] / d h[i] = h[k-i] + h[i-k] (if indices valid)
    """
    N_local = len(h)
    grad = np.zeros(N_local)
    for i in range(N_local):
        val = 0.0
        idx1 = k_star - i
        if 0 <= idx1 < N_local:
            val += h[idx1]
        idx2 = i - k_star
        if 0 <= idx2 < N_local:
            val += h[idx2]
        grad[i] = val
    return grad

def _generate_sidon_set(n, max_size=None):
    if max_size is None:
        max_size = n
    sidon = [0]
    used_sums = {0}
    for candidate in range(1, n):
        valid = True
        for i in sidon:
            s = i + candidate
            if s in used_sums:
                valid = False
                break
        if valid:
            for i in sidon:
                if i + candidate in used_sums:
                    valid = False
                    break
            if valid:
                sidon.append(candidate)
                used_sums.add(candidate + candidate)
                for i in sidon[:-1]:
                    used_sums.add(i + candidate)
        if len(sidon) >= max_size:
            break
    return sidon

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill

    best_h = np.ones(N)
    best_score = _ratio(best_h)

    # --- Phase 1: Generate diverse initializations ---
    np.random.seed(42)
    initializations = []
    
    # 1. Sidon sets
    for size in [8, 10, 12, 14]:
        sidon_set = _generate_sidon_set(N, max_size=size)
        h_sidon = np.zeros(N)
        for idx in sidon_set:
            h_sidon[idx] = 1.0
        initializations.append(h_sidon)
        
    # 2. Sparse random patterns
    for trial in range(15):
        h_sparse = np.zeros(N)
        num_ones = np.random.randint(5, 15)
        indices = np.random.choice(N, num_ones, replace=False)
        h_sparse[indices] = np.random.uniform(0.5, 1.5, num_ones)
        initializations.append(h_sparse)
        
    # 3. Uniform
    initializations.append(np.ones(N))
    
    # 4. Alternating
    h_alt = np.zeros(N)
    h_alt[::2] = 1.0
    initializations.append(h_alt)
    
    # 5. Gaussian
    t = np.linspace(-1, 1, N)
    initializations.append(np.exp(-t**2/0.3))
    
    # 6. Bimodal
    initializations.append(np.maximum(1.0 + 2.0*np.exp(-((t-0.6)**2)/0.1) + 2.0*np.exp(-((t+0.6)**2)/0.1), 0.05))

    # --- Phase 2: Refine each initialization using a hybrid strategy ---
    # For each init:
    # 1. Run a short Nelder-Mead to get to a good local basin.
    # 2. Run the multiplicative gradient flow (from inspiration 1) to polish.
    #    The gradient flow is great for fine-tuning the "sidelobes" because it
    #    directly attacks the peak of the autoconvolution.
    
    def refine_h(h0):
        h0 = np.maximum(h0, 0)
        s0 = np.sum(h0)
        if s0 < 1e-15:
            return None, 0.0
        
        h = h0 / s0
        
        # Step 2a: Nelder-Mead (short run)
        def neg_ratio(x):
            x = np.maximum(x, 0)
            s = np.sum(x)
            if s < 1e-15:
                return 0.0
            return -_ratio(x / s)
        
        try:
            result = minimize(neg_ratio, h.copy(), method='Nelder-Mead',
                             options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            h = np.maximum(result.x, 0)
            s = np.sum(h)
            if s > 1e-15:
                h = h / s
        except Exception:
            pass
        
        # Step 2b: Multiplicative Gradient Flow
        # This is the structural change: using the game-theoretic gradient flow
        # to refine the solution found by NM.
        
        eta = 0.05
        max_iters = 200
        for i in range(max_iters):
            if time.time() > DEADLINE - 5.0:
                break
            
            # Compute autoconvolution
            # We need the peak location and value.
            # For efficiency, we can just compute the full conv via the helper or a faster method.
            # Since N=50, np.convolve is fast enough.
            conv = np.convolve(h, h, mode='full')
            sup_conv = np.max(conv)
            if sup_conv < 1e-15:
                break
            
            k_star = np.argmax(conv)
            
            # Compute gradient
            grad_h = _grad_peak_conv(h, k_star)
            
            # Multiplicative update
            # h_i <- h_i * exp(-eta * grad_h_i)
            # We want to minimize the peak, so we decrease h where it contributes to the peak.
            update = np.exp(-eta * grad_h)
            h_new = h * update
            
            # Renormalize
            s_new = np.sum(h_new)
            if s_new < 1e-15:
                break
            h_new = h_new / s_new
            
            # Check improvement
            r_old = _ratio(h)
            r_new = _ratio(h_new)
            
            if r_new > r_old:
                h = h_new
            else:
                # Decay eta if no improvement
                eta *= 0.95
                if eta < 1e-4:
                    break
                    
        return h, _ratio(h)

    for h0 in initializations:
        if time.time() > DEADLINE - 20:
            break
        
        h_candidate, score_candidate = refine_h(h0)
        if h_candidate is not None and score_candidate > best_score:
            best_score = score_candidate
            best_h = h_candidate.copy()
            
    # --- Phase 3: Final Polish on the best solution ---
    # Run a longer gradient flow on the best solution found.
    if time.time() < DEADLINE - 10:
        h_best = best_h.copy()
        s = np.sum(h_best)
        if s > 1e-15:
            h_best = h_best / s
            
        eta = 0.02
        max_iters = 500
        for i in range(max_iters):
            if time.time() > DEADLINE - 2.0:
                break
            
            conv = np.convolve(h_best, h_best, mode='full')
            sup_conv = np.max(conv)
            if sup_conv < 1e-15:
                break
            
            k_star = np.argmax(conv)
            grad_h = _grad_peak_conv(h_best, k_star)
            
            update = np.exp(-eta * grad_h)
            h_new = h_best * update
            
            s_new = np.sum(h_new)
            if s_new < 1e-15:
                break
            h_new = h_new / s_new
            
            r_old = _ratio(h_best)
            r_new = _ratio(h_new)
            
            if r_new > r_old:
                h_best = h_new
            else:
                eta *= 0.95
                if eta < 1e-5:
                    break
        
        if _ratio(h_best) > best_score:
            best_score = _ratio(h_best)
            best_h = h_best.copy()
            
    return best_h.tolist()
