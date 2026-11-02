"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_START_TIME = time.time()
_TIME_LIMIT = 140.0

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _max_overlap(h):
    """
    Compute the maximum overlap of h with shifted (1-h).
    Overlap at shift k: sum_i h[i] * (1 - h[(i+k) % N])
    Since sum(h) = N/2, this equals N/2 - sum_i h[i]*h[(i+k)%N]
    So minimizing max overlap is equivalent to maximizing min circular autocorrelation.
    """
    N_local = len(h)
    # Use FFT for circular autocorrelation
    H = np.fft.fft(h)
    autocorr = np.fft.ifft(H * np.conj(H)).real
    # autocorr[k] = sum_i h[i] * h[(i+k) % N]
    # We want to maximize the minimum of autocorr[k] for k=1..N-1 (k=0 is always max)
    # Actually, let's just compute overlaps directly to be safe and clear
    overlaps = []
    for k in range(1, N_local):
        shifted = np.roll(h, k)
        overlap = np.sum(h * (1 - shifted))
        overlaps.append(overlap)
    return max(overlaps)

def _project_to_simplex_with_bounds(params, target_sum):
    """
    Project params to sum = target_sum with bounds [0, 1].
    Uses iterative water-filling.
    """
    p = np.clip(params, 0, 1)
    for _ in range(100):
        diff = target_sum - np.sum(p)
        if abs(diff) < 1e-12:
            break
        mask = (p > 0) & (p < 1)
        if np.sum(mask) == 0:
            break
        p[mask] += diff / np.sum(mask)
        p = np.clip(p, 0, 1)
    return p

def _soft_min_obj_and_grad(h, beta):
    """
    Objective: minimize max overlap = N/2 - min(autocorr[k]) for k=1..N-1
    Equivalent to: maximize min(autocorr[k])
    Using soft-min of autocorr values.
    
    Returns (objective, gradient) where objective = -soft_min(autocorr)
    (negated because we want to maximize soft_min, but minimize function)
    """
    N_local = len(h)
    H = np.fft.fft(h)
    autocorr = np.fft.ifft(H * np.conj(H)).real
    
    # We care about lags 1 to N-1
    # autocorr[0] is the energy, which is fixed-ish but varies with h
    # Actually, we want to maximize the minimum of autocorr[1:]
    # Let's use the overlap formulation directly to be consistent with _max_overlap
    # overlap[k] = N/2 - autocorr[k]
    # max overlap = N/2 - min(autocorr[k])
    # So minimizing max overlap is same as maximizing min(autocorr[k])
    
    # Use soft-min of autocorr[1:]
    a = autocorr[1:]
    min_val = np.min(a)
    shifted = a - min_val
    # Avoid numerical issues
    exp_terms = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_terms)
    if sum_exp == 0:
        soft_min = min_val
    else:
        soft_min = min_val - np.log(sum_exp) / beta
    
    # Objective is -soft_min (we want to minimize this)
    obj = -soft_min
    
    # Gradient
    # d(soft_min)/d(a_j) = exp(-beta * a_j) / sum(exp(-beta * a))
    weights = exp_terms / sum_exp
    
    # We need d(a_k)/d(h_i)
    # a_k = sum_i h[i] * h[(i+k)%N]
    # This is a circular autocorrelation.
    # Using FFT: a = ifft(|H|^2)
    # d(a)/d(h) is complex. Let's use the direct formula for gradient.
    # grad_h[i] = sum_k weights[k-1] * d(a_k)/d(h_i)
    # d(a_k)/d(h_i) = h[(i+k)%N] + h[(i-k)%N]  (for k != 0)
    # Note: a_k = a_{N-k} due to symmetry? No, for real signals, autocorr is symmetric.
    # a_k = sum h[i]h[i+k].
    # Let's compute gradient via the explicit sum for accuracy, or use FFT adjoint.
    
    # Explicit gradient computation (N=100, so 100*100 = 10000 ops, very fast)
    grad_h = np.zeros(N_local)
    for k in range(1, N_local):
        w = weights[k-1]
        if w == 0: continue
        # a_k = sum_i h[i] * h[(i+k)%N]
        # d(a_k)/d(h_i) = h[(i+k)%N] + h[(i-k)%N]
        # Note: if k=0, it would be 2*h[i], but we skip k=0.
        # Also note: a_{N-k} = a_k. So we are double counting if we iterate all k?
        # No, weights are for specific lags.
        # Let's just sum the contributions.
        grad_h += w * (np.roll(h, -k) + np.roll(h, k))
        
    # Since obj = -soft_min, gradient is -d(soft_min)/d(h)
    grad_h = -grad_h
    
    return obj, grad_h

def construct():
    start_time = time.time()
    deadline = start_time + _TIME_LIMIT - 10.0  # Safety margin
    
    # Initial candidates
    candidates = []
    
    # 1. Uniform
    h0 = np.full(N, 0.5)
    candidates.append(h0)
    
    # 2. Alternating
    h1 = np.zeros(N)
    h1[0::2] = 1.0
    candidates.append(h1)
    
    # 3. Sinusoids
    x = np.linspace(0, 2*np.pi, N, endpoint=False)
    for freq in [1, 2, 3, 5, 10]:
        for phase in [0, np.pi/2]:
            h_s = 0.5 + 0.5 * np.cos(freq * x + phase)
            # Normalize
            s = np.sum(h_s)
            if s > 0:
                h_s = h_s / s * (N / 2)
                h_s = np.clip(h_s, 0, 1)
                # Re-normalize after clip
                s2 = np.sum(h_s)
                if s2 > 0:
                    h_s = h_s / s2 * (N / 2)
                    h_s = np.clip(h_s, 0, 1)
            candidates.append(h_s)
            
    # 4. Random
    np.random.seed(42)
    for _ in range(10):
        h_r = np.random.uniform(0, 1, N)
        h_r = h_r / np.sum(h_r) * (N / 2)
        h_r = np.clip(h_r, 0, 1)
        s = np.sum(h_r)
        if s > 0:
             h_r = h_r / s * (N / 2)
             h_r = np.clip(h_r, 0, 1)
        candidates.append(h_r)

    # Evaluate initial candidates
    best_h = candidates[0].copy()
    best_score = _max_overlap(best_h)
    
    for h in candidates[1:]:
        s = _max_overlap(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
    
    # Optimization loop
    # We will use L-BFGS-B with the soft-min objective
    # We need to handle the equality constraint sum(h) = N/2
    # L-BFGS-B doesn't support equality constraints directly.
    # Strategy: 
    # 1. Use a penalty method or just project after each step? 
    #    Projection inside the objective/gradient is tricky for L-BFGS-B.
    # 2. Better: Use scipy.optimize.minimize with method='SLSQP' which supports constraints.
    #    SLSQP might be slower but N=100 is small.
    # 3. Alternative: Parameterize h to satisfy the constraint automatically?
    #    e.g., h = sigmoid(x) and adjust x? Hard to hit exact sum.
    # 4. Let's stick with SLSQP for the main optimization, using the soft-min objective.
    
    def slsqp_obj(h):
        # h is a flat array
        obj, _ = _soft_min_obj_and_grad(h, beta=100.0)
        return obj
        
    def slsqp_grad(h):
        _, grad = _soft_min_obj_and_grad(h, beta=100.0)
        return grad

    bounds = [(0, 1)] * N
    constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - N/2, 'jac': lambda h: np.ones(N)}]
    
    # Try optimizing from the best candidate and a few others
    for h_init in [best_h, candidates[0], candidates[1]]:
        if _time_remaining() < 10.0:
            break
        
        try:
            # SLSQP
            res = minimize(slsqp_obj, h_init, method='SLSQP', jac=slsqp_grad,
                           bounds=bounds, constraints=constraints,
                           options={'maxiter': 300, 'ftol': 1e-8, 'disp': False})
            if res.fun < best_score:
                h_opt = np.clip(res.x, 0, 1)
                # Ensure sum constraint is met exactly
                s = np.sum(h_opt)
                if s > 0:
                    h_opt = h_opt / s * (N / 2)
                    h_opt = np.clip(h_opt, 0, 1)
                s_final = _max_overlap(h_opt)
                if s_final < best_score:
                    best_score = s_final
                    best_h = h_opt.copy()
        except Exception:
            pass

    # If SLSQP didn't improve much, try a local search / hill climbing
    # or a different optimizer like COBYLA (derivative-free)
    if _time_remaining() > 20.0:
        def cobyla_obj(h):
            return _max_overlap(h)
            
        try:
            res = minimize(cobyla_obj, best_h, method='COBYLA',
                           options={'maxiter': 500, 'rhobeg': 0.1, 'catol': 1e-6, 'disp': False})
            if res.fun < best_score:
                h_opt = np.clip(res.x, 0, 1)
                s = np.sum(h_opt)
                if s > 0:
                    h_opt = h_opt / s * (N / 2)
                    h_opt = np.clip(h_opt, 0, 1)
                s_final = _max_overlap(h_opt)
                if s_final < best_score:
                    best_score = s_final
                    best_h = h_opt.copy()
        except Exception:
            pass

    # Final polish: ensure validity
    h_final = np.clip(best_h, 0, 1)
    s = np.sum(h_final)
    if s > 0:
        h_final = h_final / s * (N / 2)
        h_final = np.clip(h_final, 0, 1)
    
    # One last check: if sum is not exactly N/2 due to floating point, adjust
    # The problem requires sum to be exactly len/2.
    # We can adjust one element to fix the sum.
    diff = N/2 - np.sum(h_final)
    if abs(diff) > 1e-9:
        # Find an element that can be adjusted
        for i in range(N):
            new_val = h_final[i] + diff
            if 0 <= new_val <= 1:
                h_final[i] = new_val
                break
    
    return h_final
# EVOLVE-BLOCK-END
