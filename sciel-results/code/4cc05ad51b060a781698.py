"""
Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Uses joint optimization of h and epigraph variable t with log-sum-exp smoothing.
Includes spectral initialization to target flat autocorrelation structures.
"""
import time
import numpy as np

N = 100
_T_START = time.time()
_DEADLINE = _T_START + 145


def _linear_overlap_values(h):
    """Compute C_k = (2/N)*(N/2 - r_k) where r_k = sum_{i=k}^{N-1} h_i h_{i-k}."""
    n = len(h)
    n_fft = 1
    while n_fft < 2 * n:
        n_fft *= 2
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs


def _score(h):
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))


def _project(h, target_sum):
    """
    Project h onto the set {x : 0 <= x_i <= 1, sum(x) = target_sum}.
    Uses bisection on the Lagrange multiplier for the sum constraint.
    """
    h = np.asarray(h, dtype=np.float64)
    low = -np.max(h) - 1.0
    high = 1.0 - np.min(h) + 1.0
    
    # If sum is already correct and within bounds, return h
    current_sum = np.sum(h)
    if abs(current_sum - target_sum) < 1e-12 and np.all(h >= -1e-12) and np.all(h <= 1 + 1e-12):
        return np.clip(h, 0.0, 1.0)

    def sum_of_clipped(offset):
        return np.sum(np.clip(h + offset, 0.0, 1.0))

    # Bisection
    for _ in range(60):
        mid = (low + high) / 2.0
        s_mid = sum_of_clipped(mid)
        if s_mid > target_sum:
            high = mid
        else:
            low = mid
    
    offset = (low + high) / 2.0
    h_proj = np.clip(h + offset, 0.0, 1.0)
    
    # Final correction for numerical precision
    diff = target_sum - np.sum(h_proj)
    if abs(diff) > 1e-14:
        # Adjust the element with the most "room"
        if diff > 0:
            # Add to smallest
            idx = np.argmin(h_proj)
            h_proj[idx] += diff
        else:
            # Subtract from largest
            idx = np.argmax(h_proj)
            h_proj[idx] += diff
        h_proj = np.clip(h_proj, 0.0, 1.0)
        
    return h_proj


def _smooth_max_obj_grad(h, beta):
    """
    Compute the value and gradient of the smooth max of C_k.
    F(h) = (1/beta) * log(sum(exp(beta * C_k)))
    grad F = sum(w_k * grad C_k)
    """
    cs = _linear_overlap_values(h)
    
    # Compute weights w_k
    # Shift to avoid overflow
    cs_max = np.max(cs)
    exp_cs = np.exp(beta * (cs - cs_max))
    sum_exp = np.sum(exp_cs)
    w = exp_cs / sum_exp
    
    val = (1.0 / beta) * (np.log(sum_exp) + beta * cs_max)
    
    # Compute gradient
    # grad C_k_j = -(2/N) * (h_{j-k} [j>=k] + h_{j+k} [j+k<N])
    # grad_F_j = -(2/N) * ( sum_{k=0}^j w_k h_{j-k} + sum_{k=0}^{N-1-j} w_k h_{j+k} )
    
    # Efficient calculation using FFT
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
        
    w_pad = np.zeros(n_fft)
    w_pad[:N] = w
    h_pad = np.zeros(n_fft)
    h_pad[:N] = h
    
    W = np.fft.rfft(w_pad)
    H = np.fft.rfft(h_pad)
    
    # Convolution w * h
    conv = np.fft.irfft(W * H, n=n_fft)
    A = conv[:N]
    
    # Correlation of w and h
    w_rev = w[::-1]
    w_rev_pad = np.zeros(n_fft)
    w_rev_pad[:N] = w_rev
    W_rev = np.fft.rfft(w_rev_pad)
    
    conv_rev = np.fft.irfft(W_rev * H, n=n_fft)
    B = conv_rev[N-1 : N-1+N]
    
    grad = -(2.0 / N) * (A + B)
    
    return val, grad


def _epigraph_obj_grad(h, t, beta):
    """
    Objective: t + (1/beta) * log(sum(exp(beta * (C_k(h) - t))))
    Gradient w.r.t h: sum(w_k * grad C_k)
    Gradient w.r.t t: 1 - sum(w_k) = 0 (since w sums to 1)
    Wait, grad_t = 1 - sum(exp(beta(C_k-t)) / sum(exp(beta(C_k-t)))) = 1 - 1 = 0?
    No, the term is (1/beta) log(sum(exp(beta C_k - beta t))).
    d/dt = (1/beta) * (1/sum) * sum(exp(beta C_k - beta t) * (-beta)) = - sum(exp(...))/sum = -1.
    So grad_t = 1 - 1 = 0.
    This implies that for any fixed h, the optimal t is max(C_k).
    So minimizing F(h,t) is equivalent to minimizing max(C_k).
    However, the Hessian structure might help if we view it as a regularized problem.
    Actually, if we just minimize max(C_k), we can use the smooth max directly.
    The epigraph form is useful if we want to constrain C_k <= T and minimize T.
    Let's stick to the smooth max of C_k, but use better initialization.
    """
    val, grad = _smooth_max_obj_grad(h, beta)
    return val, grad


def _nesterov_smooth_max(h0, beta, n_iter=500, lr=0.01):
    """
    Nesterov accelerated gradient descent on the smooth max objective.
    """
    h = np.array(h0, dtype=np.float64)
    h = _project(h, N / 2.0)
    
    h_prev = h.copy()
    t_mom = 1.0
    
    best_h = h.copy()
    best_score = _score(h)
    
    for i in range(n_iter):
        if time.time() > _DEADLINE - 5:
            break
            
        t_new = (1 + np.sqrt(1 + 4 * t_mom * t_mom)) / 2
        momentum = (t_mom - 1) / t_new
        
        y = h + momentum * (h - h_prev)
        
        val_y, grad = _smooth_max_obj_grad(y, beta)
        
        # Backtracking line search
        step_lr = lr
        h_new = None
        for bt in range(20):
            h_candidate = y - step_lr * grad
            h_candidate = _project(h_candidate, N / 2.0)
            val_candidate = _score(h_candidate)
            if val_candidate <= val_y + 1e-6:
                h_new = h_candidate
                break
            step_lr *= 0.5
        
        if h_new is None:
            h_new = y - lr * grad
            h_new = _project(h_new, N / 2.0)
        
        h_prev = h
        h = h_new
        t_mom = t_new
        
        current_score = _score(h)
        if current_score < best_score:
            best_score = current_score
            best_h = h.copy()
            
    return best_h, best_score


def _generate_spectral_candidates():
    """
    Generate candidates with flat spectral magnitude.
    We want |H_k| to be constant.
    We can construct H_k = A * exp(i * phi_k) where phi_k are random phases.
    Then h = ifft(H_k).
    We project h onto [0,1] and sum N/2.
    """
    candidates = []
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
    
    np.random.seed(42)
    
    # Number of frequency bins to set (excluding DC)
    # For real signal, H_k = H_{N-k}^*
    # We set magnitudes to be constant for k=1..N/2
    
    for _ in range(10):
        # Random phases
        phases = np.random.randn(n_fft // 2 + 1)
        # Magnitude: constant for k>0, 0 for DC? 
        # Sum h = N/2 means DC component is fixed.
        # Let's just generate random complex spectrum with constant magnitude for k!=0
        H = np.zeros(n_fft // 2 + 1, dtype=np.complex128)
        # DC component determines the mean. We want mean 0.5.
        # Sum h = N * h_0 (DC term in ifft is H_0/n_fft? No, ifft(H)_0 = 1/n_fft sum H_k)
        # Let's just create h with mean 0.5 and flat spectrum.
        
        # Simpler: Generate random h, compute FFT, zero out DC, set magnitudes to constant, add DC back.
        h_rand = np.random.randn(N)
        h_rand -= np.mean(h_rand) # Mean 0
        H_rand = np.fft.rfft(h_rand, n=n_fft)
        
        # Set magnitudes of H_rand[1:] to constant
        mags = np.abs(H_rand)
        if np.any(mags[1:] > 0):
            target_mag = np.mean(mags[1:])
            phases = np.angle(H_rand)
            H_new = np.zeros(n_fft // 2 + 1, dtype=np.complex128)
            H_new[1:] = target_mag * np.exp(1j * phases[1:])
            
            # Add DC component to make mean 0.5
            # ifft(H_new)[0] = H_new[0] / n_fft (approx, depending on normalization)
            # numpy ifft: x_n = 1/n sum X_k exp(2pi i nk/n)
            # So x_0 = 1/n sum X_k.
            # We want mean x = 0.5 => sum x = N/2 => sum x = N * mean(x)
            # mean(x) = 1/n_fft * H_new[0] (if n_fft=N, but we padded)
            # Actually, if we pad to n_fft, the time domain is length n_fft.
            # We only care about first N.
            # Let's just set H_new[0] such that the resulting h has mean 0.5.
            # h = ifft(H_new)
            # mean(h[0:N]) is not exactly H_new[0]/n_fft due to truncation.
            # We will project later, so we just need a good shape.
            
            H_new[0] = 0.5 * n_fft # Approximate DC for mean 0.5 in length n_fft
            h_candidate = np.fft.irfft(H_new, n=n_fft)[:N]
            
            # Project
            h_candidate = _project(h_candidate, N / 2.0)
            candidates.append(h_candidate)
            
    # Also add the uniform vector
    candidates.append(np.full(N, 0.5))
    
    return candidates


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    best_h = None
    best_score = float('inf')
    
    # Phase 1: Spectral Initialization + Nesterov
    candidates = _generate_spectral_candidates()
    
    # Add some random inits
    np.random.seed(42)
    for _ in range(5):
        h_init = np.random.rand(N)
        h_init = _project(h_init, N / 2.0)
        candidates.append(h_init)
        
    # Run Nesterov on each candidate
    # We use a schedule of increasing beta (temperature)
    betas = [10.0, 20.0, 50.0, 100.0]
    
    for h_init in candidates:
        if time.time() > _DEADLINE - 30:
            break
            
        h_opt = h_init.copy()
        for beta in betas:
            if time.time() > _DEADLINE - 10:
                break
            h_opt, s_opt = _nesterov_smooth_max(h_opt, beta, n_iter=200, lr=0.05)
            if s_opt < best_score:
                best_score = s_opt
                best_h = h_opt.copy()
                
    # Phase 2: Coordinate Descent Polish
    if best_h is not None and time.time() < _DEADLINE - 5:
        h_ref = best_h.copy()
        for outer in range(50):
            if time.time() > _DEADLINE - 2:
                break
            improved = False
            for i in range(N):
                if time.time() > _DEADLINE - 2:
                    break
                old_val = h_ref[i]
                # Try a few values
                candidates_val = [old_val - 0.01, old_val - 0.001, old_val + 0.001, old_val + 0.01, 0.0, 1.0]
                for new_val in candidates_val:
                    if new_val < 0 or new_val > 1:
                        continue
                    h_test = h_ref.copy()
                    h_test[i] = new_val
                    h_test = _project(h_test, N / 2.0)
                    s_test = _score(h_test)
                    if s_test < best_score - 1e-12:
                        best_score = s_test
                        best_h = h_test.copy()
                        h_ref = h_test.copy()
                        improved = True
            if not improved:
                break

    if best_h is None:
        best_h = np.full(N, 0.5)
        
    best_h = _project(best_h, N / 2.0)
    return best_h
