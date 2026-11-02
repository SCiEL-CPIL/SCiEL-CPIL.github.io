import time
import numpy as np
from scipy.optimize import minimize

N = 600
DEADLINE = 140.0  # Safety margin before 150s kill

def compute_score(heights):
    """Compute the normalized maximum of autoconvolution for a step function.
    Returns the value such that uniform heights give score 1.0.
    """
    heights = np.asarray(heights, dtype=np.float64)
    if np.any(heights < -1e-15):
        return np.inf
    heights = np.maximum(heights, 0.0)
    
    n = len(heights)
    w = 0.5 / n  # width of each interval
    
    # Use FFT for linear autoconvolution
    M = 1
    while M < 2 * n - 1:
        M *= 2
    
    f = np.zeros(M, dtype=np.float64)
    f[:n] = heights
    
    F = np.fft.rfft(f)
    conv = np.fft.irfft(F * F, n=M)
    
    max_conv = np.max(conv)
    total = np.sum(heights)
    
    if total <= 0:
        return np.inf
    
    score = max_conv * n / (total * total * w)
    return score

def _autoconv(heights):
    """Compute autoconvolution using FFT."""
    n = len(heights)
    M = 1
    while M < 2 * n - 1:
        M *= 2
    f = np.zeros(M, dtype=np.float64)
    f[:n] = heights
    F = np.fft.rfft(f)
    conv = np.fft.irfft(F * F, n=M)
    return conv

def _smooth_max_approx(x, alpha):
    """Smooth approximation of max(x) using log-sum-exp."""
    max_x = np.max(x)
    if max_x == 0:
        return 0.0
    shifted = x - max_x
    return max_x + np.log(np.sum(np.exp(alpha * shifted))) / alpha

def _fista_optimize(h0, max_iters=500, tol=1e-6):
    """
    FISTA (Fast Iterative Shrinkage-Thresholding Algorithm) for:
    minimize max(conv(h)) subject to h >= 0, sum(h) = 1
    
    We reformulate as: minimize f(h) + g(h) where
    f(h) = max(conv(h)) (smoothed), g(h) = indicator of {h >= 0, sum(h)=1}
    
    Actually, let's use projected gradient on the normalized space.
    We minimize: F(h) = max(conv(h)) / (sum(h)^2 * w) subject to h >= 0
    But this is non-convex due to the normalization.
    
    Alternative: Fix sum(h) = 1 by constraint, then minimize max(conv(h)).
    The score becomes max(conv(h)) * N / (1^2 * w) = max(conv(h)) * N / w
    
    So we minimize max(conv(h)) subject to h >= 0, sum(h) = 1.
    This is a convex optimization problem!
    """
    n = len(h0)
    w = 0.5 / n
    
    # Normalize initial point
    h = h0.copy()
    total = np.sum(h)
    if total <= 0:
        h = np.ones(n) / n
    else:
        h = h / total
    
    # FISTA parameters
    L = 1.0  # Lipschitz constant estimate (will be adaptive)
    t_k = 1.0
    
    h_prev = h.copy()
    
    for k in range(max_iters):
        # Compute gradient of smoothed max(conv(h))
        # conv(h) = h * h (linear convolution)
        # d(conv)/dh = 2 * h (in the sense that conv is quadratic)
        # But we need gradient of max over indices
        
        M = 1
        while M < 2 * n - 1:
            M *= 2
        
        f_arr = np.zeros(M, dtype=np.float64)
        f_arr[:n] = h
        F = np.fft.rfft(f_arr)
        conv = np.fft.irfft(F * F, n=M)
        
        # Find index of max
        idx_max = np.argmax(conv)
        
        # Gradient of conv[idx_max] with respect to h
        # conv[j] = sum_i h[i] * h[j-i] (for valid j)
        # d(conv[j])/dh[i] = h[j-i] + h[i-j] (if both indices valid)
        # For j = idx_max:
        grad = np.zeros(n)
        j = idx_max
        for i in range(n):
            # conv[j] = sum_{k} h[k] * h[j-k]
            # d/dh[i] = h[j-i] (if 0 <= j-i < n) + h[i-j] (if 0 <= i-j < n)
            if 0 <= j - i < n:
                grad[i] += h[j - i]
            if 0 <= i - j < n:
                grad[i] += h[i - j]
        
        # Projected gradient step
        step = 1.0 / L
        h_new = h - step * grad
        
        # Project onto {h >= 0, sum(h) = 1}
        # First enforce non-negativity
        h_new = np.maximum(h_new, 0)
        # Then normalize
        total_new = np.sum(h_new)
        if total_new > 0:
            h_new = h_new / total_new
        else:
            h_new = np.ones(n) / n
        
        # Check convergence
        if np.linalg.norm(h_new - h) < tol:
            break
        
        # FISTA momentum update
        t_next = (1 + np.sqrt(1 + 4 * t_k**2)) / 2
        h_prev = h_new + ((t_k - 1) / t_next) * (h_new - h)
        h = h_prev
        t_k = t_next
        
        # Adaptive L (backtracking)
        # For simplicity, keep L fixed but could increase if needed
        # The objective is smooth with bounded Hessian for normalized h
        
    return h

def _two_block_optimize():
    """Optimize a two-block structure: heights are a on [0,a], 0 on (a,b), b on [b,1].
    Parameters: a (left block end), b (right block start), left_height, right_height
    """
    def param_to_heights(params):
        a, b, lh, rh = params
        a = np.clip(a, 0, 1)
        b = np.clip(b, a, 1)
        lh = max(0, lh)
        rh = max(0, rh)
        h = np.zeros(N)
        if a > 0:
            h[:int(a * N)] = lh
        if b < 1:
            h[int(b * N):] = rh
        return h
    
    def param_score(params):
        h = param_to_heights(params)
        if np.sum(h) <= 0:
            return 1e10
        return compute_score(h)
    
    # Multi-start for two-block
    best_params = None
    best_sc = np.inf
    
    for a in [0.1, 0.2, 0.3, 0.4, 0.5]:
        for b in [a + 0.05, a + 0.1, a + 0.15, a + 0.2, 0.8, 0.9]:
            for lh in [0.5, 1.0, 1.5, 2.0]:
                for rh in [0.5, 1.0, 1.5, 2.0]:
                    params = [a, b, lh, rh]
                    sc = param_score(params)
                    if sc < best_sc:
                        best_sc = sc
                        best_params = params.copy()
    
    # Refine with scipy
    if best_params is not None:
        result = minimize(param_score, best_params, method='Nelder-Mead',
                         options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4})
        if result.fun < best_sc:
            best_sc = result.fun
            best_params = result.x
    
    return param_to_heights(best_params) if best_params is not None else np.ones(N)

def construct():
    start_time = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    def deadline_reached():
        return time_left() < 5.0
    
    rng = np.random.default_rng(42)
    
    # Start with uniform
    h = np.ones(N, dtype=np.float64)
    best_h = h.copy()
    best_score = compute_score(h)
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Uniform
    candidates.append(np.ones(N, dtype=np.float64))
    
    # 2. Gaussian centered with various widths
    x = np.linspace(0, 1, N)
    for width in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]:
        hg = np.exp(-((x - 0.5) / width) ** 2)
        candidates.append(hg)
    
    # 3. Two blocks with various configurations
    for left_frac in [0.1, 0.2, 0.3, 0.4]:
        for gap_frac in [0.0, 0.05, 0.1, 0.15, 0.2]:
            left_end = int(left_frac * N)
            gap_end = left_end + int(gap_frac * N)
            if gap_end >= N:
                continue
            hb = np.zeros(N)
            hb[:left_end] = 1.0
            hb[gap_end:] = 1.0
            if np.sum(hb) > 0:
                candidates.append(hb)
                
    # 4. Random sparse with various densities
    for _ in range(10):
        hr = np.zeros(N)
        density = rng.uniform(0.2, 0.8)
        mask = rng.random(N) < density
        hr[mask] = 1.0
        candidates.append(hr)
        
    # 5. Cosine tapers
    for tf in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]:
        ht = np.ones(N)
        tl = int(tf * N)
        if tl > 1:
            ramp = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, tl)))
            ht[:tl] = ramp
            ht[-tl:] = ramp[::-1]
        candidates.append(ht)
        
    # 6. Plateau with tapered edges
    for plateau_frac in [0.3, 0.5, 0.7]:
        hp = np.zeros(N)
        p_start = int((1 - plateau_frac) / 2 * N)
        p_end = int((1 + plateau_frac) / 2 * N)
        hp[p_start:p_end] = 1.0
        # Taper
        taper_len = max(1, int(0.1 * N))
        if p_start > taper_len:
            ramp = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, taper_len)))
            hp[p_start - taper_len:p_start] = ramp
        if p_end < N - taper_len:
            ramp = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, taper_len)))
            hp[p_end:p_end + taper_len] = ramp[::-1]
        candidates.append(hp)

    # Evaluate initial candidates
    for hc in candidates:
        if deadline_reached():
            break
        hc = np.maximum(hc, 0.0)
        if np.sum(hc) <= 0:
            continue
        sc = compute_score(hc)
        if sc < best_score:
            best_score = sc
            best_h = hc.copy()

    # NEW: FISTA-based convex optimization from multiple starting points
    # The problem: minimize max(conv(h)) subject to h >= 0, sum(h) = 1
    # is convex! The score is then max(conv(h)) * N / w
    
    fista_starts = [
        best_h.copy(),
        np.ones(N) / N,
        np.exp(-((np.linspace(0, 1, N) - 0.5) / 0.2) ** 2),
        np.zeros(N),  # will be handled
        np.concatenate([np.ones(int(0.3*N)), np.zeros(int(0.4*N)), np.ones(int(0.3*N))]),
    ]
    
    for start_h in fista_starts:
        if deadline_reached():
            break
        
        start_h = np.maximum(start_h, 0)
        total = np.sum(start_h)
        if total <= 0:
            start_h = np.ones(N) / N
        else:
            start_h = start_h / total
        
        # Run FISTA
        max_iters = 200
        h_fista = _fista_optimize(start_h, max_iters=max_iters, tol=1e-5)
        
        # Compute score
        sc = compute_score(h_fista)
        if sc < best_score:
            best_score = sc
            best_h = h_fista.copy()
    
    # Two-block optimization
    if time_left() > 20:
        h_2block = _two_block_optimize()
        sc_2block = compute_score(h_2block)
        if sc_2block < best_score:
            best_score = sc_2block
            best_h = h_2block.copy()
    
    # Final polish: Local refinement around best
    if time_left() > 10:
        current_h = best_h.copy()
        current_score = best_score
        
        # Adaptive local search
        step_size = 0.05
        for _ in range(500):
            if deadline_reached():
                break
            
            # Perturb random subset
            num_perturb = rng.integers(3, 15)
            indices = rng.choice(N, num_perturb, replace=False)
            noise = rng.normal(0, step_size, num_perturb)
            new_h = current_h.copy()
            new_h[indices] = np.maximum(current_h[indices] + noise, 0.0)
            
            # Renormalize to keep sum constant (helps with the convex structure)
            total_new = np.sum(new_h)
            if total_new > 0:
                new_h = new_h / total_new
            
            new_score = compute_score(new_h)
            
            if new_score < current_score:
                current_h = new_h
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_h = current_h.copy()
                step_size = min(0.1, step_size * 1.05)
            else:
                step_size = max(0.01, step_size * 0.9)
    
    best_h = np.maximum(best_h, 0.0)
    return list(best_h)

if __name__ == "__main__":
    res = construct()
    assert len(res) == N
    assert all(h >= 0 for h in res)
    print("Constructed optimized step function.")
    print(f"Score: {compute_score(np.array(res))}")
