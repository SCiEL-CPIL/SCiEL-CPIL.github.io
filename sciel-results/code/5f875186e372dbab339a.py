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

def _smooth_max_and_grad(heights, alpha=50.0):
    """Compute smooth max of autoconvolution and its gradient w.r.t. heights."""
    n = len(heights)
    M = 1
    while M < 2 * n - 1:
        M *= 2
    
    f = np.zeros(M, dtype=np.float64)
    f[:n] = heights
    
    F = np.fft.rfft(f)
    conv = np.fft.irfft(F * F, n=M)
    
    # Soft max: (1/alpha) * log(sum(exp(alpha * conv)))
    max_conv = np.max(conv)
    shifted = alpha * (conv - max_conv)
    log_sum_exp = max_conv + (1.0 / alpha) * np.log(np.sum(np.exp(shifted)))
    
    # Weights for the soft max
    weights = np.exp(shifted) / np.sum(np.exp(shifted))
    
    # Gradient: d(soft_max)/d(heights[i]) = sum_k weights[k] * d(conv[k])/d(heights[i])
    # d(conv[k])/d(f[i]) = f[k-i] + f[i-k] (with out-of-range = 0)
    # For efficiency, use only top few peaks where weights are significant
    
    top_k = min(10, len(conv))
    idx = np.argsort(conv)[-top_k:]
    w_idx = weights[idx]
    w_sum = np.sum(w_idx)
    if w_sum > 0:
        w_idx = w_idx / w_sum
    else:
        w_idx = np.ones(top_k) / top_k
    
    grad = np.zeros(n)
    for k in idx:
        wk = w_idx[np.where(idx == k)[0][0]]
        # d(conv[k])/d(f[i]) for i in [0, n)
        # conv[k] = sum_j f[j] * f[k-j]
        # d/d(f[i]) = f[k-i] + f[i-k] (if indices valid)
        j1 = k - np.arange(n)  # k - i for i in 0..n-1
        j2 = np.arange(n) - k  # i - k for i in 0..n-1
        mask1 = (j1 >= 0) & (j1 < n)
        mask2 = (j2 >= 0) & (j2 < n)
        contrib1 = np.where(mask1, f[j1[mask1]], 0.0)
        contrib2 = np.where(mask2, f[j2[mask2]], 0.0)
        grad += wk * (contrib1 + contrib2)
    
    return log_sum_exp, grad

def _mirror_descent_step(heights, grad, step_size, eta=0.1):
    """Mirror descent update for nonnegativity constraint.
    Uses entropy regularizer: h_new[i] = h[i] * exp(-eta * grad[i] / (step_size + eps))
    Then normalize to maintain similar scale.
    """
    eps = 1e-12
    # Mirror descent: h_new = h * exp(-eta * grad / step_size)
    # But we need to be careful with the scaling
    log_h = np.log(np.maximum(heights, eps))
    log_h_new = log_h - eta * grad / (step_size + eps)
    h_new = np.exp(log_h_new)
    h_new = np.maximum(h_new, 0.0)
    
    # Normalize to maintain similar total mass
    total_old = np.sum(heights)
    total_new = np.sum(h_new)
    if total_new > 0:
        h_new = h_new * (total_old / total_new)
    
    return h_new

def _gradient_descent_step(heights, grad, step_size):
    """Simple gradient descent with projection onto nonnegative orthant."""
    h_new = heights - step_size * grad
    h_new = np.maximum(h_new, 0.0)
    return h_new

def _two_block_params_to_heights(params):
    """Convert 6 parameters to a height array.
    params: [left_center, left_width, left_height, right_center, right_width, right_height]
    """
    lc, lw, lh, rc, rw, rh = params
    lc = np.clip(lc, 0, 1)
    lw = np.clip(lw, 0.01, 1)
    lh = np.clip(lh, 0, 10)
    rc = np.clip(rc, 0, 1)
    rw = np.clip(rw, 0.01, 1)
    rh = np.clip(rh, 0, 10)
    
    x = np.linspace(0, 1, N)
    h = np.zeros(N)
    
    left_block = np.zeros(N)
    mask_l = np.abs(x - lc) <= lw / 2
    if np.any(mask_l):
        dist = np.abs(x[mask_l] - lc) / (lw / 2)
        left_block[mask_l] = lh * 0.5 * (1 + np.cos(np.pi * dist))
    
    right_block = np.zeros(N)
    mask_r = np.abs(x - rc) <= rw / 2
    if np.any(mask_r):
        dist = np.abs(x[mask_r] - rc) / (rw / 2)
        right_block[mask_r] = rh * 0.5 * (1 + np.cos(np.pi * dist))
    
    h = np.maximum(left_block, right_block)
    return h

def _parametric_score(params):
    """Score for a parametric two-block function."""
    h = _two_block_params_to_heights(params)
    return compute_score(h)

def _two_block_optimize(rng, start_time, deadline):
    """Optimize two-block parameters using gradient-based methods."""
    best_score = np.inf
    best_params = None
    
    n_starts = 12
    for ps in range(n_starts):
        if time.time() - start_time > deadline - 5:
            break
        
        lc = rng.uniform(0.1, 0.4)
        lw = rng.uniform(0.1, 0.3)
        lh = rng.uniform(0.5, 2.0)
        rc = rng.uniform(0.6, 0.9)
        rw = rng.uniform(0.1, 0.3)
        rh = rng.uniform(0.5, 2.0)
        
        x0 = np.array([lc, lw, lh, rc, rw, rh])
        
        try:
            result = minimize(_parametric_score, x0, method='Nelder-Mead',
                              options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-7})
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
        except Exception:
            continue
    
    # Symmetric configurations
    for width_frac in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
        if time.time() - start_time > deadline - 5:
            break
        for height in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0]:
            x0 = np.array([0.25, width_frac, height, 0.75, width_frac, height])
            try:
                result = minimize(_parametric_score, x0, method='Nelder-Mead',
                                  options={'maxiter': 150, 'xatol': 1e-5, 'fatol': 1e-7})
                if result.fun < best_score:
                    best_score = result.fun
                    best_params = result.x.copy()
            except Exception:
                continue
    
    return best_score, best_params

def _smooth_objective(params, alpha=50.0):
    """Smooth objective for parametric optimization."""
    h = _two_block_params_to_heights(params)
    smooth_val, _ = _smooth_max_and_grad(h, alpha)
    total = np.sum(h)
    if total <= 0:
        return 1e10
    w = 0.5 / N
    return smooth_val * N / (total * total * w)

def _smooth_objective_grad(params, alpha=50.0):
    """Gradient of smooth objective w.r.t. parametric parameters."""
    h = _two_block_params_to_heights(params)
    smooth_val, grad_h = _smooth_max_and_grad(h, alpha)
    total = np.sum(h)
    if total <= 0:
        return 1e10, np.zeros(len(params))
    w = 0.5 / N
    obj = smooth_val * N / (total * total * w)
    
    # Numerical gradient w.r.t. parameters
    eps = 1e-6
    grad_params = np.zeros(len(params))
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += eps
        h_plus = _two_block_params_to_heights(params_plus)
        smooth_plus, _ = _smooth_max_and_grad(h_plus, alpha)
        total_plus = np.sum(h_plus)
        if total_plus <= 0:
            obj_plus = 1e10
        else:
            obj_plus = smooth_plus * N / (total_plus * total_plus * w)
        grad_params[i] = (obj_plus - obj) / eps
    
    return obj, grad_params

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
    x = np.linspace(0, 1, N)
    
    # 1. Uniform
    candidates.append(np.ones(N, dtype=np.float64))
    
    # 2. Gaussian centered
    for width in [0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4]:
        hg = np.exp(-((x - 0.5) / width) ** 2)
        candidates.append(hg)
    
    # 3. Two blocks
    for left_frac in [0.1, 0.15, 0.2, 0.25, 0.3]:
        for gap_frac in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
            left_end = int(left_frac * N)
            gap_end = left_end + int(gap_frac * N)
            if gap_end >= N:
                continue
            hb = np.zeros(N)
            hb[:left_end] = 1.0
            hb[gap_end:] = 1.0
            if np.sum(hb) > 0:
                candidates.append(hb)
    
    # 4. Random sparse
    for _ in range(10):
        hr = np.zeros(N)
        density = rng.uniform(0.3, 0.7)
        mask = rng.random(N) < density
        hr[mask] = 1.0
        candidates.append(hr)
    
    # 5. Cosine tapers
    for tf in [0.05, 0.1, 0.15, 0.2, 0.3]:
        ht = np.ones(N)
        tl = int(tf * N)
        if tl > 1:
            ramp = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, tl)))
            ht[:tl] = ramp
            ht[-tl:] = ramp[::-1]
        candidates.append(ht)
    
    # 6. Plateau with tapered edges
    for plateau_frac in [0.3, 0.4, 0.5, 0.6, 0.7]:
        hp = np.zeros(N)
        p_start = int((1 - plateau_frac) / 2 * N)
        p_end = int((1 + plateau_frac) / 2 * N)
        hp[p_start:p_end] = 1.0
        taper_len = max(1, int(0.05 * N))
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

    # Phase 1: Parametric two-block optimization
    if time_left() > 35:
        two_block_score, two_block_params = _two_block_optimize(rng, start_time, DEADLINE)
        if two_block_params is not None and two_block_score < best_score:
            best_h = _two_block_params_to_heights(two_block_params)
            best_score = two_block_score

    # Phase 2: Mirror descent with smooth max surrogate
    if time_left() > 20:
        current_h = best_h.copy()
        current_score = best_score
        alpha = 50.0
        eta = 0.05
        step_size = 0.1
        patience = 0
        max_patience = 50
        
        for it in range(200):
            if deadline_reached():
                break
            
            smooth_val, grad = _smooth_max_and_grad(current_h, alpha)
            
            # Normalize gradient
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-12:
                grad = grad / grad_norm
            
            # Mirror descent step
            h_new = _mirror_descent_step(current_h, grad, step_size, eta)
            
            new_score = compute_score(h_new)
            
            if new_score < current_score:
                current_h = h_new
                current_score = new_score
                patience = 0
                if current_score < best_score:
                    best_score = current_score
                    best_h = current_h.copy()
                # Slowly increase alpha for sharper max approximation
                alpha = min(200.0, alpha * 1.05)
                step_size = min(0.2, step_size * 1.02)
            else:
                patience += 1
                step_size = max(0.01, step_size * 0.9)
                eta = max(0.01, eta * 0.95)
                
                if patience >= max_patience:
                    break
    
    # Phase 3: Local refinement with coordinate descent
    if time_left() > 15:
        current_h = best_h.copy()
        current_score = best_score
        
        coord_passes = 3
        for _ in range(coord_passes):
            if deadline_reached():
                break
            for i in range(N):
                if deadline_reached():
                    break
                old_val = current_h[i]
                for delta in [-0.06, -0.03, -0.01, 0.01, 0.03, 0.06]:
                    new_h = current_h.copy()
                    new_h[i] = max(0.0, old_val + delta)
                    new_score = compute_score(new_h)
                    if new_score < current_score:
                        current_h = new_h
                        current_score = new_score
                        if current_score < best_score:
                            best_score = current_score
                            best_h = current_h.copy()
                        break

    # Phase 4: Random polish
    if time_left() > 8:
        current_h = best_h.copy()
        current_score = best_score
        step_size = 0.03
        
        for _ in range(1500):
            if deadline_reached():
                break
            num_perturb = rng.integers(3, 15)
            indices = rng.choice(N, num_perturb, replace=False)
            noise = rng.normal(0, step_size, num_perturb)
            new_h = current_h.copy()
            new_h[indices] = np.maximum(current_h[indices] + noise, 0.0)
            new_score = compute_score(new_h)
            
            if new_score < current_score:
                current_h = new_h
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_h = current_h.copy()
                step_size = min(0.08, step_size * 1.03)
            else:
                step_size = max(0.002, step_size * 0.93)

    # Phase 5: Final high-precision polish
    if time_left() > 5:
        current_h = best_h.copy()
        current_score = best_score
        step_size = 0.008
        
        for _ in range(400):
            if deadline_reached():
                break
            num_perturb = rng.integers(1, 4)
            indices = rng.choice(N, num_perturb, replace=False)
            noise = rng.normal(0, step_size, num_perturb)
            new_h = current_h.copy()
            new_h[indices] = np.maximum(current_h[indices] + noise, 0.0)
            new_score = compute_score(new_h)
            
            if new_score < current_score:
                current_h = new_h
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_h = current_h.copy()
                step_size = min(0.015, step_size * 1.02)
            else:
                step_size = max(0.0003, step_size * 0.9)

    best_h = np.maximum(best_h, 0.0)
    return list(best_h)

if __name__ == "__main__":
    res = construct()
    assert len(res) == N
    assert all(h >= 0 for h in res)
    print("Constructed optimized step function.")
    print(f"Score: {compute_score(np.array(res))}")
