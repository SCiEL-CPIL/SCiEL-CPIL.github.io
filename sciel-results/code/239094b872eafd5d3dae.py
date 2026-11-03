"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Uses projected gradient descent with log-sum-exp smoothing, SLSQP polish,
and multiple diverse initializations for robust basin exploration."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 110  # Safety margin before 120s kill


def _autocorr(h):
    """
    Compute linear autocorrelation r_k = sum_{i=k}^{N-1} h_i * h_{i-k} for k=0..N-1.
    Uses FFT for O(N log N) computation.
    """
    n = len(h)
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    return corr[:n]


def _overlap_values(h):
    """
    Compute overlap values C_k = (2/N) * (N/2 - r_k) for k=0..N-1.
    """
    n = len(h)
    r = _autocorr(h)
    return (2.0 / n) * (n / 2.0 - r)


def _score(h):
    """
    Compute the maximum linear overlap.
    """
    h = np.asarray(h, dtype=np.float64)
    cs = _overlap_values(h)
    return float(np.max(cs))


def _smooth_max_value(h, beta):
    """
    Smooth maximum of overlap values using log-sum-exp.
    """
    h = np.asarray(h, dtype=np.float64)
    cs = _overlap_values(h)
    m = np.max(cs)
    return m + np.log(np.sum(np.exp(beta * (cs - m)))) / beta


def _smooth_max_grad(h, beta):
    """
    Gradient of the smooth maximum objective.
    Uses FFT to compute the gradient in O(N log N).
    Returns (value, gradient)
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _overlap_values(h)
    m = np.max(cs)
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp  # Weights for soft max
    
    val = m + np.log(sum_exp) / beta
    
    # grad_j = -(2/n) * sum_k w_k * (h_{j-k} + h_{j+k})
    # This is -(2/n) * (w * h + w_rev * h) where * is linear autocorrelation
    # Use FFT: if A = fft(w_pad), B = fft(h_pad), then corr = ifft(A * conj(B))
    n_fft = 2 * n
    w_pad = np.zeros(n_fft)
    w_pad[:n] = w
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    
    W = np.fft.rfft(w_pad, n=n_fft)
    H = np.fft.rfft(h_pad, n=n_fft)
    
    # Linear cross-correlation: sum_k w_k * h_{j+k} and sum_k w_k * h_{j-k}
    # corr1[j] = sum_k w_k * h_{(j+k) mod ...} -- use full linear
    # Actually: grad_j = -(2/n) * [sum_{k=0}^{n-1} w_k * h_{j+k} (if j+k<n) + sum_{k=0}^{n-1} w_k * h_{j-k} (if j>=k)]
    # = -(2/n) * [ (w * h)_j + (w_rev * h)_j ] where * is linear convolution
    # (w * h)_j = sum_k w_k * h_{j-k} for j-k in [0,n-1]
    # (w_rev * h)_j = sum_k w_k * h_{j+k} for j+k in [0,n-1]
    
    # Linear convolution via FFT
    conv_wh = np.fft.irfft(W * H, n=n_fft)[:n]  # (w*h)[j] = sum_k w[k]*h[j-k]
    # For w_rev * h: reverse w
    w_rev_pad = np.zeros(n_fft)
    w_rev_pad[:n] = w[::-1]
    W_rev = np.fft.rfft(w_rev_pad, n=n_fft)
    conv_wh_rev = np.fft.irfft(W_rev * H, n=n_fft)[:n]  # (w_rev*h)[j] = sum_k w[n-1-k]*h[j-k]
    
    # Actually let me think more carefully.
    # grad_j = -(2/n) * sum_k w_k * [h_{j-k} (if 0<=j-k<n) + h_{j+k} (if 0<=j+k<n)]
    # Term 1: sum_k w_k * h_{j-k} = (w * h)_j where * is linear conv (w padded, h padded)
    #   This is exactly conv_wh[j] from above.
    # Term 2: sum_k w_k * h_{j+k} = sum_k w_k * h_{(j+k)}
    #   Let l = j+k, then k = l-j, sum_l w_{l-j} * h_l (where l-j in [0,n-1] and l in [0,n-1])
    #   = sum_l w_{l-j} * h_l = (w_rev * h)_{n-1-j} ... hmm this is getting complex.
    #   Alternative: sum_k w_k * h_{j+k} = (w * h_rev)_{n-1-j} where h_rev = h[::-1]
    #   Or more simply: it's the cross-correlation of w and h at shift -j.
    #   Actually: sum_k w_k * h_{j+k} = sum_k w_{j+k} * h_k reversed... 
    #   
    # Let me just use: sum_k w_k * h_{j+k} for valid indices.
    # This equals the linear cross-correlation (w ⊛ h)[j] where (w⊛h)[j] = sum_k w_k h_{j+k}
    # Cross-correlation via FFT: ifft(conj(W) * H) gives sum_k w_k h_{j+k}... 
    # Actually cross-correlation: (w ⊛ h)[j] = sum_k w_{k} h_{k+j}
    # FFT: ifft(conj(fft(w)) * fft(h)) -- but need to be careful with padding.
    
    # Let me just do it directly with the FFT approach:
    # sum_k w_k * h_{j+k}: this is the cross-correlation at lag j
    # = ifft(conj(W) * H) evaluated appropriately
    
    # cross_corr[j] = sum_k w_k * h_{k+j} for k+j < n
    # Using FFT with zero padding:
    # cross = ifft(conj(W) * H, n=n_fft)
    # cross[j] = sum_{k=0}^{n-1} w_k * h_{k+j} (with h=0 outside [0,n-1])
    
    cross = np.fft.irfft(np.conj(W) * H, n=n_fft)
    # cross[j] = sum_k w_k * h_{k+j} for j=0..n-1
    # We need sum_k w_k * h_{j+k} = cross[j] (same thing, just renamed index)
    
    # So grad_j = -(2/n) * (conv_wh[j] + cross[j])
    # Wait: conv_wh[j] = sum_k w[k] * h[j-k] (standard linear convolution)
    # cross[j] = sum_k w[k] * h[k+j]
    # Together: sum_k w_k (h_{j-k} + h_{j+k}) -- yes!
    
    grad = -(2.0 / n) * (conv_wh + cross[:n])
    
    return val, grad


def _epigraph_objective(x, beta):
    """
    Epigraph formulation: minimize t subject to smooth_max(h) <= t,
    with x = [h_0, ..., h_{N-1}, t].
    """
    h = x[:N]
    t = x[N]
    val, _ = _smooth_max_grad(h, beta)
    return t + 0.01 * max(0, val - t)  # Penalize constraint violation


def _epigraph_grad(x, beta):
    """
    Gradient of epigraph objective.
    """
    h = x[:N]
    t = x[N]
    val, grad_h = _smooth_max_grad(h, beta)
    
    grad = np.zeros(len(x))
    if val > t:
        grad[:N] = grad_h * 0.01
        grad[N] = 1.0 + 0.01
    else:
        grad[:N] = grad_h * 0.0
        grad[N] = 1.0
    
    return grad


def _fix_sum(h):
    """
    Adjust h to have sum exactly N/2, keeping values in [0, 1].
    """
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0
    diff = target - np.sum(h)
    h += diff / N
    h = np.clip(h, 0.0, 1.0)
    
    # Second pass if clipping changed sum
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        mask = (h > 0) & (h < 1)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
            
    # Final tiny correction
    diff = target - np.sum(h)
    if abs(diff) > 1e-15:
        if diff > 0:
            idx = np.argmin(h)
        else:
            idx = np.argmax(h)
        h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
        
    return h


def _project_sum_box(h):
    """
    Project h onto the set {x : sum(x) = N/2, 0 <= x_i <= 1}.
    Uses exact bisection on the Lagrange multiplier.
    """
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0
    
    # Find lambda such that sum(clip(h - lambda, 0, 1)) = target
    # f(lambda) = sum(clip(h_i - lambda, 0, 1)) is monotonically decreasing in lambda
    # f(+inf) = 0, f(-inf) = N
    # We need f(lambda) = target
    
    def f(lam):
        return np.sum(np.clip(h - lam, 0.0, 1.0))
    
    # Bisection
    lo, hi = -1.0, 1.0
    # Expand bounds if needed
    while f(lo) < target:
        lo -= 1.0
    while f(hi) > target:
        hi += 1.0
    
    for _ in range(80):
        mid = (lo + hi) / 2.0
        fm = f(mid)
        if abs(fm - target) < 1e-15:
            break
        if fm > target:
            lo = mid
        else:
            hi = mid
    
    lam = (lo + hi) / 2.0
    h = np.clip(h - lam, 0.0, 1.0)
    
    # Tiny correction for floating point
    diff = target - np.sum(h)
    if abs(diff) > 1e-15:
        mask = (h > 0) & (h < 1.0)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
    
    return h


def _pgd_optimize(h0, beta, max_iter=500, lr_schedule='adaptive', tol=1e-10):
    """
    Nesterov-accelerated projected gradient descent with adaptive step size.
    Minimizes smooth_max(h) subject to sum(h)=N/2, 0<=h<=1.
    Optimized for speed: reduced iterations, simplified line search.
    """
    h = _fix_sum(h0.copy())
    best_h = h.copy()
    best_val = _score(h)
    
    # Nesterov momentum
    y = h.copy()
    momentum = 0.0
    prev_momentum = 0.0
    
    # Fixed learning rate for speed and stability
    lr = 0.02
    prev_val = _smooth_max_value(h, beta)
    
    for it in range(max_iter):
        if time.time() > _DEADLINE - 5:
            break
        
        # Nesterov look-ahead
        if it > 0:
            momentum = (1 + np.sqrt(1 + 4 * prev_momentum**2)) / 2
            y = h + (momentum - 1) / momentum * (h - y)
        
        val, grad = _smooth_max_grad(y, beta)
        
        # Gradient step
        h_new = h - lr * grad
        
        # Project onto feasible set
        h_new = _project_sum_box(h_new)
        
        new_val = _smooth_max_value(h_new, beta)
        
        if new_val < prev_val:
            prev_momentum = momentum
            h = h_new
            prev_val = new_val
            s = _score(h)
            if s < best_val:
                best_val = s
                best_h = h.copy()
        else:
            # Simple line search: reduce step size once
            lr *= 0.5
            h_new = h - lr * grad
            h_new = _project_sum_box(h_new)
            new_val = _smooth_max_value(h_new, beta)
            if new_val < prev_val:
                prev_momentum = momentum
                h = h_new
                prev_val = new_val
                s = _score(h)
                if s < best_val:
                    best_val = s
                    best_h = h.copy()
        
        # Convergence check (using val from gradient step to avoid extra FFT)
        if it > 100 and abs(prev_val - val) < tol:
            break
    
    return best_h


def _run_slsqp(h0, beta, maxiter, ftol, gtol):
    """
    Run SLSQP optimization on the smooth max objective.
    """
    cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0,
             'jac': lambda x: np.ones(N)}]
    bounds = [(0.0, 1.0)] * N
    
    def obj_func(x):
        val, _ = _smooth_max_grad(x, beta)
        return val
        
    def jac_func(x):
        _, grad = _smooth_max_grad(x, beta)
        return grad
        
    try:
        res = minimize(obj_func, h0.copy(), jac=jac_func, method='SLSQP',
                       bounds=bounds, constraints=cons,
                       options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol})
        return res.x
    except Exception:
        return h0


def _polish(h, beta, maxiter, ftol, gtol):
    """
    High-precision polish pass with tighter tolerances.
    """
    h0 = h.copy()
    h_new = _run_slsqp(h0, beta, maxiter, ftol, gtol)
    h_new = _fix_sum(h_new)
    
    # Verify improvement
    if _score(h_new) < _score(h0):
        return h_new
    return h0


def _spectral_h_from_coeffs(coeffs):
    """
    Reconstruct h from real DFT coefficients (modes 1..K).
    coeffs = [a1, b1, a2, b2, ..., aK, bK]
    h_i = 0.5 + sum_{k=1}^K (a_k cos(2*pi*k*i/N) + b_k sin(2*pi*k*i/N))
    """
    n_modes = len(coeffs) // 2
    h = np.full(N, 0.5)
    i = np.arange(N)
    for k in range(1, n_modes + 1):
        a = coeffs[2*(k-1)]
        b = coeffs[2*(k-1)+1]
        h += a * np.cos(2.0 * np.pi * k * i / N)
        h += b * np.sin(2.0 * np.pi * k * i / N)
    return h


def _h_to_spectral_coeffs(h, K):
    """
    Extract the first K real DFT coefficients from h.
    Returns [a1, b1, ..., aK, bK] where h ~ 0.5 + sum (a_k cos + b_k sin)
    """
    H = np.fft.rfft(h)
    out = []
    for k in range(1, K + 1):
        a = 2.0 / N * H[k].real
        b = -2.0 / N * H[k].imag
        out.append(a)
        out.append(b)
    return out


def _optimize_spectral(h_init, K, beta, maxiter):
    """
    Optimize the spectral coefficients of h to minimize the smooth max objective.
    """
    coeffs0 = _h_to_spectral_coeffs(h_init, K)
    
    def obj(coeffs):
        h = _spectral_h_from_coeffs(coeffs)
        
        # Penalty for out-of-bounds to keep solution valid
        penalty = 0.0
        if np.any(h < 0) or np.any(h > 1):
            penalty = np.sum(np.maximum(0, -h)**2) + np.sum(np.maximum(0, h-1)**2)
        
        val, _ = _smooth_max_grad(h, beta)
        return val + 1.0 * penalty

    try:
        res = minimize(obj, coeffs0, method='L-BFGS-B', 
                       options={'maxiter': maxiter, 'ftol': 1e-12})
        if res.success:
            h_opt = _spectral_h_from_coeffs(res.x)
            return h_opt
    except Exception:
        pass
    return h_init


def _coordinate_descent_polish(h, beta, max_sweeps=20, tol=1e-12):
    """
    Fast coordinate descent polish. Reduced sweeps and larger steps for speed.
    """
    h = _fix_sum(h.copy())
    active_mask = (h > 1e-10) & (h < 1.0 - 1e-10)
    active_idx = np.where(active_mask)[0]
    
    if len(active_idx) < 2:
        return h
    
    best_val = _score(h)
    best_h = h.copy()
    
    for sweep in range(max_sweeps):
        if time.time() > _DEADLINE - 3:
            break
        improved = False
        for idx in active_idx:
            if time.time() > _DEADLINE - 3:
                break
            # Try fewer, larger perturbations
            for delta in [1e-3, -1e-3, 5e-4, -5e-4]:
                h_trial = h.copy()
                h_trial[idx] += delta
                h_trial = _fix_sum(h_trial)
                if np.any(h_trial < 0) or np.any(h_trial > 1):
                    continue
                s = _score(h_trial)
                if s < best_val - 1e-14:
                    h = h_trial
                    best_val = s
                    best_h = h.copy()
                    improved = True
                    break
        if not improved:
            break
    
    return best_h


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    
    def _update_best(h):
        nonlocal best_h, best_score
        h = _fix_sum(h)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
    
    # 1. Generate diverse initializations (Reduced count for speed)
    i = np.arange(N)
    initializations = []
    
    # Cosine
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    initializations.append(_fix_sum(h_cos))
    
    # Sine
    h_sin = 0.5 + 0.5 * np.sin(2.0 * np.pi * i / N)
    initializations.append(_fix_sum(h_sin))
    
    # Alternating
    h_alt = np.where(i % 2 == 0, 0.7, 0.3)
    initializations.append(_fix_sum(h_alt))
    
    # Random (2 starts)
    for _ in range(2):
        h_rand = np.random.uniform(0, 1, N)
        initializations.append(_fix_sum(h_rand))
    
    # Block pattern (1 start)
    h_block = np.zeros(N)
    for start in range(0, N, 20):
        h_block[start:start+10] = 0.8
        h_block[start+10:start+20] = 0.2
    initializations.append(_fix_sum(h_block))
    
    # 2. Run Nesterov PGD from each initialization
    for h_init in initializations:
        if time.time() > _DEADLINE - 30:
            break
        h_pgd = _pgd_optimize(h_init, beta=50.0, max_iter=500)
        _update_best(h_pgd)
    
    # 3. SLSQP polish on best
    if best_h is not None and time.time() < _DEADLINE - 20:
        h_slsqp = _run_slsqp(best_h, beta=50.0, maxiter=100, ftol=1e-10, gtol=1e-8)
        _update_best(h_slsqp)
    
    # 4. Higher beta PGD polish
    if best_h is not None and time.time() < _DEADLINE - 10:
        h_pgd_high = _pgd_optimize(best_h, beta=100.0, max_iter=500)
        _update_best(h_pgd_high)
    
    # 5. SLSQP with higher beta
    if best_h is not None and time.time() < _DEADLINE - 5:
        h_slsqp_high = _run_slsqp(best_h, beta=100.0, maxiter=100, ftol=1e-12, gtol=1e-10)
        _update_best(h_slsqp_high)
    
    # 6. Coordinate descent polish on active set
    if best_h is not None and time.time() < _DEADLINE - 2:
        h_cd = _coordinate_descent_polish(best_h, beta=200.0, max_sweeps=20)
        _update_best(h_cd)
    
    if best_h is None:
        best_h = np.full(N, 0.5)
    
    best_h = _fix_sum(best_h)
    return best_h
