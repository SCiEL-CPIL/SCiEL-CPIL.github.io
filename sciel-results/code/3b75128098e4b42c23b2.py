import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 130.0  # Safety margin before 150s kill

def _elapsed(start):
    return time.time() - start

def _time_left(start):
    return DEADLINE - _elapsed(start)

def _autoconv(h):
    return np.convolve(h, h, mode='full')

def _compute_ratio(h):
    """Compute the actual ratio: ||h||^2_2 / (||h||_1 * ||h*h||_inf)."""
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    l2_sq = np.sum(h ** 2)
    conv = _autoconv(h)
    l_inf = np.max(conv)
    if l1 * l_inf < 1e-15:
        return 0.0
    return l2_sq / (l1 * l_inf)

def _dft_param(z):
    """Map complex z (length N) to nonnegative h = |FFT(z)|^2, normalized."""
    H = np.fft.fft(z)
    h = np.abs(H) ** 2
    s = np.sum(h)
    if s < 1e-15:
        return np.ones(N) / N
    return h / s

def _ratio_from_z(z, beta=100.0):
    """Compute smoothed ratio from DFT parameterization."""
    h = _dft_param(z)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0, np.zeros(2 * N)
    l2_sq = np.sum(h ** 2)
    conv = _autoconv(h)
    max_val = np.max(conv)
    if max_val < 1e-15:
        return 0.0, np.zeros(2 * N)
    
    # Soft-max for smooth max
    shifted = beta * (conv - max_val)
    max_shifted = np.max(shifted)
    exp_shifted = np.exp(shifted - max_shifted)
    log_sum_exp = max_shifted + np.log(np.sum(exp_shifted))
    soft_max = (log_sum_exp + beta * max_val) / beta
    
    if soft_max < 1e-15:
        return 0.0, np.zeros(2 * N)
    
    ratio = l2_sq / soft_max
    return ratio, h

def _spectral_flatness(z, beta=50.0):
    """
    Objective: minimize peak/mean of autoconvolution.
    This is a proxy that directly targets flattening the autoconvolution peak.
    Returns (objective_value, gradient) for minimization.
    """
    z_c = z[:N] + 1j * z[N:]
    h = _dft_param(z_c)
    conv = _autoconv(h)
    mean_conv = np.mean(conv)
    if mean_conv < 1e-15:
        return 0.0, np.zeros(2 * N)
    
    # Soft-max of conv / mean_conv
    shifted = beta * (conv - np.max(conv))
    max_shifted = np.max(shifted)
    exp_shifted = np.exp(shifted - max_shifted)
    log_sum_exp = max_shifted + np.log(np.sum(exp_shifted))
    soft_max = (log_sum_exp + beta * np.max(conv)) / beta
    
    obj = soft_max / mean_conv
    
    # Gradient via finite differences on z
    eps = 1e-6
    grad = np.zeros(2 * N)
    for i in range(N):
        z_plus = z.copy()
        z_plus[i] += eps
        obj_plus, _ = _spectral_flatness(z_plus, beta)
        grad[i] = (obj_plus - obj) / eps
        
        z_plus = z.copy()
        z_plus[N + i] += eps
        obj_plus, _ = _spectral_flatness(z_plus, beta)
        grad[N + i] = (obj_plus - obj) / eps
    
    return obj, grad

def _ratio_objective(z, beta=100.0):
    """
    Negative smoothed ratio for minimization over DFT parameters.
    Uses analytic gradient through chain rule.
    """
    z_c = z[:N] + 1j * z[N:]
    H = np.fft.fft(z_c)
    h_raw = np.abs(H) ** 2
    S = np.sum(h_raw)
    if S < 1e-15:
        return 0.0, np.zeros(2 * N)
    h = h_raw / S
    
    l1 = np.sum(h)  # = 1 since normalized
    l2_sq = np.sum(h ** 2)
    conv = _autoconv(h)
    max_val = np.max(conv)
    if max_val < 1e-15:
        return 0.0, np.zeros(2 * N)
    
    # Soft-max
    shifted = beta * (conv - max_val)
    max_shifted = np.max(shifted)
    exp_shifted = np.exp(shifted - max_shifted)
    log_sum_exp = max_shifted + np.log(np.sum(exp_shifted))
    soft_max = (log_sum_exp + beta * max_val) / beta
    
    if soft_max < 1e-15:
        return 0.0, np.zeros(2 * N)
    
    ratio = l2_sq / soft_max
    
    # Analytic gradient via chain rule
    # d(h_k)/d(z_m) and d(h_k)/d(conj(z_m))
    # h_k = |H_k|^2 / S where H = FFT(z)
    # dH_k/dz_m = omega^(km)/N * ... (DFT matrix element)
    # This gets complex; use numerical gradient for reliability
    
    eps = 1e-6
    grad = np.zeros(2 * N)
    for i in range(N):
        z_plus = z.copy()
        z_plus[i] += eps
        r_plus, _ = _ratio_from_z(z_plus[:N] + 1j * z_plus[N:], beta)
        grad[i] = (r_plus - ratio) / eps
        
        z_plus = z.copy()
        z_plus[N + i] += eps
        r_plus, _ = _ratio_from_z(z_plus[:N] + 1j * z_plus[N:], beta)
        grad[N + i] = (r_plus - ratio) / eps
    
    return -ratio, -grad

def _local_optimize_h(h0, start, max_iter=300):
    """Direct optimization of h with non-negativity."""
    h = np.maximum(h0, 0.0).copy()
    
    def obj(h_flat):
        h_flat = np.maximum(h_flat, 0.0)
        r = _compute_ratio(h_flat)
        return -r
    
    result = minimize(obj, h, method='L-BFGS-B',
                      bounds=[(0, None)] * N,
                      options={'maxiter': max_iter, 'gtol': 1e-9, 'maxls': 30})
    h_opt = np.maximum(result.x, 0.0)
    return h_opt, _compute_ratio(h_opt)

def _gaussian_profile(center, width, height=1.0):
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    return height * np.exp(-((x - center) ** 2) / (2 * width ** 2))

def _lorentzian_profile(center, gamma, height=1.0):
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    return height * (gamma ** 2) / ((x - center) ** 2 + gamma ** 2)

def _two_gaussian(c1, w1, c2, w2, h1=1.0, h2=1.0):
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    return h1 * np.exp(-((x - c1) ** 2) / (2 * w1 ** 2)) + h2 * np.exp(-((x - c2) ** 2) / (2 * w2 ** 2))

def construct():
    start_time = time.time()
    
    best_h = np.ones(N) / N
    best_score = _compute_ratio(best_h)
    
    # Generate diverse initializations
    inits = []
    inits.append(np.ones(N))
    
    for center in [-0.2, -0.1, 0.0, 0.1, 0.2]:
        for width in [0.03, 0.05, 0.08, 0.12, 0.18]:
            inits.append(_gaussian_profile(center, width))
    
    for center in [-0.15, 0.0, 0.15]:
        for gamma in [0.03, 0.06, 0.1, 0.15]:
            inits.append(_lorentzian_profile(center, gamma))
    
    for c1, c2 in [(-0.15, 0.15), (-0.1, 0.1), (-0.2, 0.2), (-0.25, 0.25)]:
        for w in [0.04, 0.07, 0.1]:
            inits.append(_two_gaussian(c1, w, c2, w))
    
    # Random
    rng = np.random.RandomState(123)
    for _ in range(15):
        inits.append(rng.random(N))
    
    # Sparse
    for _ in range(10):
        h = np.zeros(N)
        idx = rng.choice(N, rng.randint(3, 10), replace=False)
        h[idx] = rng.random(len(idx)) + 0.1
        inits.append(h)
    
    # Score and sort
    scored = []
    for h_init in inits:
        h_init = np.maximum(h_init, 0.0)
        s = _compute_ratio(h_init)
        scored.append((s, h_init))
    scored.sort(key=lambda x: -x[0])
    
    if scored[0][0] > best_score:
        best_score = scored[0][0]
        best_h = scored[0][1].copy()
    
    # Phase 1: Spectral flatness pre-pass on top candidates
    top_k = min(10, len(scored))
    for i in range(top_k):
        if _time_left(start_time) < 15.0:
            break
        
        s_init, h_init = scored[i]
        
        # Convert to DFT parameterization
        H = np.fft.fft(h_init)
        mags = np.abs(H)
        phases = np.angle(H)
        z_init = mags * np.exp(1j * phases)
        z_flat = np.concatenate([z_init.real, z_init.imag])
        
        # Spectral flatness optimization
        try:
            res = minimize(_spectral_flatness, z_flat, method='L-BFGS-B',
                          options={'maxiter': 150, 'gtol': 1e-8, 'maxls': 20})
            z_opt = res.x[:N] + 1j * res.x[N:]
            h_opt = _dft_param(z_opt)
            s_opt = _compute_ratio(h_opt)
            if s_opt > best_score:
                best_score = s_opt
                best_h = h_opt.copy()
        except Exception:
            pass
        
        if _time_left(start_time) < 10.0:
            break
    
    # Phase 2: Full ratio optimization from best candidates
    candidates = [best_h.copy()]
    for i in range(min(8, len(scored))):
        candidates.append(scored[i][1].copy())
    
    for h_cand in candidates:
        if _time_left(start_time) < 8.0:
            break
        
        # Direct h optimization
        h_opt, s_opt = _local_optimize_h(h_cand, start_time, max_iter=200)
        if s_opt > best_score:
            best_score = s_opt
            best_h = h_opt.copy()
        
        if _time_left(start_time) < 5.0:
            break
        
        # DFT-based ratio optimization
        H = np.fft.fft(h_opt if s_opt > _compute_ratio(h_cand) else h_cand)
        mags = np.abs(H)
        phases = np.angle(H)
        z_init = mags * np.exp(1j * phases)
        z_flat = np.concatenate([z_init.real, z_init.imag])
        
        try:
            res = minimize(_ratio_objective, z_flat, method='L-BFGS-B',
                          options={'maxiter': 200, 'gtol': 1e-9, 'maxls': 25})
            z_opt = res.x[:N] + 1j * res.x[N:]
            h_dft = _dft_param(z_opt)
            s_dft = _compute_ratio(h_dft)
            if s_dft > best_score:
                best_score = s_dft
                best_h = h_dft.copy()
        except Exception:
            pass
    
    # Phase 3: Final polish
    if _time_left(start_time) > 10.0:
        h_polish, s_polish = _local_optimize_h(best_h, start_time, max_iter=400)
        if s_polish > best_score:
            best_score = s_polish
            best_h = h_polish.copy()
        
        # DFT polish
        if _time_left(start_time) > 5.0:
            H = np.fft.fft(best_h)
            mags = np.abs(H)
            phases = np.angle(H)
            z_init = mags * np.exp(1j * phases)
            z_flat = np.concatenate([z_init.real, z_init.imag])
            try:
                res = minimize(_ratio_objective, z_flat, method='L-BFGS-B',
                              options={'maxiter': 300, 'gtol': 1e-10, 'maxls': 30})
                z_opt = res.x[:N] + 1j * res.x[N:]
                h_final = _dft_param(z_opt)
                s_final = _compute_ratio(h_final)
                if s_final > best_score:
                    best_score = s_final
                    best_h = h_final.copy()
            except Exception:
                pass
    
    # Ensure non-negative
    best_h = np.maximum(best_h, 0.0)
    if np.sum(best_h) < 1e-15:
        best_h = np.ones(N) / N
    
    return list(best_h)
