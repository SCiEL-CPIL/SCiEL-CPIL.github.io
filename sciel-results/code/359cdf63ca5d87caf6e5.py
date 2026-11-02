import numpy as np
import time
from scipy.optimize import minimize

N = 400
START_TIME = time.time()
TIME_LIMIT = 145
SAFETY_MARGIN = 5

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - SAFETY_MARGIN)

def _psd_to_x(P):
    """Convert non-negative PSD (length L//2+1) to real sequence x of length N with sum=N."""
    L = 2 * N
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    # P is length L_fft//2+1
    X = np.sqrt(np.maximum(P, 0.0))
    # Build full Hermitian spectrum
    X_full = np.zeros(L_fft, dtype=complex)
    X_full[:L_fft // 2 + 1] = X
    for k in range(1, L_fft // 2):
        X_full[L_fft - k] = np.conj(X[k])
    x = np.fft.irfft(X_full, n=L_fft)[:N]
    s = np.sum(x)
    if abs(s) > 1e-12:
        x = x * (N / s)
    return x

def _psd_objective_and_grad(P, beta=100.0):
    """
    Objective: max|ac| / s^2 where ac = IFFT(|FFT(x)|^2), x = _psd_to_x(P).
    Gradient w.r.t. P via chain rule.
    """
    L_fft = 1
    while L_fft < 2 * N:
        L_fft *= 2
    
    P_clipped = np.maximum(P, 0.0)
    x = _psd_to_x(P_clipped)
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros_like(P)
    
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    max_abs = np.max(abs_ac)
    
    if max_abs < 1e-15:
        return 0.0, np.zeros_like(P)
    
    # Softmax for smooth max
    shifted = beta * (abs_ac - max_abs)
    log_sum_exp = max_abs + (1.0 / beta) * np.log(np.sum(np.exp(shifted)))
    val = log_sum_exp / (s * s)
    
    # Gradient: d(val)/dP
    # ac = x*x (linear conv), x depends on P through sqrt(P)
    # d(x)/dP[k] = (1/(2*sqrt(P[k]))) * d(x)/d(X[k]) where X[k]=sqrt(P[k])
    # This is complex; use automatic differentiation via finite differences for gradient
    # For speed, use a simpler approach: gradient of softmax(|ac|) w.r.t. x, then chain to P
    
    # d(softmax)/dx computed as before
    weights = np.exp(shifted) / np.sum(np.exp(shifted))
    g = weights * np.sign(ac)
    
    # d(sum_k g[k]*ac[k])/dx = 2 * IFFT(FFT(g_padded) * FFT(x_padded))
    L_conv = 1
    while L_conv < 3 * N:
        L_conv *= 2
    G = np.fft.rfft(g, n=L_conv)
    Xf = np.fft.rfft(x, n=L_conv)
    term1 = np.fft.irfft(np.conj(G) * Xf, n=L_conv)[:N]
    term2 = np.fft.irfft(G * Xf, n=L_conv)[:N]
    d_softmax_dx = beta * (term1 + term2)
    
    d_val_dx = (1.0 / (s * s)) * d_softmax_dx - (2.0 * log_sum_exp / (s ** 3)) * np.ones(N)
    
    # Now chain: d(val)/dP[k] = sum_j d(val)/dx[j] * dx[j]/dP[k]
    # x[j] = (1/L_fft) * sum_m X[m] * exp(2pi i j m / L_fft)
    # X[m] = sqrt(P[m]) for m in [0, L_fft//2], X[L_fft-m] = conj(X[m])
    # dx[j]/dP[k] = (1/(2*sqrt(P[k]))) * (1/L_fft) * [exp(2pi i j k / L_fft) + exp(-2pi i j k / L_fft)]
    #            = (1/(sqrt(P[k]) * L_fft)) * cos(2pi j k / L_fft)
    
    # This is O(N * L_fft) which is too slow for large L_fft.
    # Alternative: compute gradient in frequency domain.
    # d(val)/dX[m] = sum_j d(val)/dx[j] * dx[j]/dX[m]
    # dx[j]/dX[m] = (1/L_fft) * exp(2pi i j m / L_fft)
    # So d(val)/dX[m] = (1/L_fft) * sum_j d(val)/dx[j] * exp(2pi i j m / L_fft) = (1/L_fft) * FFT(d_val_dx)[m]
    
    D = np.fft.rfft(d_val_dx, n=L_fft)
    # d(val)/dX[m] = D[m] / L_fft for m in [0, L_fft//2]
    d_val_dX = D / L_fft
    
    # d(val)/dP[k] = d(val)/dX[k] * dX[k]/dP[k] = d_val_dX[k] * (1/(2*sqrt(P[k])))
    P_safe = np.maximum(P_clipped, 1e-10)
    grad_P = d_val_dX[:P.shape[0]] / (2.0 * np.sqrt(P_safe))
    
    # Project gradient to only where P > 0
    grad_P[P_clipped <= 0] = 0.0
    
    return val, grad_P

def _psd_optimize(P0, max_iter=200, lr=0.01, beta=100.0):
    """Projected gradient descent on PSD."""
    P = P0.copy()
    best_P = P.copy()
    best_val = _psd_objective_and_grad(P, beta)[0]
    
    for it in range(max_iter):
        if _check_deadline():
            break
        
        val, grad = _psd_objective_and_grad(P, beta)
        if val < best_val:
            best_val = val
            best_P = P.copy()
        
        # Projected gradient step
        P_new = P - lr * grad
        P_new = np.maximum(P_new, 0.0)
        
        # Normalize to keep same energy scale
        P_new = P_new * (np.sum(P) / (np.sum(P_new) + 1e-10))
        
        P = P_new
    
    return best_P, best_val

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _objective(x):
    """Compute max(|autoconv|) / (sum x)^2."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    ac = _autoconv_fft(x)
    return np.max(np.abs(ac)) / (s * s)

def _objective_grad_smooth(x, beta=50.0):
    """
    Compute smooth soft-max surrogate objective and its gradient.
    Uses log-sum-exp for numerical stability.
    
    The soft-max approximates max_k |ac[k]| as:
    (1/beta) * log(sum_k exp(beta * |ac[k]|))
    
    Gradient is well-defined and smooth, enabling better optimization.
    """
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros(N)
    
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    
    # Soft-max: (1/beta) * log(sum exp(beta * |ac|))
    max_abs = np.max(abs_ac)
    if max_abs < 1e-15:
        return 0.0, np.zeros(N)
    
    # Numerically stable log-sum-exp
    shifted = beta * (abs_ac - max_abs)
    log_sum_exp = max_abs + (1.0 / beta) * np.log(np.sum(np.exp(shifted)))
    
    val = log_sum_exp / (s * s)
    
    # Gradient of soft-max w.r.t. x
    # d/dx [log(sum exp(beta*|ac_k|))] = (1/sum exp(beta*|ac_k|)) * sum exp(beta*|ac_k|) * beta * sign(ac_k) * d(ac_k)/dx
    # = beta * sum_k [exp(beta*|ac_k|) / sum_j exp(beta*|ac_j|)] * sign(ac_k) * d(ac_k)/dx
    
    # Compute weights (softmax over |ac|)
    weights = np.exp(shifted) / np.sum(np.exp(shifted))
    
    # Build gradient using FFT
    L_fft = 1
    while L_fft < 3 * N:
        L_fft *= 2
    
    # g[k] = weights[k] * sign(ac[k])
    g = weights * np.sign(ac)
    
    G = np.fft.rfft(g, n=L_fft)
    Xf = np.fft.rfft(x, n=L_fft)
    
    # d(sum_k g[k]*ac[k])/dx = sum_k g[k] * d(ac[k])/dx
    # d(ac[k])/dx[j] involves convolution structure
    # Using FFT: if ac = x * x (linear conv), then d(ac[k])/dx = 2 * (x convolved with delta_k)
    # More precisely: d(ac[k])/dx[j] = x[k-j] + x[j+k] (with valid indices)
    # In frequency domain: this is equivalent to the gradient of x^T T_k x
    
    # The gradient of sum_k g[k] * ac[k] w.r.t. x is:
    # 2 * IFFT(FFT(g) * FFT(x)) truncated to N, where g is padded to 2N-1
    term1 = np.fft.irfft(np.conj(G) * Xf, n=L_fft)[:N]
    term2 = np.fft.irfft(G * Xf, n=L_fft)[:N]
    
    d_softmax_dx = beta * (term1 + term2)
    
    # Full gradient: d(val)/dx = (1/s^2) * d_softmax_dx - (2*softmax/s^3) * ones
    grad = (1.0 / (s * s)) * d_softmax_dx - (2.0 * log_sum_exp / (s ** 3)) * np.ones(N)
    
    return val, grad

def _epigraph_objective(x, t, beta=50.0):
    """
    Epigraph formulation: minimize t subject to soft_max(|ac|) <= t * s^2
    We use a penalty: t + lambda * max(0, soft_max(|ac|) - t*s^2)^2
    """
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros(len(x))
    
    ac = _autoconv_fft(x[:N])
    abs_ac = np.abs(ac)
    max_abs = np.max(abs_ac)
    
    if max_abs < 1e-15:
        return t, np.zeros(len(x))
    
    shifted = beta * (abs_ac - max_abs)
    log_sum_exp = max_abs + (1.0 / beta) * np.log(np.sum(np.exp(shifted)))
    
    constraint_violation = max(0.0, log_sum_exp - t * s * s)
    val = t + 100.0 * constraint_violation ** 2
    
    # Gradient
    if constraint_violation > 0:
        weights = np.exp(shifted) / np.sum(np.exp(shifted))
        g = weights * np.sign(ac)
        
        L_fft = 1
        while L_fft < 3 * N:
            L_fft *= 2
        
        G = np.fft.rfft(g, n=L_fft)
        Xf = np.fft.rfft(x[:N], n=L_fft)
        
        term1 = np.fft.irfft(np.conj(G) * Xf, n=L_fft)[:N]
        term2 = np.fft.irfft(G * Xf, n=L_fft)[:N]
        
        d_softmax_dx = beta * (term1 + term2)
        
        grad_x = 200.0 * constraint_violation * ((1.0 / (s * s)) * d_softmax_dx - (2.0 * log_sum_exp / (s ** 3)) * np.ones(N))
        grad_t = 1.0 - 200.0 * constraint_violation * s * s
    else:
        grad_x = np.zeros(N)
        grad_t = 1.0
    
    return val, np.concatenate([grad_x, [grad_t]])

def _coordinate_descent(x0, max_rounds=200, beta=50.0):
    """
    Coordinate descent: optimize one coordinate at a time using the smooth surrogate.
    This is fast per iteration and can escape local minima.
    """
    best_x = x0.copy()
    best_val, _ = _objective_grad_smooth(best_x, beta)
    
    for rnd in range(max_rounds):
        if _check_deadline():
            break
        
        improved = False
        for j in range(N):
            if _check_deadline():
                break
            
            # Try small perturbations in coordinate j
            current = best_x[j]
            best_local = best_val
            best_local_j = current
            
            for delta in [-0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05]:
                trial = best_x.copy()
                trial[j] = current + delta
                val, _ = _objective_grad_smooth(trial, beta)
                if val < best_local:
                    best_local = val
                    best_local_j = current + delta
            
            if best_local < best_val - 1e-12:
                best_x[j] = best_local_j
                best_val = best_local
                improved = True
        
        if not improved:
            break
    
    return best_x, best_val

def _spectral_design():
    """
    Design x via its power spectrum to flatten the autoconvolution.
    
    Key insight: ac = IFFT(|FFT(x)|^2). To minimize max|ac|,
    we want |FFT(x)|^2 to be as flat as possible (white noise spectrum),
    which makes ac concentrated at lag 0 with minimal sidelobes.
    
    We parameterize x = IFFT(sqrt(P) * exp(i*phase)) and optimize P.
    """
    L_fft = 1
    while L_fft < 2 * N:
        L_fft *= 2
    
    # Start with flat power spectrum
    P = np.ones(L_fft // 2 + 1)
    
    # Add some structure: boost low frequencies slightly for smoothness
    freqs = np.arange(L_fft // 2 + 1)
    P = P * (1.0 + 0.3 * np.exp(-freqs / 50.0))
    
    # Random phase
    rng = np.random.default_rng(42)
    phase = rng.uniform(0, 2 * np.pi, size=L_fft // 2 + 1)
    phase[0] = 0  # DC component real
    
    # Construct x
    X_mag = np.sqrt(P)
    X = X_mag * np.exp(1j * phase)
    # Make Hermitian symmetric for real x
    X_full = np.zeros(L_fft, dtype=complex)
    X_full[:L_fft // 2 + 1] = X
    for k in range(1, L_fft // 2):
        X_full[L_fft - k] = np.conj(X[k])
    
    x = np.fft.irfft(X_full, n=L_fft)[:N]
    
    # Normalize to have sum = N (so average value = 1)
    s = np.sum(x)
    if abs(s) > 1e-10:
        x = x * (N / s)
    
    return x

def _signed_pattern_search(x0, max_rounds=50):
    """
    Explore sign patterns: flip signs of subsets of x to see if
    negative values help reduce max|ac|.
    """
    best_x = x0.copy()
    best_val = _objective(best_x)
    
    for rnd in range(max_rounds):
        if _check_deadline():
            break
        
        improved = False
        
        # Try flipping individual coordinates
        for j in range(0, N, 2):  # Step by 2 for speed
            if _check_deadline():
                break
            
            trial = best_x.copy()
            trial[j] = -trial[j]
            val = _objective(trial)
            if val < best_val - 1e-12:
                best_x = trial
                best_val = val
                improved = True
        
        # Try flipping pairs
        if not improved:
            for j in range(0, N - 1, 3):
                if _check_deadline():
                    break
                
                trial = best_x.copy()
                trial[j] = -trial[j]
                trial[j + 1] = -trial[j + 1]
                val = _objective(trial)
                if val < best_val - 1e-12:
                    best_x = trial
                    best_val = val
                    improved = True
        
        if not improved:
            break
    
    return best_x, best_val

def construct():
    best_x = np.ones(N)
    best_val = _objective(best_x)
    
    if _check_deadline():
        return best_x
    
    # === PSD-SHAPING APPROACH (structural change) ===
    L_fft = 1
    while L_fft < 2 * N:
        L_fft *= 2
    half = L_fft // 2 + 1
    
    # Multiple PSD initializations
    rng_psd = np.random.default_rng(2024)
    psd_inits = []
    
    # Flat PSD
    psd_inits.append(np.ones(half))
    
    # Low-pass PSD
    freqs = np.arange(half)
    psd_inits.append(np.exp(-freqs / 30.0))
    psd_inits.append(np.exp(-freqs / 100.0))
    
    # Bandpass PSD
    psd_inits.append(1.0 + 0.5 * np.cos(2 * np.pi * freqs / L_fft * 5))
    psd_inits.append(1.0 + 0.3 * np.cos(2 * np.pi * freqs / L_fft * 10))
    
    # Random PSDs
    for _ in range(15):
        P_rand = 0.5 + rng_psd.random(half)
        P_rand = P_rand * np.exp(-freqs / rng_psd.uniform(20, 200))
        psd_inits.append(P_rand)
    
    # Optimize each PSD initialization
    for P0 in psd_inits:
        if _check_deadline():
            break
        
        P_opt, val_psd = _psd_optimize(P0, max_iter=150, lr=0.005, beta=80.0)
        x_from_psd = _psd_to_x(P_opt)
        val_x = _objective(x_from_psd)
        
        if val_x < best_val:
            best_val = val_x
            best_x = x_from_psd.copy()
    
    if _check_deadline():
        return best_x
    
    # === FALLBACK: existing pipeline from best_x ===
    
    # Phase 1: Spectral design initialization
    x_spec = _spectral_design()
    val_spec = _objective(x_spec)
    if val_spec < best_val:
        best_val = val_spec
        best_x = x_spec.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 2: Additional structured initializations
    t = np.linspace(0, 1, N)
    initializations = [
        np.ones(N),
        1.0 + 0.5 * np.cos(np.pi * t),
        1.0 + 0.3 * np.sin(2 * np.pi * t),
        np.exp(-2.0 * t),
        1.0 + 0.2 * np.cos(2 * np.pi * 3 * t),
    ]
    
    # Add spectral designs with different phases
    rng_init = np.random.default_rng(123)
    for _ in range(10):
        L_fft2 = 1
        while L_fft2 < 2 * N:
            L_fft2 *= 2
        freqs2 = np.arange(L_fft2 // 2 + 1)
        P2 = 1.0 + 0.5 * np.exp(-freqs2 / rng_init.uniform(20, 100))
        phase2 = rng_init.uniform(0, 2 * np.pi, size=L_fft2 // 2 + 1)
        phase2[0] = 0
        X_mag2 = np.sqrt(P2)
        X2 = X_mag2 * np.exp(1j * phase2)
        X_full2 = np.zeros(L_fft2, dtype=complex)
        X_full2[:L_fft2 // 2 + 1] = X2
        for k in range(1, L_fft2 // 2):
            X_full2[L_fft2 - k] = np.conj(X2[k])
        x_init = np.fft.irfft(X_full2, n=L_fft2)[:N]
        s = np.sum(x_init)
        if abs(s) > 1e-10:
            x_init = x_init * (N / s)
        initializations.append(x_init)
    
    for x0 in initializations:
        if _check_deadline():
            break
        val = _objective(x0)
        if val < best_val:
            best_val = val
            best_x = x0.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 3: Smooth gradient optimization with L-BFGS
    def _smooth_obj(x):
        return _objective_grad_smooth(x, beta=50.0)
    
    try:
        result = minimize(_smooth_obj, best_x, method='L-BFGS-B', jac=True,
                        options={'maxiter': 100, 'gtol': 1e-8, 'ftol': 1e-14})
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x.copy()
    except Exception:
        pass
    
    if _check_deadline():
        return best_x
    
    # Phase 4: Coordinate descent for fine-tuning
    best_x, best_val = _coordinate_descent(best_x, max_rounds=50, beta=50.0)
    
    if _check_deadline():
        return best_x
    
    # Phase 5: Signed pattern search
    best_x, best_val = _signed_pattern_search(best_x, max_rounds=30)
    
    if _check_deadline():
        return best_x
    
    # Phase 6: Another round of smooth optimization
    try:
        result = minimize(_smooth_obj, best_x, method='L-BFGS-B', jac=True,
                        options={'maxiter': 50, 'gtol': 1e-9, 'ftol': 1e-15})
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x.copy()
    except Exception:
        pass
    
    if _check_deadline():
        return best_x
    
    # Phase 7: Final coordinate descent
    best_x, best_val = _coordinate_descent(best_x, max_rounds=30, beta=100.0)
    
    return best_x
