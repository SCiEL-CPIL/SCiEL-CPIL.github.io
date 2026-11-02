import numpy as np
import time
from scipy.optimize import minimize

N = 400
START_TIME = time.time()
TIME_LIMIT = 145
SAFETY_MARGIN = 5

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - SAFETY_MARGIN)

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

def _max_abs_ac_and_grad(x):
    """
    Compute max|ac| and its gradient w.r.t. x.
    
    ac[k] = sum_{j} x[j] * x[k-j] for valid indices (linear convolution)
    d(ac[k])/dx[j] = x[k-j] + x[j+k] where terms are 0 if index out of [0,N-1]
    
    For the max, gradient is the gradient at the argmax lag (with sign of ac[lag]).
    """
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros(N)
    
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    max_val = np.max(abs_ac)
    
    if max_val < 1e-15:
        return 0.0, np.zeros(N)
    
    # Find all lags achieving (near) the maximum
    threshold = max_val * (1 - 1e-6)
    peak_lags = np.where(abs_ac >= threshold)[0]
    
    # Gradient of max|ac| is sum over peak lags of sign(ac[lag]) * d(ac[lag])/dx
    grad = np.zeros(N)
    for lag in peak_lags:
        sign = np.sign(ac[lag])
        if sign == 0:
            sign = 1.0
        
        # d(ac[lag])/dx[j] = x[lag-j] + x[j+lag] (with bounds)
        # For lag >= 0:
        #   ac[lag] = sum_{j=0}^{N-1-lag} x[j]*x[j+lag]  (if lag < N)
        #   d/dx[j] = x[j+lag] + x[j-lag] if both in range
        # More carefully for linear conv:
        #   ac[k] = sum_{j=max(0,k-N+1)}^{min(N-1,k)} x[j]*x[k-j]
        #   d(ac[k])/dx[j] = x[k-j] + x[j+k] if 0<=k-j<N and 0<=j+k<N else just one term
        
        for j in range(N):
            g = 0.0
            # Term 1: x[k-j] if 0 <= k-j < N
            idx1 = lag - j
            if 0 <= idx1 < N:
                g += x[idx1]
            # Term 2: x[j+k] if 0 <= j+k < N
            idx2 = j + lag
            if 0 <= idx2 < N:
                g += x[idx2]
            grad[j] += sign * g
    
    # Scale by 1/s^2 for the full objective
    val = max_val / (s * s)
    # Gradient of val = (1/s^2)*grad_max - (2*max_val/s^3)*1
    grad_obj = (1.0 / (s * s)) * grad - (2.0 * max_val / (s ** 3)) * np.ones(N)
    
    return val, grad_obj

def _smooth_objective_grad(x, beta=20.0):
    """
    Smooth approximation using log-sum-exp of |ac|, with correct gradient.
    
    f(x) = (1/beta) * log(sum_k exp(beta*|ac[k]|)) / s^2
    
    Gradient computed via chain rule with correct d(ac[k])/dx[j].
    """
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros(N)
    
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    max_abs = np.max(abs_ac)
    
    if max_abs < 1e-15:
        return 0.0, np.zeros(N)
    
    # Numerically stable log-sum-exp
    shifted = beta * (abs_ac - max_abs)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    log_sum_exp = max_abs + (1.0 / beta) * np.log(sum_exp)
    
    val = log_sum_exp / (s * s)
    
    # Weights: w[k] = exp(beta*|ac[k]|) / sum_j exp(beta*|ac[j]|)
    weights = exp_shifted / sum_exp
    
    # Gradient of log(sum exp(beta*|ac|)) w.r.t. x:
    # = beta * sum_k w[k] * sign(ac[k]) * d(ac[k])/dx
    
    grad_inner = np.zeros(N)
    for k in range(len(ac)):
        wk = weights[k]
        if wk < 1e-15:
            continue
        sign_k = np.sign(ac[k])
        if sign_k == 0:
            sign_k = 1.0
        
        # d(ac[k])/dx[j] for all j
        # ac[k] = sum_j x[j]*x[k-j] (linear conv, valid range)
        # d/dx[j] = x[k-j] + x[j+k] (with bounds)
        for j in range(N):
            g = 0.0
            idx1 = k - j
            if 0 <= idx1 < N:
                g += x[idx1]
            idx2 = j + k
            if 0 <= idx2 < N:
                g += x[idx2]
            grad_inner[j] += sign_k * g
    
    d_logsumexp_dx = beta * grad_inner
    
    # Full gradient: d(val)/dx = (1/s^2)*d_logsumexp_dx - (2*logsumexp/s^3)*1
    grad = (1.0 / (s * s)) * d_logsumexp_dx - (2.0 * log_sum_exp / (s ** 3)) * np.ones(N)
    
    return val, grad

def _spectral_flatten(x, n_iters=50):
    """
    Iteratively flatten the autocorrelation by adjusting the power spectrum.
    
    Key idea: ac = IFFT(|FFT(x)|^2). To reduce peaks in ac, we can 
    adjust |FFT(x)| to make ac more uniform.
    
    Use iterative reweighting: at each step, find the peak in ac,
    and adjust x to reduce that peak.
    """
    x = x.copy()
    s_target = np.sum(x)
    
    for it in range(n_iters):
        if _check_deadline():
            break
        
        ac = _autoconv_fft(x)
        abs_ac = np.abs(ac)
        max_val = np.max(abs_ac)
        
        if max_val < 1e-10:
            break
        
        # Find the peak lag
        peak_lag = np.argmax(abs_ac)
        peak_sign = np.sign(ac[peak_lag])
        if peak_sign == 0:
            peak_sign = 1.0
        
        # Compute gradient at this peak only (sparse)
        grad = np.zeros(N)
        for j in range(N):
            g = 0.0
            idx1 = peak_lag - j
            if 0 <= idx1 < N:
                g += x[idx1]
            idx2 = j + peak_lag
            if 0 <= idx2 < N:
                g += x[idx2]
            grad[j] = peak_sign * g
        
        # Step in negative gradient direction
        step_size = 0.01 * max_val / (np.max(np.abs(grad)) + 1e-10)
        x = x - step_size * grad
        
        # Renormalize to maintain sum
        s = np.sum(x)
        if abs(s) > 1e-10:
            x = x * (s_target / s)
    
    return x

def _frequency_domain_optimize(x0, n_iters=30):
    """
    Optimize in the frequency domain: x = IFFT(X), minimize max|IFFT(|X|^2)|/s^2.
    
    Parameterize X and use gradient descent on the magnitude.
    """
    L_fft = 1
    while L_fft < 2 * N:
        L_fft *= 2
    
    X = np.fft.rfft(x0, n=L_fft)
    X_mag = np.abs(X)
    X_phase = np.angle(X)
    
    s_target = np.sum(x0)
    
    for it in range(n_iters):
        if _check_deadline():
            break
        
        # Reconstruct x
        X_full = np.zeros(L_fft, dtype=complex)
        X_full[:L_fft // 2 + 1] = X_mag * np.exp(1j * X_phase)
        for k in range(1, L_fft // 2):
            X_full[L_fft - k] = np.conj(X_full[k])
        
        x = np.fft.irfft(X_full, n=L_fft)[:N]
        s = np.sum(x)
        if abs(s) < 1e-10:
            break
        
        # Compute autocorrelation
        ac = _autoconv_fft(x)
        abs_ac = np.abs(ac)
        max_val = np.max(abs_ac)
        
        if max_val < 1e-10:
            break
        
        # Peak lag
        peak_lag = np.argmax(abs_ac)
        peak_sign = np.sign(ac[peak_lag])
        if peak_sign == 0:
            peak_sign = 1.0
        
        # Gradient w.r.t. x
        grad_x = np.zeros(N)
        for j in range(N):
            g = 0.0
            idx1 = peak_lag - j
            if 0 <= idx1 < N:
                g += x[idx1]
            idx2 = j + peak_lag
            if 0 <= idx2 < N:
                g += x[idx2]
            grad_x[j] = peak_sign * g
        
        # Map gradient to frequency domain
        # dx/dX_mag[j] involves the phase
        # Simplified: step in X_mag opposite to projected gradient
        grad_X = np.fft.rfft(grad_x, n=L_fft)[:L_fft // 2 + 1]
        
        # Project to magnitude direction
        grad_mag = np.real(grad_X * np.conj(np.exp(1j * X_phase)))
        
        # Update magnitude
        step_size = 0.005 * max_val / (np.max(np.abs(grad_mag)) + 1e-10)
        X_mag = np.maximum(0, X_mag - step_size * grad_mag)
        
        # Renormalize
        x_new = np.fft.irfft(X_mag * np.exp(1j * X_phase), n=L_fft)[:N]
        s_new = np.sum(x_new)
        if abs(s_new) > 1e-10:
            scale = s_target / s_new
            X_mag = X_mag * scale
    
    # Final reconstruction
    X_full = np.zeros(L_fft, dtype=complex)
    X_full[:L_fft // 2 + 1] = X_mag * np.exp(1j * X_phase)
    for k in range(1, L_fft // 2):
        X_full[L_fft - k] = np.conj(X_full[k])
    x = np.fft.irfft(X_full, n=L_fft)[:N]
    
    s = np.sum(x)
    if abs(s) > 1e-10:
        x = x * (s_target / s)
    
    return x

def _random_spectral_init(seed):
    """Generate random spectral initialization."""
    L_fft = 1
    while L_fft < 2 * N:
        L_fft *= 2
    
    rng = np.random.default_rng(seed)
    
    # Random power spectrum with some structure
    freqs = np.arange(L_fft // 2 + 1)
    # Mix of flat and decaying
    alpha = rng.uniform(0.5, 3.0)
    P = 1.0 + alpha * np.exp(-freqs / rng.uniform(50, 200))
    
    # Random phase
    phase = rng.uniform(0, 2 * np.pi, size=L_fft // 2 + 1)
    phase[0] = 0
    
    X_mag = np.sqrt(P)
    X = X_mag * np.exp(1j * phase)
    
    X_full = np.zeros(L_fft, dtype=complex)
    X_full[:L_fft // 2 + 1] = X
    for k in range(1, L_fft // 2):
        X_full[L_fft - k] = np.conj(X[k])
    
    x = np.fft.irfft(X_full, n=L_fft)[:N]
    
    s = np.sum(x)
    if abs(s) > 1e-10:
        x = x * (N / s)
    
    return x

def construct():
    best_x = np.ones(N)
    best_val = _objective(best_x)
    
    if _check_deadline():
        return best_x
    
    # Phase 1: Generate diverse initializations
    initializations = [
        np.ones(N),
    ]
    
    # Structured initializations
    t = np.linspace(0, 1, N)
    initializations.extend([
        1.0 + 0.5 * np.cos(np.pi * t),
        1.0 + 0.3 * np.sin(2 * np.pi * t),
        np.exp(-2.0 * t),
        1.0 + 0.2 * np.cos(2 * np.pi * 3 * t),
        1.0 + 0.1 * np.sin(4 * np.pi * t),
    ])
    
    # Random spectral initializations
    for seed in range(20):
        if _check_deadline():
            break
        x_init = _random_spectral_init(seed)
        initializations.append(x_init)
    
    # Find best initialization
    for x0 in initializations:
        if _check_deadline():
            break
        val = _objective(x0)
        if val < best_val:
            best_val = val
            best_x = x0.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 2: Spectral flattening (iterative peak suppression)
    x_flattened = _spectral_flatten(best_x, n_iters=30)
    val_flat = _objective(x_flattened)
    if val_flat < best_val:
        best_val = val_flat
        best_x = x_flattened.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 3: Frequency domain optimization
    x_freq = _frequency_domain_optimize(best_x, n_iters=20)
    val_freq = _objective(x_freq)
    if val_freq < best_val:
        best_val = val_freq
        best_x = x_freq.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 4: Gradient-based refinement with smooth objective
    def _smooth_obj(x):
        return _smooth_objective_grad(x, beta=20.0)
    
    try:
        result = minimize(_smooth_obj, best_x, method='L-BFGS-B', jac=True,
                        options={'maxiter': 50, 'gtol': 1e-8, 'ftol': 1e-14})
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x.copy()
    except Exception:
        pass
    
    if _check_deadline():
        return best_x
    
    # Phase 5: More spectral flattening
    x_flattened2 = _spectral_flatten(best_x, n_iters=20)
    val_flat2 = _objective(x_flattened2)
    if val_flat2 < best_val:
        best_val = val_flat2
        best_x = x_flattened2.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 6: Final gradient refinement
    def _smooth_obj_fine(x):
        return _smooth_objective_grad(x, beta=50.0)
    
    try:
        result = minimize(_smooth_obj_fine, best_x, method='L-BFGS-B', jac=True,
                        options={'maxiter': 30, 'gtol': 1e-9, 'ftol': 1e-15})
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x.copy()
    except Exception:
        pass
    
    if _check_deadline():
        return best_x
    
    # Phase 7: Local perturbation search
    rng = np.random.default_rng(999)
    for _ in range(50):
        if _check_deadline():
            break
        
        trial = best_x.copy()
        # Random small perturbation
        mask = rng.random(N) < 0.1
        trial[mask] += rng.normal(0, 0.05, size=np.sum(mask))
        
        # Renormalize
        s = np.sum(trial)
        s_orig = np.sum(best_x)
        if abs(s) > 1e-10:
            trial = trial * (s_orig / s)
        
        val = _objective(trial)
        if val < best_val:
            best_val = val
            best_x = trial.copy()
    
    return best_x
