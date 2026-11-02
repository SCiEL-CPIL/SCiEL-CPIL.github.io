import numpy as np
import time
from scipy.signal import fftconvolve
from scipy.optimize import minimize, Bounds

N = 400
DEADLINE = 145.0  # seconds, with safety margin

def _compute_c3prime(x):
    """Compute C3' = max(|x*x|) / (sum(x))^2"""
    s = np.sum(x)
    if abs(s) < 1e-15:
        return np.inf
    conv = fftconvolve(x, x, mode='full')
    return np.max(np.abs(conv)) / (s * s)

def _autoconv_fft(x):
    """Compute autoconvolution via FFT (full mode)."""
    n = len(x)
    size = 2 * n - 1
    X = np.fft.rfft(x, n=size)
    conv = np.fft.irfft(X * X, n=size)
    return conv

def _max_abs_autoconv(x):
    """Return max |autoconv| and the sum."""
    conv = _autoconv_fft(x)
    return np.max(np.abs(conv)), np.sum(x)

def _flip_delta(x, idx):
    """
    Compute the new max|autoconv| and new sum if we flip x[idx].
    Uses FFT: flipping x[idx] changes X by -2*x[idx]*exp(-2πi*k*idx/size).
    We recompute the full autoconv via FFT for correctness but this is fast.
    """
    x_new = x.copy()
    x_new[idx] = -x_new[idx]
    s_new = np.sum(x_new)
    if abs(s_new) < 1e-15:
        return np.inf, s_new
    conv_new = _autoconv_fft(x_new)
    return np.max(np.abs(conv_new)), s_new

def _soft_max_abs(conv, beta):
    """Smooth approximation of max|conv| using log-sum-exp."""
    abs_conv = np.abs(conv)
    max_val = np.max(abs_conv)
    if max_val < 1e-15:
        return 0.0
    shifted = abs_conv / max_val
    log_sum = np.log(np.sum(np.exp(beta * (shifted - 1.0))))
    return max_val * (1.0 + np.log(np.sum(np.exp(beta * (shifted - 1.0)))) - beta) / beta

def _objective_and_grad(a, beta, start_time):
    """
    Objective for continuous amplitude vector a (length N).
    We want to minimize max|autoconv| / (sum(a))^2.
    Using epigraph: minimize t s.t. t >= |autoconv_k| / (sum(a))^2 for all k.
    Here we use a soft-max formulation for differentiability.
    
    Parameters:
        a: continuous amplitudes (can be negative)
        beta: softness parameter
    Returns:
        objective value, gradient
    """
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 5:
        return np.inf, np.zeros_like(a)
    
    s = np.sum(a)
    if abs(s) < 1e-10:
        return 1e10, np.zeros_like(a)
    
    # Compute autoconvolution
    n = len(a)
    size = 2 * n - 1
    A = np.fft.rfft(a, n=size)
    conv = np.fft.irfft(A * A, n=size)
    
    # Soft max of |conv| / s^2
    abs_conv = np.abs(conv)
    max_abs = np.max(abs_conv)
    if max_abs < 1e-15:
        obj = 0.0
        grad = np.zeros_like(a)
    else:
        # Soft max approximation
        scaled = abs_conv / max_abs
        weights = np.exp(beta * (scaled - 1.0))
        weights /= np.sum(weights)
        obj = max_abs / (s * s)
        
        # Gradient via chain rule
        # d(obj)/d(a) = d(max_abs)/d(a) / s^2 - 2*max_abs*sum_grad(s) / s^3
        # d(max_abs)/d(a) ≈ sum_k weights_k * d|conv_k|/d(a)
        # d|conv_k|/d(a) = sign(conv_k) * d(conv_k)/d(a)
        # conv_k = sum_i a[i]*a[i+k], so d(conv_k)/d(a[j]) = a[j+k] + a[j-k] (with boundary handling)
        
        # For FFT-based approach, use automatic differentiation via finite differences for gradient
        # This is expensive but we'll use it sparingly
        eps = 1e-6
        grad = np.zeros_like(a)
        for j in range(n):
            a_plus = a.copy()
            a_plus[j] += eps
            s_plus = np.sum(a_plus)
            if abs(s_plus) < 1e-10:
                grad[j] = 0
                continue
            A_plus = np.fft.rfft(a_plus, n=size)
            conv_plus = np.fft.irfft(A_plus * A_plus, n=size)
            abs_plus = np.abs(conv_plus)
            max_plus = np.max(abs_plus)
            if max_plus < 1e-15:
                obj_plus = 0.0
            else:
                obj_plus = max_plus / (s_plus * s_plus)
            grad[j] = (obj_plus - obj) / eps
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e10:
            grad = grad / grad_norm * 1e10
    
    return obj, grad

def _spectral_phase_optimize(rng, start_time, n_iter=200):
    """
    Optimize the spectral phase to minimize peak autoconvolution.
    Fix |X[k]| = 1 for all k, optimize phase.
    """
    size = 2 * N - 1
    # Initialize with random phases
    phase = rng.uniform(0, 2 * np.pi, size=size // 2 + 1)
    phase[0] = 0  # DC component real
    
    best_x = np.ones(N)
    best_score = _compute_c3prime(best_x)
    
    # Gradient descent on phase
    lr = 0.1
    for it in range(n_iter):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        # Reconstruct signal from fixed magnitude and current phase
        X_mag = np.ones(size // 2 + 1)
        X = X_mag * np.exp(1j * phase)
        # Make X Hermitian symmetric for real signal
        X_full = np.zeros(size, dtype=complex)
        X_full[:size // 2 + 1] = X
        X_full[size // 2 + 1:] = np.conj(X[1:size // 2 + 1][::-1])
        
        x = np.fft.irfft(X_full, n=size)[:N]
        # Normalize to have sum ~ N (positive bias)
        s = np.sum(x)
        if abs(s) > 1e-10:
            x = x * (N / s)
        
        # Compute objective
        conv = _autoconv_fft(x)
        max_abs = np.max(np.abs(conv))
        s_val = np.sum(x)
        if abs(s_val) < 1e-10:
            continue
        score = max_abs / (s_val * s_val)
        
        if score < best_score:
            best_score = score
            best_x = x.copy()
        
        # Compute gradient of score w.r.t. phase
        # Use finite differences for a few key phases
        # This is expensive, so we do a simplified version
        # Just do a few random phase perturbations
        for _ in range(3):
            idx = rng.integers(1, size // 2)
            delta_phase = rng.uniform(-0.5, 0.5)
            phase_new = phase.copy()
            phase_new[idx] += delta_phase
            
            X_mag_new = np.ones(size // 2 + 1)
            X_new = X_mag_new * np.exp(1j * phase_new)
            X_full_new = np.zeros(size, dtype=complex)
            X_full_new[:size // 2 + 1] = X_new
            X_full_new[size // 2 + 1:] = np.conj(X_new[1:size // 2 + 1][::-1])
            
            x_new = np.fft.irfft(X_full_new, n=size)[:N]
            s_new = np.sum(x_new)
            if abs(s_new) < 1e-10:
                continue
            x_new = x_new * (N / s_new)
            conv_new = _autoconv_fft(x_new)
            max_abs_new = np.max(np.abs(conv_new))
            s_new_val = np.sum(x_new)
            if abs(s_new_val) < 1e-10:
                continue
            score_new = max_abs_new / (s_new_val * s_new_val)
            
            if score_new < score:
                phase[idx] = phase_new[idx]
                score = score_new
                if score < best_score:
                    best_score = score
                    best_x = x_new.copy()
    
    return best_x, best_score

def _continuous_relaxation(x_init, start_time):
    """
    Continuous relaxation: optimize amplitudes freely, then threshold.
    Uses L-BFGS with a soft-max objective.
    """
    n = len(x_init)
    size = 2 * n - 1
    
    def objective(a):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 5:
            return 1e10
        s = np.sum(a)
        if abs(s) < 1e-10:
            return 1e10
        A = np.fft.rfft(a, n=size)
        conv = np.fft.irfft(A * A, n=size)
        max_abs = np.max(np.abs(conv))
        return max_abs / (s * s)
    
    # Start from initial ±1 values
    a0 = x_init.astype(float)
    
    # Use Nelder-Mead for a few iterations (doesn't need gradient)
    result = minimize(objective, a0, method='Nelder-Mead', 
                      options={'maxiter': 50, 'xatol': 0.01, 'fatol': 1e-6,
                               'adaptive': True})
    
    if result.success or result.fun < objective(a0):
        a_opt = result.x
        # Threshold to ±1
        x_thresholded = np.sign(a_opt)
        x_thresholded[x_thresholded == 0] = 1
    else:
        x_thresholded = x_init.copy()
    
    return x_thresholded

def construct():
    """
    Constructs a signed step function (±1 values) minimizing C3'.
    Uses spectral phase design + continuous relaxation + discrete polish.
    """
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    best_x = np.ones(N)
    best_score = _compute_c3prime(best_x)
    
    def _time_ok(margin=5.0):
        return (time.time() - start_time) < (DEADLINE - margin)
    
    # Phase 1: Spectral phase optimization
    if _time_ok(30.0):
        x_spectral, score_spectral = _spectral_phase_optimize(rng, start_time, n_iter=100)
        if score_spectral < best_score:
            best_score = score_spectral
            best_x = x_spectral.copy()
        # Threshold and use as starting point
        x_thresh = np.sign(x_spectral)
        x_thresh[x_thresh == 0] = 1
        score_thresh = _compute_c3prime(x_thresh)
        if score_thresh < best_score:
            best_score = score_thresh
            best_x = x_thresh.copy()
    
    # Phase 2: Multi-start discrete search from diverse initializations
    n_starts = 6
    for start in range(n_starts):
        if not _time_ok(15.0):
            break
        
        # Diverse initializations
        if start == 0:
            x = best_x.copy()
        elif start == 1:
            x = np.ones(N)
            x[rng.choice(N, size=N//4, replace=False)] = -1
        elif start == 2:
            x = np.ones(N)
            x[rng.choice(N, size=N//2, replace=False)] = -1
        elif start == 3:
            x = np.ones(N)
            x[rng.choice(N, size=3*N//4, replace=False)] = -1
        elif start == 4:
            x = rng.choice([-1, 1], size=N)
        else:
            # Use spectral result with noise
            if _time_ok(5.0):
                x_s, _ = _spectral_phase_optimize(rng, start_time, n_iter=50)
                x = np.sign(x_s)
                x[x == 0] = 1
                # Add some random flips
                flip_idx = rng.choice(N, size=N//10, replace=False)
                x[flip_idx] = -x[flip_idx]
            else:
                x = rng.choice([-1, 1], size=N)
        
        s = np.sum(x)
        if abs(s) < 1e-15:
            continue
        conv = _autoconv_fft(x)
        max_abs = np.max(np.abs(conv))
        score = max_abs / (s * s)
        
        if score < best_score:
            best_score = score
            best_x = x.copy()
        
        # Simulated annealing local search
        T = 0.3
        T_min = 0.01
        cooling = 0.9995
        no_improve = 0
        max_no_improve = 1500
        
        for it in range(30000):
            if not _time_ok(5.0):
                break
            if no_improve > max_no_improve:
                break
            
            idx = rng.integers(0, N)
            new_max_abs, s_new = _flip_delta(x, idx)
            
            if abs(s_new) < 1e-15:
                no_improve += 1
                continue
            
            new_score = new_max_abs / (s_new * s_new)
            delta = new_score - score
            
            if delta < 0:
                # Accept improvement
                x[idx] = -x[idx]
                score = new_score
                no_improve = 0
                if score < best_score:
                    best_score = score
                    best_x = x.copy()
            else:
                # Accept with probability exp(-delta/T)
                if np.log(rng.random()) > -delta / T:
                    x[idx] = -x[idx]
                    score = new_score
                no_improve += 1
            
            T = max(T * cooling, T_min)
        
        # After SA, do a final greedy polish
        improved = True
        while improved and _time_ok(5.0):
            improved = False
            for idx in range(N):
                if not _time_ok(5.0):
                    break
                new_max_abs, s_new = _flip_delta(x, idx)
                if abs(s_new) < 1e-15:
                    continue
                new_score = new_max_abs / (s_new * s_new)
                if new_score < score:
                    x[idx] = -x[idx]
                    score = new_score
                    improved = True
                    if score < best_score:
                        best_score = score
                        best_x = x.copy()
    
    # Phase 3: Continuous relaxation polish on best solution
    if _time_ok(20.0):
        x_relaxed = _continuous_relaxation(best_x, start_time)
        score_relaxed = _compute_c3prime(x_relaxed)
        if score_relaxed < best_score:
            best_score = score_relaxed
            best_x = x_relaxed.copy()
        
        # Polish the relaxed result
        x = best_x.copy()
        s = np.sum(x)
        if abs(s) > 1e-15:
            conv = _autoconv_fft(x)
            max_abs = np.max(np.abs(conv))
            score = max_abs / (s * s)
            
            # Greedy polish
            improved = True
            while improved and _time_ok(5.0):
                improved = False
                for idx in range(N):
                    if not _time_ok(5.0):
                        break
                    new_max_abs, s_new = _flip_delta(x, idx)
                    if abs(s_new) < 1e-15:
                        continue
                    new_score = new_max_abs / (s_new * s_new)
                    if new_score < score:
                        x[idx] = -x[idx]
                        score = new_score
                        improved = True
                        if score < best_score:
                            best_score = score
                            best_x = x.copy()
    
    return best_x
