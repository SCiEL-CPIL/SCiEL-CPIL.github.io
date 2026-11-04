"""
Zadoff-Chu chirp parameterization + smooth optimization polish.

Key insight: Zadoff-Chu sequences have constant amplitude and zero periodic autocorrelation
in the complex domain. By taking the real part of a chirped Zadoff-Chu sequence with
continuously optimized phase parameters, we exploit the inherent low-autocorrelation
structure to minimize off-diagonal autoconvolution peaks.

Strategy:
1. Generate candidates using chirp parameter optimization (fast, 3D L-BFGS-B).
2. Add spectral flatness-based initialization for additional diversity.
3. Use smooth objective (soft-max) + L-BFGS-B to refine top candidates.
4. Final gradient descent polish with corrected gradient.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 400

def _autoconv(x):
    """Linear autoconvolution via FFT."""
    n = len(x)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    X = np.fft.rfft(x, fsize)
    c = np.fft.irfft(X * X, fsize)[:size]
    return c

def _metric_offdiag(x):
    """Compute max |c_k| for k != 0 divided by (sum x)^2."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    c = _autoconv(x)
    if len(c) > 1:
        max_val = np.max(np.abs(c[1:]))
    else:
        max_val = 0.0
    return max_val / (s * s)

def _objective_smooth(x, temp=1.0):
    """Smooth approximation of the off-diagonal metric for optimization."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return 1e10
    c = _autoconv(x)
    if len(c) > 1:
        abs_c = np.abs(c[1:])
    else:
        return 0.0
    max_c = np.max(abs_c)
    if max_c < 1e-15:
        return 0.0
    # Numerically stable soft-max
    shifted = (abs_c - max_c) / temp
    log_sum_exp = max_c / temp + np.log(np.sum(np.exp(shifted)))
    soft_max = temp * log_sum_exp
    return soft_max / (s * s)

def _chirp_sequence(beta, gamma=0.0, delta=0.0):
    """
    Generate a real-valued sequence from a generalized Zadoff-Chu chirp.
    """
    n = np.arange(N, dtype=float)
    phase = np.pi * beta * (n ** 2) / N + gamma * n + delta
    z = np.exp(1j * phase)
    x = np.real(z)
    return x

def _chirp_objective(params):
    """
    Objective function for chirp parameter optimization.
    params: [beta, gamma, delta]
    """
    beta, gamma, delta = params
    x = _chirp_sequence(beta, gamma, delta)
    return _metric_offdiag(x)

def _optimize_chirp_params(x0_params, deadline=None, max_iter=200):
    """
    Optimize chirp parameters (beta, gamma, delta) using L-BFGS-B.
    Returns best params and corresponding sequence.
    """
    try:
        res = minimize(
            _chirp_objective,
            x0_params,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        best_params = res.x
        x = _chirp_sequence(*best_params)
        val = _metric_offdiag(x)
        return x, val, best_params
    except Exception:
        return None, float('inf'), x0_params

def _flat_spectrum_candidate(rng, n_harmonics=10):
    """
    Generate a candidate with flat-ish spectrum by random phase modulation.
    This exploits the insight that flat-spectrum designs minimize autoconvolution peaks.
    """
    # Start with a base pattern
    base = np.ones(N)
    
    # Apply random phase shifts at selected harmonic frequencies
    n = np.arange(N, dtype=float)
    x = np.zeros(N)
    for k in range(1, n_harmonics + 1):
        phase = rng.uniform(0, 2 * np.pi)
        x += np.cos(2 * np.pi * k * n / N + phase)
    
    # Ensure non-zero sum by adding DC component
    s = np.sum(x)
    if abs(s) < 1e-6:
        x[0] += 1.0
        s = np.sum(x)
    
    return x

def _gradient_descent_polish(x0, max_iter=500, initial_lr=0.01, deadline=None):
    """
    Projected gradient descent with annealed temperature for final polish.
    Uses corrected gradient computation via direct convolution.
    """
    x = x0.copy().astype(float)
    s = np.sum(x)
    if abs(s) < 1e-12:
        return x
    
    # Normalize sum to 1
    x = x / s
    
    best_x = x.copy()
    best_val = _metric_offdiag(best_x)
    
    lr = initial_lr
    temp = 0.5
    fsize = 1
    while fsize < 2 * N:
        fsize *= 2
    
    for i in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        c = _autoconv(x)
        if len(c) <= 1:
            break
        abs_c = np.abs(c[1:])
        
        max_c = np.max(abs_c)
        if max_c < 1e-15:
            break
        
        # Soft-max weights
        shifted = (abs_c - max_c) / temp
        exp_vals = np.exp(shifted)
        sum_exp = np.sum(exp_vals)
        soft_max = max_c + temp * np.log(sum_exp)
        
        # Gradient of soft-max w.r.t. c[1:]
        grad_abs_c = exp_vals / sum_exp
        c_off = c[1:]
        sign_c = np.sign(c_off)
        grad_c = grad_abs_c * sign_c
        
        # Build full convolution gradient vector for all lags
        # c[k] = sum_m x[m]*x[m+k]
        # d(c[k])/dx[m] = x[m+k] + x[m-k] (with appropriate boundaries)
        # So grad_x[m] = sum_k grad_c[k] * (x[m+k] + x[m-k])
        # This is equivalent to convolving grad_c (extended symmetrically) with x
        
        # Build the weight vector for all lags from -(N-1) to (N-1)
        # For positive lag k: weight is grad_c[k-1] (since c[1:] starts at lag 1)
        # For negative lag -k: by symmetry of autoconvolution, c[-k] = c[k]
        # So grad_c for negative lags mirrors positive lags
        w_full = np.zeros(2 * N - 1)
        # w_full[k] corresponds to lag k-(N-1)
        # Positive lags 1..N-1: w_full[N-1+k] = grad_c[k-1]
        w_full[N:] = grad_c  # lags 1 to N-1
        # Negative lags -(N-1)..-1: w_full[N-1-k] = grad_c[k-1] (by symmetry)
        w_full[:N-1] = grad_c[::-1]  # lags -(N-1) to -1
        
        # Now grad_x[m] = sum_k w_full[k] * x[m + (k - (N-1))]
        # This is a linear convolution of w_full with x, but we need to be careful
        # Actually: grad_x[m] = sum_j w_full[j] * x[m - (j - (N-1))]
        # Let's use FFT for this cross-correlation
        X = np.fft.rfft(x, fsize)
        W = np.fft.rfft(w_full, fsize)
        # Cross-correlation: grad_x[m] = sum_j w_full[j] * x[m - j + (N-1)]
        # This is ifft(W * conj(X)) shifted appropriately
        grad_x = np.fft.irfft(W * np.conj(X), fsize)
        # The result is circular cross-correlation; take the first N elements
        # and adjust for the shift
        grad_x = grad_x[:N]
        
        s = np.sum(x)
        metric = soft_max / (s * s)
        
        # Gradient of metric = d(soft_max)/dx * 1/s^2 + soft_max * d(1/s^2)/dx
        # d(1/s^2)/dx[m] = -2/s^3
        grad_metric = grad_x / (s * s) - 2.0 * soft_max / (s ** 3)
        
        x_new = x - lr * grad_metric
        
        s_new = np.sum(x_new)
        if abs(s_new) < 1e-12:
            x_new = x + 1e-6 * np.random.randn(N)
            s_new = np.sum(x_new)
        x_new = x_new / s_new
        
        val = _metric_offdiag(x_new)
        if val < best_val:
            best_val = val
            best_x = x_new.copy()
        
        x = x_new
        lr *= 0.995
        temp *= 0.999
        
    return best_x

def construct():
    t0 = time.time()
    deadline = t0 + 115.0
    
    best_x = np.ones(N)
    best_val = _metric_offdiag(best_x)
    
    rng = np.random.default_rng(42)
    
    candidates = []
    
    # 1. All ones (baseline)
    candidates.append(np.ones(N))
    
    # 2. Chirp parameter optimization from multiple starting points
    chirp_starts = [
        [0.5, 0.0, 0.0],
        [-0.5, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.3, 0.5, 0.0],
        [-0.3, -0.5, 0.0],
        [0.7, 0.3, 0.2],
        [-0.7, -0.3, -0.2],
        [0.1, 0.1, 0.1],
        [-0.1, -0.1, -0.1],
    ]
    
    # Limit chirp optimization to ~30s total
    chirp_deadline = t0 + 30.0
    for start in chirp_starts:
        if time.time() > chirp_deadline:
            break
        x_chirp, val_chirp, params = _optimize_chirp_params(
            np.array(start, dtype=float),
            deadline=chirp_deadline,
            max_iter=80
        )
        if x_chirp is not None and val_chirp < best_val:
            best_val = val_chirp
            best_x = x_chirp.copy()
        if x_chirp is not None:
            candidates.append(x_chirp)
    
    # 3. Spectral flatness-based candidates
    for _ in range(8):
        if time.time() > deadline - 50:
            break
        x_flat = _flat_spectrum_candidate(rng, n_harmonics=rng.integers(3, 20))
        candidates.append(x_flat)
    
    # 4. Random binary sequences with non-zero mean
    for _ in range(8):
        if time.time() > deadline - 40:
            break
        r = rng.random(N)
        x = np.where(r > 0.35, 1.0, -1.0)
        if abs(np.sum(x)) > 5:
            candidates.append(x)
    
    # 5. Block patterns
    for block_size in [10, 20, 50, 100]:
        if time.time() > deadline - 35:
            break
        pattern = np.zeros(N)
        for i in range(N):
            if (i // block_size) % 2 == 0:
                pattern[i] = 1.0
            else:
                pattern[i] = -1.0
        pattern[0] += 0.5
        candidates.append(pattern)
    
    # 6. Smooth sinusoidal patterns with varying frequencies
    for freq in [2, 3, 5, 7, 13]:
        if time.time() > deadline - 30:
            break
        n = np.arange(N, dtype=float)
        x = np.cos(2 * np.pi * freq * n / N + rng.uniform(0, 2 * np.pi))
        x += 0.5  # Ensure positive mean
        candidates.append(x)
    
    # 7. Multi-scale patterns
    for base_freq in [1, 2, 4]:
        if time.time() > deadline - 25:
            break
        n = np.arange(N, dtype=float)
        x = np.zeros(N)
        for k in range(1, 6):
            x += np.cos(2 * np.pi * base_freq * k * n / N + rng.uniform(0, 2 * np.pi)) / k
        x += 1.0  # Ensure positive mean
        candidates.append(x)
    
    # Optimize top candidates with L-BFGS-B on smooth objective
    # Sort by current metric to prioritize best
    scored = [(c, _metric_offdiag(c)) for c in candidates]
    scored.sort(key=lambda t: t[1])
    
    # Take top 8 candidates for optimization with multi-stage temperature
    for x0, cur_val in scored[:8]:
        if time.time() > deadline - 15:
            break
        
        x0 = x0.copy()
        s0 = np.sum(x0)
        if abs(s0) < 1e-6:
            continue
        x0 = x0 / s0
        
        try:
            # Stage 1: High temperature for broad exploration
            res1 = minimize(
                lambda x: _objective_smooth(x, temp=2.0),
                x0,
                method='L-BFGS-B',
                options={'maxiter': 60, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            x_stage1 = res1.x
            
            # Stage 2: Medium temperature
            res2 = minimize(
                lambda x: _objective_smooth(x, temp=0.5),
                x_stage1,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            x_stage2 = res2.x
            
            # Stage 3: Low temperature for fine polishing
            res3 = minimize(
                lambda x: _objective_smooth(x, temp=0.05),
                x_stage2,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            x_opt = res3.x
            
            val = _metric_offdiag(x_opt)
            if val < best_val:
                best_val = val
                best_x = x_opt.copy()
        except Exception:
            pass
    
    # Final polish with two stages: coarse then fine
    if time.time() < deadline - 10:
        try:
            # Coarse polish
            polished1 = _gradient_descent_polish(
                best_x,
                max_iter=400,
                initial_lr=0.01,
                deadline=deadline - 10
            )
            val1 = _metric_offdiag(polished1)
            if val1 < best_val:
                best_val = val1
                best_x = polished1
        except Exception:
            pass
    
    # Fine polish with smaller learning rate
    if time.time() < deadline - 3:
        try:
            polished2 = _gradient_descent_polish(
                best_x,
                max_iter=600,
                initial_lr=0.002,
                deadline=deadline - 3
            )
            val2 = _metric_offdiag(polished2)
            if val2 < best_val:
                best_val = val2
                best_x = polished2
        except Exception:
            pass
    
    return best_x
