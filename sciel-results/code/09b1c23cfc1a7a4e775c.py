import numpy as np
import time
from scipy.optimize import minimize

N = 400
START_TIME = time.time()
TIME_LIMIT = 140
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

def _build_toeplitz_for_lag(lag):
    """
    Build the row of the autoconvolution Jacobian for a given lag.
    ac[lag] = sum_j x[j] * x[j+lag]  (for lag >= 0, with appropriate indexing)
    
    For linear autoconvolution: ac[k] = sum_{j} x[j] * x[j + k - (N-1)]
    where k ranges from 0 to 2N-2.
    
    Actually: ac[k] = sum_{j=max(0,k-N+1)}^{min(N-1,k)} x[j] * x[k-j]
    
    The gradient d(ac[k])/dx[j] = x[k-j] + x[j+k] (with bounds)
    """
    pass

def _peak_penalty_objective(x, target_lags, gamma=1.0):
    """
    Objective: (sum x)^2 * gamma + sum_{k in target_lags} max(0, |ac[k]| - threshold)^2
    This is a penalty method approach.
    
    Actually, let's use a different formulation:
    Minimize: max_k |ac[k]| / (sum x)^2
    
    Using the fact that at optimum, the peak is at specific lags,
    we can use a quadratic barrier/penalty on the top few lags.
    """
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros(N)
    
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    max_val = np.max(abs_ac)
    val = max_val / (s * s)
    
    # Gradient of max |ac[k]| / s^2
    # Find which lag(s) achieve the max
    peak_lags = np.where(abs_ac >= max_val * (1 - 1e-12))[0]
    
    # Average sign over peak lags for gradient
    if len(peak_lags) > 0:
        signs = np.sign(ac[peak_lags])
        avg_sign = np.mean(signs)
        # Gradient of ac[k] w.r.t. x[j]
        # ac[k] = sum_j x[j] * x[k-j] (with valid range)
        # d(ac[k])/dx[j] = x[k-j] (if 0<=k-j<N) + x[j+k] (if 0<=j+k<N and j+k != k-j)
        # More carefully: d(ac[k])/dx[j] = x[k-j] + x[j+k] where indices are valid
        # and we avoid double-counting when k-j == j+k, i.e., j = k/2
        
        grad_ac = np.zeros(N)
        for lag in peak_lags:
            # ac[lag] = sum_{j} x[j] * x[lag-j] where 0<=j<N and 0<=lag-j<N
            # d(ac[lag])/dx[j] = x[lag-j] (for valid lag-j) + x[j+lag] (for valid j+lag, j+lag != lag-j)
            # Actually for linear conv: ac[lag] = sum_{j=max(0,lag-N+1)}^{min(N-1,lag)} x[j]*x[lag-j]
            # d(ac[lag])/dx[j] = x[lag-j] if 0<=lag-j<N
            # Plus if we think of it as x convolved with x, the derivative w.r.t x[j] is:
            # sum over all i where x[j] appears: x[i] * delta(j,i) * x[lag-i] + x[i] * x[lag-i] * delta(j,lag-i)
            # = x[lag-j] (if valid) + x[j+lag] (if valid and j+lag != lag-j)
            
            # Simpler: use the FFT-based gradient
            pass
        
        # Use FFT for gradient
        L_fft = 1
        while L_fft < 3 * N:
            L_fft *= 2
        
        # Build gradient vector g where g[k] = sign(ac[k]) * weight[k]
        g = np.zeros(2*N-1)
        for lag in peak_lags:
            g[lag] = avg_sign * np.sign(ac[lag]) / len(peak_lags)
        
        G = np.fft.rfft(g, n=L_fft)
        Xf = np.fft.rfft(x, n=L_fft)
        
        term1 = np.fft.irfft(np.conj(G) * Xf, n=L_fft)[:N]
        term2 = np.fft.irfft(G * Xf, n=L_fft)[:N]
        
        d_max_dx = term1 + term2
    else:
        d_max_dx = np.zeros(N)
    
    grad = (1.0 / (s * s)) * d_max_dx - (2.0 * max_val / (s ** 3)) * np.ones(N)
    
    return val, grad

def _spectral_envelope_optimize(x0, max_iters=50, step_size=0.05):
    """
    Directly optimize the spectral envelope to minimize peak autoconvolution.
    
    Key insight: |ac[k]| = |IFFT(|X|^2)[k]| where X = FFT(x).
    The peak of |IFFT(|X|^2)| is related to the peak of the power spectrum's
    inverse transform.
    
    We parameterize x via its DFT magnitude and phase, and optimize to
    flatten the autoconvolution peak.
    """
    best_x = x0.copy()
    best_val = _objective(best_x)
    
    for it in range(max_iters):
        if _check_deadline():
            break
        
        ac = _autoconv_fft(best_x)
        abs_ac = np.abs(ac)
        max_val = np.max(abs_ac)
        
        if max_val < 1e-15:
            break
        
        # Find the dominant lag
        peak_lag = np.argmax(abs_ac)
        
        # The autoconvolution at lag k is: ac[k] = sum_j x[j]*x[k-j]
        # We want to reduce |ac[peak_lag]| while keeping sum(x) fixed.
        
        # Gradient descent on the peak lag specifically
        s = np.sum(best_x)
        
        # Compute gradient of ac[peak_lag] w.r.t. x
        L_fft = 1
        while L_fft < 3 * N:
            L_fft *= 2
        
        g = np.zeros(2*N-1)
        g[peak_lag] = np.sign(ac[peak_lag])
        
        G = np.fft.rfft(g, n=L_fft)
        Xf = np.fft.rfft(best_x, n=L_fft)
        
        term1 = np.fft.irfft(np.conj(G) * Xf, n=L_fft)[:N]
        term2 = np.fft.irfft(G * Xf, n=L_fft)[:N]
        
        d_ac_dx = term1 + term2
        
        # Project out the component along ones (to preserve sum)
        ones = np.ones(N)
        projection = np.dot(d_ac_dx, ones) / N
        d_ac_dx_centered = d_ac_dx - projection * ones
        
        # Update
        step = step_size * d_ac_dx_centered / (np.max(np.abs(d_ac_dx_centered)) + 1e-10)
        
        trial = best_x - step * 0.1
        
        # Also try to balance: reduce peak, increase others
        # Add a regularization toward uniform
        trial = 0.95 * trial + 0.05 * (s / N) * ones
        
        val = _objective(trial)
        if val < best_val:
            best_val = val
            best_x = trial.copy()
        else:
            # Try smaller step
            step_small = step * 0.1
            trial2 = best_x - step_small
            trial2 = 0.98 * trial2 + 0.02 * (s / N) * ones
            val2 = _objective(trial2)
            if val2 < best_val:
                best_val = val2
                best_x = trial2.copy()
            else:
                break
    
    return best_x, best_val

def _iterative_peak_reduction(x0, max_rounds=100):
    """
    Iteratively identify the peak lag and apply targeted corrections.
    Uses the explicit structure of autoconvolution.
    """
    best_x = x0.copy()
    best_val = _objective(best_x)
    
    for rnd in range(max_rounds):
        if _check_deadline():
            break
        
        ac = _autoconv_fft(best_x)
        abs_ac = np.abs(ac)
        max_val = np.max(abs_ac)
        s = np.sum(best_x)
        
        if max_val < 1e-15:
            break
        
        # Find top few lags
        top_k = min(5, 2*N-1)
        top_lags = np.argsort(abs_ac)[-top_k:]
        
        # For each top lag, compute the contribution and try to reduce it
        # ac[k] = sum_j x[j] * x[k-j]
        # To reduce |ac[k]|, we can adjust x to make the correlation at lag k smaller
        
        # Strategy: for the dominant lag k*, find pairs (j, k*-j) that contribute
        # most to ac[k*] and adjust them
        k_star = top_lags[-1]  # The very peak
        
        # Compute contributions
        j_min = max(0, k_star - N + 1)
        j_max = min(N - 1, k_star)
        
        if j_max < j_min:
            continue
        
        js = np.arange(j_min, j_max + 1)
        contributions = best_x[js] * best_x[k_star - js]
        
        # The sign of ac[k_star]
        sign_k = np.sign(ac[k_star])
        
        # To reduce |ac[k_star]|, we want to make contributions cancel
        # One approach: shift some mass from high-contribution pairs
        
        # Simpler: use gradient step
        L_fft = 1
        while L_fft < 3 * N:
            L_fft *= 2
        
        g = np.zeros(2*N-1)
        for lag in top_lags:
            weight = abs_ac[lag] / (max_val + 1e-15)
            g[lag] = np.sign(ac[lag]) * weight
        
        G = np.fft.rfft(g, n=L_fft)
        Xf = np.fft.rfft(best_x, n=L_fft)
        
        term1 = np.fft.irfft(np.conj(G) * Xf, n=L_fft)[:N]
        term2 = np.fft.irfft(G * Xf, n=L_fft)[:N]
        
        d_obj_dx = term1 + term2
        
        # Center to preserve sum
        ones = np.ones(N)
        d_obj_dx_centered = d_obj_dx - np.dot(d_obj_dx, ones) / N
        
        # Normalize step
        step_norm = np.linalg.norm(d_obj_dx_centered)
        if step_norm < 1e-10:
            break
        
        step = 0.01 * d_obj_dx_centered / step_norm
        trial = best_x - step
        
        # Ensure no zeros (keep positive for stability)
        trial = np.maximum(trial, 0.01)
        
        val = _objective(trial)
        if val < best_val:
            best_val = val
            best_x = trial.copy()
        else:
            # Try half step
            trial = best_x - 0.5 * step
            trial = np.maximum(trial, 0.01)
            val = _objective(trial)
            if val < best_val:
                best_val = val
                best_x = trial.copy()
            else:
                # Try even smaller
                trial = best_x - 0.1 * step
                trial = np.maximum(trial, 0.01)
                val = _objective(trial)
                if val < best_val:
                    best_val = val
                    best_x = trial.copy()
                else:
                    break
    
    return best_x, best_val

def _toeplitz_quadratic_optimize(x0, max_iters=200):
    """
    Use the Toeplitz structure of autoconvolution for a quadratic optimization.
    
    The autoconvolution can be written as x^T T_k x for each lag k,
    where T_k is a Toeplitz matrix.
    
    We minimize: max_k x^T T_k x / (1^T x)^2
    
    Using a Dinkelbach-like or parametric approach:
    For a given threshold t, check if max_k x^T T_k x <= t * (1^T x)^2
    
    This is equivalent to checking if the maximum eigenvalue of a certain
    matrix is bounded.
    
    Practically, we use a penalty method:
    Minimize: (1/N) sum_k (x^T T_k x)^2 / (1^T x)^4
    which approximates the max via the L2 norm of all lags.
    """
    best_x = x0.copy()
    best_val = _objective(best_x)
    
    # Use L2 norm of autoconvolution as a smooth proxy for max
    # This is different from soft-max: it's the RMS of all lags
    def _rms_obj(x):
        s = np.sum(x)
        if abs(s) < 1e-10:
            return 1e10, np.zeros(N)
        ac = _autoconv_fft(x)
        rms = np.sqrt(np.mean(ac**2))
        val = rms / (s * s)
        
        # Gradient
        L_fft = 1
        while L_fft < 3 * N:
            L_fft *= 2
        
        # d(rms)/dx = (1/(2*rms)) * d(sum ac^2)/dx = (1/(2*rms)) * 2 * sum ac[k] * d(ac[k])/dx
        # = (1/rms) * sum_k ac[k] * d(ac[k])/dx
        
        g = ac / rms  # weight for each lag
        G = np.fft.rfft(g, n=L_fft)
        Xf = np.fft.rfft(x, n=L_fft)
        
        term1 = np.fft.irfft(np.conj(G) * Xf, n=L_fft)[:N]
        term2 = np.fft.irfft(G * Xf, n=L_fft)[:N]
        
        d_rms_dx = term1 + term2
        
        grad = (1.0 / (s * s)) * d_rms_dx - (2.0 * rms / (s ** 3)) * np.ones(N)
        
        return val, grad
    
    try:
        result = minimize(_rms_obj, best_x, method='L-BFGS-B', jac=True,
                        options={'maxiter': 100, 'gtol': 1e-8, 'ftol': 1e-14})
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x.copy()
    except Exception:
        pass
    
    if _check_deadline():
        return best_x, best_val
    
    # Now refine with peak-targeting
    return _iterative_peak_reduction(best_x, max_rounds=50)

def construct():
    best_x = np.ones(N)
    best_val = _objective(best_x)
    
    if _check_deadline():
        return best_x
    
    # Phase 1: Structured initializations
    t = np.linspace(0, 1, N)
    initializations = [
        np.ones(N),
        np.linspace(2.0, 1.0, N),
        np.linspace(1.0, 2.0, N),
        1.0 + 0.5 * np.cos(np.pi * t),
        1.0 + np.cos(np.pi * t),
        1.0 + 0.3 * np.sin(2 * np.pi * t),
        1.0 + 0.5 * np.sin(np.pi * t),
        np.exp(-2.0 * t),
        np.exp(2.0 * t - 2.0),
        1.0 + 0.2 * np.cos(2 * np.pi * 3 * t),
        1.0 + 0.3 * np.cos(2 * np.pi * 5 * t),
    ]
    
    for r in [0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 1.1, 1.2, 1.5, 2.0]:
        initializations.append(r ** np.arange(N))
    
    # Add some random smooth initializations
    rng_init = np.random.default_rng(123)
    for _ in range(20):
        freqs = rng_init.uniform(0.1, 3.0, size=5)
        phases = rng_init.uniform(0, 2*np.pi, size=5)
        amps = rng_init.uniform(0.1, 0.5, size=5)
        sig = 1.0 + sum(a * np.cos(2*np.pi*f*t + p) for a, f, p in zip(amps, freqs, phases))
        initializations.append(sig)
    
    for x0 in initializations:
        if _check_deadline():
            break
        val = _objective(x0)
        if val < best_val:
            best_val = val
            best_x = x0.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 2: Toeplitz-based quadratic optimization (new structural approach)
    best_x, best_val = _toeplitz_quadratic_optimize(best_x, max_iters=200)
    
    if _check_deadline():
        return best_x
    
    # Phase 3: Iterative peak reduction (targeted lag optimization)
    best_x, best_val = _iterative_peak_reduction(best_x, max_rounds=100)
    
    if _check_deadline():
        return best_x
    
    # Phase 4: Spectral envelope optimization
    best_x, best_val = _spectral_envelope_optimize(best_x, max_iters=50)
    
    if _check_deadline():
        return best_x
    
    # Phase 5: Another round of peak reduction with smaller steps
    best_x, best_val = _iterative_peak_reduction(best_x, max_rounds=50)
    
    if _check_deadline():
        return best_x
    
    # Phase 6: Final L-BFGS with peak penalty
    def _final_peak_obj(x):
        return _peak_penalty_objective(x, [], gamma=1.0)
    
    try:
        result = minimize(_final_peak_obj, best_x, method='L-BFGS-B', jac=True,
                        options={'maxiter': 100, 'gtol': 1e-9, 'ftol': 1e-15})
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x.copy()
    except Exception:
        pass
    
    if _check_deadline():
        return best_x
    
    # Phase 7: Final polish
    best_x, best_val = _iterative_peak_reduction(best_x, max_rounds=30)
    
    return best_x
