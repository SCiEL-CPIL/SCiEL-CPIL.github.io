import numpy as np
import time
from scipy.optimize import minimize

N = 400

def _check_deadline(start_time, limit=140):
    return time.time() - start_time < limit

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT with 3N-1 padding."""
    n = len(x)
    size = 3 * n - 1
    f = np.fft.rfft(x, size)
    ac = np.fft.irfft(f * np.conj(f), size)[:2*n-1]
    return ac

def _objective_only(x):
    """Compute max(|autoconv|) / (sum x)^2."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return np.inf
    ac = _autoconv_fft(x)
    return np.max(np.abs(ac)) / (s * s)

def _soft_max_and_grad(x, beta=50.0):
    """
    Compute smooth approximation of max(|ac|) and its gradient.
    Uses soft-max with temperature 1/beta.
    """
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-15:
        return 0.0, np.zeros_like(x)
    
    # Soft-max: (1/beta) * log(sum(exp(beta * (abs_ac_i - m))))
    shifted = beta * (abs_ac - m)
    shifted = np.clip(shifted, -100, 0)
    exp_shifted = np.exp(shifted)
    w_sum = np.sum(exp_shifted)
    if w_sum < 1e-15:
        return m, np.zeros_like(x)
    
    soft_max_val = m + np.log(w_sum) / beta
    weights = exp_shifted / w_sum
    
    # Gradient of soft_max w.r.t. x
    # d(soft_max)/dx_k = sum_i w_i * sign(ac_i) * d(ac_i)/dx_k
    # d(ac_i)/dx_k = x_{i-k} + x_{k-i} (for valid indices)
    
    sign_ac = np.sign(ac) * weights
    L = len(ac)
    
    # Compute gradient efficiently using FFT
    # grad_k = sum_i sign_ac[i] * (x[i-k] + x[k-i])
    
    # For N=400, direct computation is O(N^2) = 160K ops, acceptable
    grad_sm = np.zeros(N)
    for k in range(N):
        # A[k] = sum_{j=0}^{N-1} sign_ac[j+k] * x[j] where j+k in [0,L)
        ia = np.arange(N) + k
        valid_a = (ia >= 0) & (ia < L)
        term_a = np.sum(sign_ac[ia[valid_a]] * x[np.arange(N)[valid_a]])
        
        # B[k] = sum_{j=0}^{N-1} sign_ac[k-j] * x[j] where k-j in [0,L)
        ib = k - np.arange(N)
        valid_b = (ib >= 0) & (ib < L)
        term_b = np.sum(sign_ac[ib[valid_b]] * x[np.arange(N)[valid_b]])
        
        grad_sm[k] = term_a + term_b
    
    return soft_max_val, grad_sm

def _objective_and_grad(x, beta=50.0):
    """
    Objective: max(|ac|) / (sum x)^2 with gradient.
    Uses soft-max approximation for differentiability.
    """
    s = np.sum(x)
    if abs(s) < 1e-10:
        return np.inf, np.zeros_like(x)
    
    soft_max_val, grad_sm = _soft_max_and_grad(x, beta)
    
    # Objective = soft_max / s^2
    val = soft_max_val / (s * s)
    
    # Gradient: d/dx_k [soft_max / s^2] = grad_sm_k / s^2 - 2 * soft_max / s^3
    grad_obj = grad_sm / (s * s) - 2 * soft_max_val / (s * s * s)
    
    return val, grad_obj

def _spectral_init(rng, n_freqs=10):
    """
    Generate initial signal via spectral phase retrieval.
    Create random phases in frequency domain, take inverse FFT.
    """
    # Create random spectrum with random phases
    freq_size = N // 2 + 1
    magnitude = np.ones(freq_size)
    # Add some spectral shaping
    t = np.linspace(0, 1, freq_size)
    for shape in [1.0, 2.0, 0.5, 1.5]:
        magnitude *= np.exp(-shape * t)
    
    phases = rng.uniform(0, 2*np.pi, size=freq_size)
    spectrum = magnitude * np.exp(1j * phases)
    
    # Inverse FFT to get time domain signal
    x = np.fft.irfft(spectrum, n=N)
    x = np.real(x)
    
    # Ensure reasonable scale
    if np.sum(x) != 0:
        x = x / np.sum(x) * N
    
    return x

def _spectral_phase_optimize(x0, start_time, max_iters=200, beta_start=20.0, beta_end=100.0):
    """
    Optimize spectral phases to minimize peak autoconvolution.
    Uses L-BFGS on the time-domain signal with soft-max objective.
    """
    best_x = x0.copy()
    best_val = _objective_only(best_x)
    
    beta = beta_start
    for phase_iter in range(max_iters):
        if not _check_deadline(start_time):
            break
        
        # Increase beta for sharper max approximation
        beta = beta_start + (beta_end - beta_start) * (phase_iter / max_iters)
        
        # L-BFGS optimization
        result = minimize(
            fun=lambda x: _objective_and_grad(x, beta)[0],
            x0=best_x,
            jac=lambda x: _objective_and_grad(x, beta)[1],
            method='L-BFGS-B',
            options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if result.success and result.fun < best_val:
            best_x = result.x.copy()
            best_val = result.fun
        
        if not _check_deadline(start_time):
            break
    
    return best_x, best_val

def _amplitude_optimize(x0, start_time, max_rounds=30):
    """
    Optimize amplitudes (magnitudes in frequency domain) while keeping phases fixed.
    """
    best_x = x0.copy()
    best_val = _objective_only(best_x)
    
    # Get current spectral representation
    freq_size = N // 2 + 1
    spectrum = np.fft.rfft(best_x, n=N)
    magnitudes = np.abs(spectrum)
    phases = np.angle(spectrum)
    
    for round_iter in range(max_rounds):
        if not _check_deadline(start_time):
            break
        
        # Optimize magnitudes with L-BFGS
        def amp_obj(mag_log):
            mags = np.exp(mag_log)
            spec = mags * np.exp(1j * phases)
            x = np.fft.irfft(spec, n=N)
            return _objective_only(x)
        
        def amp_grad(mag_log):
            mags = np.exp(mag_log)
            spec = mags * np.exp(1j * phases)
            x = np.fft.irfft(spec, n=N)
            val, grad = _objective_and_grad(x, beta=50.0)
            
            # Chain rule: d(x)/d(mag_log) via FFT
            # This is complex; use finite differences for now
            eps = 1e-6
            grad_mag = np.zeros_like(mag_log)
            for i in range(min(freq_size, 20)):  # Only optimize first 20 for speed
                mag_log_plus = mag_log.copy()
                mag_log_plus[i] += eps
                mags_plus = np.exp(mag_log_plus)
                spec_plus = mags_plus * np.exp(1j * phases)
                x_plus = np.fft.irfft(spec_plus, n=N)
                val_plus = _objective_only(x_plus)
                grad_mag[i] = (val_plus - val) / eps
            
            return grad_mag
        
        # Simple coordinate descent on log-magnitudes
        mag_log = np.log(magnitudes + 1e-10)
        step_size = 0.1
        
        for coord in range(min(freq_size, 30)):
            if not _check_deadline(start_time):
                break
            
            best_coord_val = best_val
            best_coord_mag = mag_log[coord]
            
            for delta in [-0.5, -0.3, -0.1, 0.1, 0.3, 0.5]:
                mag_log_trial = mag_log.copy()
                mag_log_trial[coord] += delta
                mags_trial = np.exp(mag_log_trial)
                spec_trial = mags_trial * np.exp(1j * phases)
                x_trial = np.fft.irfft(spec_trial, n=N)
                val_trial = _objective_only(x_trial)
                
                if val_trial < best_coord_val:
                    best_coord_val = val_trial
                    best_coord_mag = mag_log_trial[coord]
            
            mag_log[coord] = best_coord_mag
        
        # Update best
        mags_final = np.exp(mag_log)
        spec_final = mags_final * np.exp(1j * phases)
        x_final = np.fft.irfft(spec_final, n=N)
        val_final = _objective_only(x_final)
        
        if val_final < best_val:
            best_x = x_final.copy()
            best_val = val_final
            magnitudes = mags_final.copy()
    
    return best_x, best_val

def _peak_targeted_polish(x0, start_time, max_lags=50):
    """
    Target the largest autoconvolution peaks and try to reduce them.
    """
    best_x = x0.copy()
    best_val = _objective_only(best_x)
    
    for polish_iter in range(100):
        if not _check_deadline(start_time):
            break
        
        ac = _autoconv_fft(best_x)
        abs_ac = np.abs(ac)
        
        # Find top peaks
        top_lags = np.argsort(abs_ac)[-max_lags:]
        
        improved = False
        for lag in top_lags:
            if not _check_deadline(start_time):
                break
            
            # Try to reduce this peak by adjusting nearby coefficients
            center = lag if lag < N else lag - (N - 1)
            radius = 5
            
            for delta in [-0.2, -0.1, -0.05, 0.05, 0.1, 0.2]:
                x_trial = best_x.copy()
                start_idx = max(0, center - radius)
                end_idx = min(N, center + radius + 1)
                
                # Apply small perturbation to reduce peak
                for k in range(start_idx, end_idx):
                    x_trial[k] += delta * np.cos(np.pi * (k - center) / radius)
                
                val_trial = _objective_only(x_trial)
                if val_trial < best_val:
                    best_x = x_trial.copy()
                    best_val = val_trial
                    improved = True
                    break
        
        if not improved:
            break
    
    return best_x, best_val

def _structured_init(rng):
    """Generate diverse structured initializations."""
    initializations = []
    t = np.linspace(0, 1, N)
    
    # Basic patterns
    initializations.append(np.ones(N))
    initializations.append(np.linspace(1.0, 2.0, N))
    initializations.append(np.linspace(2.0, 1.0, N))
    
    # Exponential decays
    for r in [0.9, 0.8, 0.7, 0.6, 0.5, 1.2, 1.5, 2.0]:
        initializations.append(r ** np.arange(N))
    
    # Cosine patterns
    for freq in [1, 2, 3, 5, 10]:
        initializations.append(1.0 + 0.5 * np.cos(2 * np.pi * freq * t))
        initializations.append(1.0 + np.cos(2 * np.pi * freq * t))
    
    # Sine patterns
    for freq in [1, 2, 3, 5, 10]:
        initializations.append(1.0 + 0.5 * np.sin(2 * np.pi * freq * t))
        initializations.append(1.0 + np.sin(2 * np.pi * freq * t))
    
    # Signed patterns
    for freq in [1, 2, 3, 5, 10, 20]:
        initializations.append(np.sign(np.sin(2 * np.pi * freq * t)))
        initializations.append(1.0 + 0.5 * np.sign(np.sin(2 * np.pi * freq * t)))
    
    # Gaussian-like
    for sigma in [0.1, 0.2, 0.3, 0.5]:
        initializations.append(np.exp(-((t - 0.5)**2) / (2 * sigma**2)))
    
    # Random smooth
    for _ in range(10):
        x = rng.normal(0, 1, N)
        # Smooth it
        kernel = np.exp(-np.arange(-5, 6)**2 / 10)
        x = np.convolve(x, kernel, mode='same')
        initializations.append(x)
    
    return initializations

def _random_init(rng, n_samples=20):
    """Generate random initializations."""
    initializations = []
    for _ in range(n_samples):
        x = rng.normal(0, 1, N)
        # Random smoothing
        if rng.random() > 0.5:
            kernel_size = rng.integers(3, 11)
            kernel = np.ones(kernel_size) / kernel_size
            x = np.convolve(x, kernel, mode='same')
        initializations.append(x)
    return initializations

def construct():
    START_TIME = time.time()
    TIME_LIMIT = 140  # Safety margin before 150s kill
    
    rng = np.random.default_rng(42)
    
    best_x = np.ones(N)
    best_val = _objective_only(best_x)
    
    # Phase 1: Try diverse initializations
    all_inits = _structured_init(rng) + _random_init(rng, 20)
    
    for x0 in all_inits:
        if not _check_deadline(START_TIME):
            break
        val = _objective_only(x0)
        if val < best_val:
            best_val = val
            best_x = x0.copy()
    
    # Phase 2: Spectral phase optimization from best init
    best_x, best_val = _spectral_phase_optimize(best_x, START_TIME, max_iters=50, beta_start=20.0, beta_end=80.0)
    
    if not _check_deadline(START_TIME):
        return best_x
    
    # Phase 3: Amplitude optimization
    best_x, best_val = _amplitude_optimize(best_x, START_TIME, max_rounds=20)
    
    if not _check_deadline(START_TIME):
        return best_x
    
    # Phase 4: Peak-targeted polish
    best_x, best_val = _peak_targeted_polish(best_x, START_TIME, max_lags=30)
    
    if not _check_deadline(START_TIME):
        return best_x
    
    # Phase 5: Final L-BFGS polish with high beta
    if _check_deadline(START_TIME, TIME_LIMIT - 30):
        result = minimize(
            fun=lambda x: _objective_and_grad(x, beta=100.0)[0],
            x0=best_x,
            jac=lambda x: _objective_and_grad(x, beta=100.0)[1],
            method='L-BFGS-B',
            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        if result.success and result.fun < best_val:
            best_x = result.x.copy()
            best_val = result.fun
    
    return best_x
