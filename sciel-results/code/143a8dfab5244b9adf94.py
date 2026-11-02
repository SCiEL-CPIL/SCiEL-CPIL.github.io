import numpy as np
import time
from scipy import optimize

N = 600
_CONV_LEN = 2 * N - 1
_FFT_SIZE = 1

def _fft_convolve(h):
    n = _FFT_SIZE
    f = np.fft.rfft(h, n=n)
    g = np.fft.irfft(f * f, n=n)[:_CONV_LEN]
    return g

def compute_score(h):
    s = np.sum(h)
    if s == 0:
        return 0.0
    conv = _fft_convolve(h)
    max_conv = np.max(conv)
    return max_conv / (s * s)

def smooth_score(h, beta=100.0):
    s = np.sum(h)
    if s == 0:
        return 0.0
    conv = _fft_convolve(h)
    max_c = np.max(conv)
    log_sum_exp = max_c + (1.0/beta) * np.log(np.sum(np.exp(beta * (conv - max_c))))
    return log_sum_exp / (s * s)

def _spectral_to_heights(params):
    """
    Convert spectral parameters to time-domain step heights.
    params: [log_sigma, log_center, log_width, log_amp, phase]
    - log_sigma: log of Gaussian width in frequency domain
    - log_center: log of center frequency offset (can be negative via sign)
    - log_width: log of polynomial correction width
    - log_amp: log of amplitude
    - phase: phase offset for the polynomial term
    
    The power spectrum is: P(k) = exp(-k^2 / (2*sigma^2)) * (1 + amp * exp(-((k-center)^2) / (2*width^2)))
    Then h = IDFT(sqrt(P)) (taking real part, clipped to nonnegative)
    """
    sigma = np.exp(params[0])
    center = params[1]  # Use raw value, can be negative
    width = np.exp(params[2])
    amp = np.exp(params[3])
    phase = params[4]
    
    # Frequency bins for rfft
    freqs = np.fft.rfftfreq(N, d=1.0/N)  # cycles per unit in discrete index space
    
    # Build power spectrum
    P = np.exp(-freqs**2 / (2.0 * sigma**2))
    P *= (1.0 + amp * np.exp(-((freqs - center)**2) / (2.0 * width**2)))
    
    # Ensure non-negative
    P = np.maximum(P, 0.0)
    
    # Get amplitude spectrum (sqrt of power)
    A = np.sqrt(P)
    
    # Inverse DFT to get time domain
    # We need to construct full spectrum for irfft
    # rfft gives N//2+1 bins; we need to mirror for full ifft
    A_full = np.zeros(N)
    A_full[:N//2+1] = A
    for k in range(1, N//2):
        A_full[N-k] = A[k]
    if N % 2 == 0:
        A_full[N//2] = A[N//2]
    
    h = np.fft.irfft(A_full, n=N)
    
    # Apply phase shift
    h = np.roll(h, int(phase * N))
    
    # Clip to non-negative
    h = np.maximum(h, 0.0)
    
    return h

def construct():
    global _FFT_SIZE
    start_time = time.time()
    deadline = start_time + 110.0

    # Set FFT size
    _FFT_SIZE = 1
    while _FFT_SIZE < _CONV_LEN:
        _FFT_SIZE *= 2

    best_h = None
    best_score = float('inf')

    # Initial candidates in time domain
    w = 0.5 / N
    x = np.linspace(-0.25 + w/2, 0.25 - w/2, N)
    
    candidates = []
    candidates.append(np.ones(N) / N)
    
    sigma_g = 0.1
    h_gauss = np.exp(-x**2 / (2 * sigma_g**2))
    h_gauss /= np.sum(h_gauss)
    candidates.append(h_gauss)
    
    h_u = x**2
    h_u /= np.sum(h_u)
    candidates.append(h_u)
    
    h_sin = np.sin(np.pi * x / 0.25)**2
    h_sin /= np.sum(h_sin)
    candidates.append(h_sin)

    # Evaluate initial candidates
    for h in candidates:
        s = compute_score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()

    # Spectral optimization: parameterize via power spectrum
    # params = [log_sigma, center, log_width, log_amp, phase]
    # We'll try multiple initial points in parameter space
    
    param_inits = [
        [np.log(0.5), 0.0, np.log(0.3), np.log(0.5), 0.0],
        [np.log(1.0), 0.0, np.log(0.5), np.log(0.3), 0.0],
        [np.log(0.3), 0.1, np.log(0.2), np.log(0.8), 0.0],
        [np.log(0.7), -0.1, np.log(0.4), np.log(0.6), 0.5],
        [np.log(0.4), 0.0, np.log(0.6), np.log(0.2), 0.25],
    ]
    
    def spectral_objective(params):
        if time.time() > deadline - 5:
            return 1.0
        try:
            h = _spectral_to_heights(params)
            s = np.sum(h)
            if s < 1e-10:
                return 1.0
            # Normalize for consistent scale
            h_norm = h / s
            score = smooth_score(h_norm, beta=50.0)
            return score
        except Exception:
            return 1.0

    for p_init in param_inits:
        if time.time() > deadline - 15:
            break
        try:
            res = optimize.minimize(
                spectral_objective, p_init,
                method='L-BFGS-B',
                bounds=[(-2.0, 2.0), (-0.5, 0.5), (-2.0, 2.0), (-2.0, 2.0), (0.0, 1.0)],
                options={'maxiter': 30, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            h_opt = _spectral_to_heights(res.x)
            s = np.sum(h_opt)
            if s > 1e-10:
                h_opt = h_opt / s
                score_opt = compute_score(h_opt)
                if score_opt < best_score:
                    best_score = score_opt
                    best_h = h_opt.copy()
        except Exception:
            pass

    # Also try direct time-domain optimization on the best found so far
    if best_h is not None and time.time() < deadline - 20:
        def objective_td(h):
            s = np.sum(h)
            if s == 0:
                return 1.0
            score = smooth_score(h, beta=50.0)
            pen = 100.0 * (s - 1.0)**2
            return score + pen

        try:
            res = optimize.minimize(
                objective_td, best_h,
                method='L-BFGS-B',
                bounds=[(0, 1.0)] * N,
                options={'maxiter': 30, 'ftol': 1e-8}
            )
            h_opt = res.x
            s = np.sum(h_opt)
            if s > 1e-10:
                h_opt = h_opt / s
                score_opt = compute_score(h_opt)
                if score_opt < best_score:
                    best_score = score_opt
                    best_h = h_opt.copy()
        except Exception:
            pass

    if best_h is None:
        best_h = np.ones(N) / N

    return best_h.tolist()
