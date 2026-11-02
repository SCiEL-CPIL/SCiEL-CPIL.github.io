import time
import numpy as np
from scipy import optimize
from scipy.fft import fft, ifft

N = 600

def _fft_convolve(h):
    """Compute autoconvolution via FFT, returns full-length convolution."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize <<= 1
    H = fft(h, fsize)
    conv = np.real(ifft(H * H, fsize))
    return conv[:size]

def compute_score(h):
    s = np.sum(h)
    if s < 1e-12:
        return 0.0
    conv = _fft_convolve(h)
    max_conv = np.max(conv)
    return max_conv / (s * s)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    # Start with uniform as baseline
    best_h = np.ones(N, dtype=np.float64)
    best_score = compute_score(best_h)

    # DCT basis representation: h[i] = sum_k c_k * phi_k(i)
    # Use first M DCT-II modes (orthonormal on [0, N-1])
    M = 12  # number of basis functions
    
    # Build DCT-II basis matrix (N x M)
    # phi_k(i) = sqrt(2/N) * cos(pi * k * (i + 0.5) / N) for k >= 1
    # phi_0(i) = 1/sqrt(N)
    i_idx = np.arange(N, dtype=np.float64)
    basis = np.zeros((N, M))
    basis[:, 0] = 1.0 / np.sqrt(N)
    for k in range(1, M):
        basis[:, k] = np.sqrt(2.0 / N) * np.cos(np.pi * k * (i_idx + 0.5) / N)
    
    def coeffs_to_h(coeffs):
        """Convert non-negative coefficients to step heights."""
        c = np.maximum(coeffs, 0.0)
        h = c @ basis.T
        # Ensure non-negativity
        h = np.maximum(h, 0.0)
        return h

    def objective_dct(coeffs):
        h = coeffs_to_h(coeffs)
        s = np.sum(h)
        if s < 1e-12:
            return 1.0
        conv = _fft_convolve(h)
        return np.max(conv) / (s * s)

    # Multiple initial guesses in coefficient space
    # The uniform function corresponds to c_0 = sqrt(N), all others 0
    initial_guesses = []
    
    # Pure uniform
    c0 = np.zeros(M)
    c0[0] = np.sqrt(N)
    initial_guesses.append(c0.copy())
    
    # Uniform + small amount of various modes
    np.random.seed(42)
    for trial in range(15):
        c = np.zeros(M)
        c[0] = np.sqrt(N) * (0.8 + 0.4 * np.random.random())
        # Add random small perturbations to other modes
        for k in range(1, M):
            c[k] = np.random.random() * 2.0 * np.sqrt(N) / M
        initial_guesses.append(c)
    
    # Structured guesses: emphasize certain modes
    # Mode 1 (lowest frequency non-constant) - gives left-right asymmetry
    c1 = np.zeros(M)
    c1[0] = np.sqrt(N)
    c1[1] = 3.0 * np.sqrt(N) / M
    initial_guesses.append(c1)
    
    # Mode 2
    c2 = np.zeros(M)
    c2[0] = np.sqrt(N)
    c2[2] = 3.0 * np.sqrt(N) / M
    initial_guesses.append(c2)
    
    # Two-block-like: combine modes 1 and 3
    c3 = np.zeros(M)
    c3[0] = np.sqrt(N)
    c3[1] = 2.0 * np.sqrt(N) / M
    c3[3] = 2.0 * np.sqrt(N) / M
    initial_guesses.append(c3)
    
    # Higher frequency emphasis
    c4 = np.zeros(M)
    c4[0] = np.sqrt(N)
    for k in range(1, M, 2):
        c4[k] = 1.5 * np.sqrt(N) / M
    initial_guesses.append(c4)
    
    bounds = [(0.0, None)] * M
    
    for guess in initial_guesses:
        if time.time() > deadline - 30:
            break
        try:
            res = optimize.minimize(objective_dct, guess, method='Nelder-Mead',
                                     bounds=bounds,
                                     options={'xatol': 1e-6, 'fatol': 1e-8, 'maxiter': 500})
            if res.fun < best_score:
                h_candidate = coeffs_to_h(res.x)
                real_score = compute_score(h_candidate)
                if real_score < best_score:
                    best_score = real_score
                    best_h = h_candidate
        except Exception:
            pass

    # Also try the two-block parameterization from the original
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.5 * (0.5 / N)
    
    def param_two_blocks(params):
        c1, w1, c2, w2 = params
        h = np.zeros(N)
        mask1 = (x >= c1 - w1) & (x <= c1 + w1)
        mask2 = (x >= c2 - w2) & (x <= c2 + w2)
        h[mask1] = 1.0
        h[mask2] = 1.0
        return h
    
    def objective(params):
        h = param_two_blocks(params)
        if np.sum(h) < 1e-12:
            return 1.0
        return compute_score(h)
    
    initial_guesses_block = [
        [0.20, 0.08, -0.20, 0.08],
        [0.18, 0.10, -0.18, 0.10],
        [0.15, 0.12, -0.15, 0.12],
        [0.22, 0.06, -0.22, 0.06],
        [0.10, 0.15, -0.10, 0.15],
        [0.24, 0.04, -0.24, 0.04],
    ]
    
    bounds_block = [(-0.25, 0.25), (0.001, 0.25), (-0.25, 0.25), (0.001, 0.25)]
    
    for guess in initial_guesses_block:
        if time.time() > deadline - 25:
            break
        try:
            res = optimize.minimize(objective, guess, method='Nelder-Mead',
                                     bounds=bounds_block,
                                     options={'xatol': 1e-5, 'fatol': 1e-7, 'maxiter': 600})
            if res.fun < best_score:
                h_candidate = param_two_blocks(res.x)
                real_score = compute_score(h_candidate)
                if real_score < best_score:
                    best_score = real_score
                    best_h = h_candidate
        except Exception:
            pass

    # Coordinate descent refinement on individual heights using FFT
    for outer in range(10):
        if time.time() > deadline - 15:
            break
        
        conv = _fft_convolve(best_h)
        k_peak = np.argmax(conv)
        
        # Priority: indices that contribute to the peak
        # For peak at index k_peak, contributions come from h[i]*h[k_peak-i]
        # Focus on indices near the center of the support
        priority_order = list(range(N))
        
        for i in priority_order:
            if time.time() > deadline - 15:
                break
            if best_h[i] < 1e-8:
                continue
            
            # Try reducing h[i] by 50%
            h_new = best_h.copy()
            h_new[i] *= 0.5
            s_new = np.sum(h_new)
            if s_new < 1e-12:
                continue
            conv_new = _fft_convolve(h_new)
            score_new = np.max(conv_new) / (s_new * s_new)
            
            if score_new < best_score - 1e-12:
                best_score = score_new
                best_h = h_new
        
        # Try small increases on zeros
        for i in priority_order:
            if time.time() > deadline - 15:
                break
            if best_h[i] > 1e-8:
                continue
            
            h_new = best_h.copy()
            h_new[i] = 0.05
            s_new = np.sum(h_new)
            conv_new = _fft_convolve(h_new)
            score_new = np.max(conv_new) / (s_new * s_new)
            
            if score_new < best_score - 1e-12:
                best_score = score_new
                best_h = h_new
    
    # Final fine-tuning: try small multiplicative adjustments
    if time.time() < deadline - 5:
        for scale in [0.95, 0.98, 1.02, 1.05]:
            if time.time() > deadline - 5:
                break
            h_try = best_h * scale
            s_try = np.sum(h_try)
            if s_try < 1e-12:
                continue
            conv_try = _fft_convolve(h_try)
            score_try = np.max(conv_try) / (s_try * s_try)
            if score_try < best_score - 1e-12:
                best_score = score_try
                best_h = h_try
    
    return best_h.tolist()
