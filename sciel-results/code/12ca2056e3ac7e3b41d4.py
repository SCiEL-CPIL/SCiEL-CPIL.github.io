"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution using FFT for speed."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return np.max(np.abs(ac))

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)

def _spectral_smooth_objective(h):
    """
    Smooth proxy for max|autoconv|/(sum h)^2 using spectral moments.
    
    The autoconvolution ac = h * h has DFT: AC[k] = H[k]^2.
    By Parseval: sum_k |ac[k]|^2 = (1/fsize) * sum_k |H[k]^2|^2 = (1/fsize) * sum_k |H[k]|^4
    
    We use a soft-max approximation: max|ac| ~ (sum |ac|^p)^(1/p) for large p.
    In frequency domain: |ac[k]| = |IDFT(H^2)[k]|, which is hard to compute smoothly.
    
    Instead, we use the fact that for circulant approximation (periodic conv),
    the eigenvalues are H[k], and the spectral norm relates to max |H[k]|.
    
    We minimize: (sum_k |H[k]|^4) / (H[0]^4) as a smooth proxy,
    which encourages flat spectrum relative to DC.
    """
    n = len(h)
    fsize = 1
    while fsize < 2 * n - 1:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    H_full = np.zeros(fsize, dtype=complex)
    H_full[:len(H)] = H
    H_full[fsize // 2 + 1:] = np.conj(H[1:fsize // 2][::-1])
    
    H0 = np.abs(H[0])
    if H0 < 1e-9:
        return 1e10
    
    # sum |H[k]|^4 / |H[0]|^4
    H_mag4 = np.abs(H_full) ** 4
    return np.sum(H_mag4) / (H0 ** 4)

def _spectral_projected_gradient():
    """
    Directly optimize in frequency domain using projected gradient descent.
    
    We work with the DFT coefficients H[k]. The constraint is that h is real,
    so H must satisfy H[-k] = conj(H[k]). We parameterize by the first half.
    
    Objective: minimize sum_k |H[k]|^4 / |H[0]|^4 (smooth proxy for peak ratio)
    Constraint: H[0] = 1 (normalization, sum h = 1)
    """
    t_start = time.time()
    deadline = t_start + 90.0
    
    n = N
    fsize = 1
    while fsize < 2 * n - 1:
        fsize *= 2
    
    # Parameterize: H[0] is real (DC), H[1..fsize//2-1] are complex, H[fsize//2] is real
    # We'll optimize over the real and imaginary parts of H[1..fsize//2-1] and H[fsize//2]
    
    def h_from_H(H_half):
        """Reconstruct h from half-spectrum (with H[0]=1)."""
        H_full = np.zeros(fsize, dtype=complex)
        H_full[0] = 1.0  # Normalization
        for k in range(1, fsize // 2):
            H_full[k] = H_half[k - 1]
            H_full[fsize - k] = np.conj(H_half[k - 1])
        if fsize % 2 == 0:
            # Nyquist frequency is real
            H_full[fsize // 2] = np.real(H_half[fsize // 2 - 1])
        h = np.fft.irfft(H_full, fsize)[:n]
        return h
    
    def H_half_from_h(h):
        """Extract half-spectrum from h."""
        H = np.fft.rfft(h, fsize)
        # H[0] should be 1 after normalization
        H_half = H[1:]  # indices 1 to fsize//2
        return H_half
    
    def objective_and_grad(H_half):
        """Compute objective and gradient with respect to H_half."""
        H_full = np.zeros(fsize, dtype=complex)
        H_full[0] = 1.0
        for k in range(1, fsize // 2):
            H_full[k] = H_half[k - 1]
            H_full[fsize - k] = np.conj(H_half[k - 1])
        if fsize % 2 == 0:
            H_full[fsize // 2] = np.real(H_half[fsize // 2 - 1])
        
        H0 = 1.0
        H_mag = np.abs(H_full)
        H_mag4 = H_mag ** 4
        obj = np.sum(H_mag4) / (H0 ** 4)
        
        # Gradient: d/dH_full[k] of sum |H|^4 = 4 * |H|^2 * conj(H)
        grad_full = 4.0 * H_mag ** 2 * np.conj(H_full)
        grad_full /= H0 ** 4
        
        # Project gradient back to H_half parameters
        grad_half = np.zeros_like(H_half)
        for k in range(1, fsize // 2):
            # d/dH_half[k-1] = grad_full[k] + conj(grad_full[fsize-k])
            grad_half[k - 1] = grad_full[k] + np.conj(grad_full[fsize - k])
        if fsize % 2 == 0:
            # Nyquist: only real part matters
            grad_half[fsize // 2 - 1] = np.real(grad_full[fsize // 2])
        
        return obj, grad_half.real, grad_half.imag
    
    # Initial guess: flat spectrum (all |H[k]| = 1/sqrt(fsize) for k>0, H[0]=1)
    # Actually, for a delta-like autoconv, we want |H[k]| constant.
    # Start with chirp-like spectrum
    i = np.arange(n)
    h_init = np.exp(1j * np.pi * i ** 2 / n).real
    h_init = h_init / np.sum(h_init)
    H_half_init = H_half_from_h(h_init)
    
    best_h = h_init.copy()
    best_score = _score(best_h)
    
    # Projected gradient descent
    lr = 0.01
    for iteration in range(500):
        if time.time() > deadline - 10:
            break
        
        obj, grad_re, grad_im = objective_and_grad(H_half_init)
        
        # Update
        H_half_init = H_half_init - lr * np.concatenate([grad_re, grad_im]) if False else H_half_init
        # Actually H_half_init is complex, let's handle properly
        H_half_new = H_half_init.copy()
        H_half_new.real -= lr * grad_re
        H_half_new.imag -= lr * grad_im
        H_half_init = H_half_new
        
        # Reconstruct and check
        h_new = h_from_H(H_half_init)
        s = np.sum(h_new)
        if abs(s) > 1e-6:
            h_new = h_new / s
            score_new = _score(h_new)
            if score_new < best_score:
                best_score = score_new
                best_h = h_new.copy()
        
        # Adaptive learning rate
        if iteration % 50 == 0:
            lr *= 0.8
    
    # Local search refinement
    if time.time() < deadline - 5:
        h_refined = best_h.copy()
        best_local = best_score
        for _ in range(5):
            improved = False
            for idx in range(n):
                if time.time() > deadline - 2:
                    break
                h_refined[idx] = -h_refined[idx]
                new_val = _score(h_refined)
                if new_val < best_local:
                    best_local = new_val
                    improved = True
                else:
                    h_refined[idx] = -h_refined[idx]
            if not improved:
                break
        if best_local < best_score:
            best_score = best_local
            best_h = h_refined.copy()
    
    return best_h, best_score

def _zadoff_chu_search():
    """
    Search for Zadoff-Chu-like sequences which have perfectly flat magnitude spectrum.
    These minimize the peak-to-average power ratio of the autoconvolution.
    """
    t_start = time.time()
    deadline = t_start + 60.0
    
    i = np.arange(N)
    best_score = np.inf
    best_h = np.ones(N) / N
    
    # Zadoff-Chu: h[i] = exp(-1j * pi * u * i * (i+1) / N) for odd N
    # For even N, use quadratic chirp
    for u in [1, 2, 3, 4, 5, 7, 11]:
        if time.time() > deadline - 10:
            break
        # Quadratic chirp (works for any N)
        h = np.exp(1j * np.pi * u * i ** 2 / N)
        h_real = h.real
        h_imag = h.imag
        
        # Try real part, imaginary part, and their combinations
        for variant in [h_real, h_imag, h_real + h_imag, h_real - h_imag]:
            s = np.sum(variant)
            if abs(s) < 1e-9:
                continue
            h_norm = variant / s
            score = _score(h_norm)
            if score < best_score:
                best_score = score
                best_h = h_norm.copy()
    
    # Also try cosine chirps with different phases
    for alpha in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]:
        if time.time() > deadline - 10:
            break
        h = np.cos(np.pi * alpha * i ** 2 / N)
        s = np.sum(h)
        if abs(s) < 1e-9:
            continue
        h_norm = h / s
        score = _score(h_norm)
        if score < best_score:
            best_score = score
            best_h = h_norm.copy()
    
    # Local search on best
    if time.time() < deadline - 5:
        h_refined = best_h.copy()
        best_local = best_score
        for _ in range(5):
            improved = False
            for idx in range(N):
                if time.time() > deadline - 2:
                    break
                h_refined[idx] = -h_refined[idx]
                new_val = _score(h_refined)
                if new_val < best_local:
                    best_local = new_val
                    improved = True
                else:
                    h_refined[idx] = -h_refined[idx]
            if not improved:
                break
        if best_local < best_score:
            best_score = best_local
            best_h = h_refined.copy()
    
    return best_h, best_score

def construct():
    t_start = time.time()
    deadline = t_start + 115.0  # Safety margin within 120s
    
    best_score = np.inf
    best_h = np.ones(N) / N
    
    # Baseline
    base_score = _score(best_h)
    best_score = min(best_score, base_score)
    
    # Strategy 1: Zadoff-Chu search (fast, good initial)
    if time.time() < deadline - 40:
        h_zc, score_zc = _zadoff_chu_search()
        if score_zc < best_score:
            best_score = score_zc
            best_h = h_zc.copy()
    
    # Strategy 2: Spectral projected gradient (direct frequency domain optimization)
    if time.time() < deadline - 20:
        h_spec, score_spec = _spectral_projected_gradient()
        if score_spec < best_score:
            best_score = score_spec
            best_h = h_spec.copy()
    
    # Final local search with more iterations
    if time.time() < deadline - 3:
        h_refined = best_h.copy()
        best_local = best_score
        for _ in range(8):
            improved = False
            for idx in range(N):
                if time.time() > deadline - 1:
                    break
                h_refined[idx] = -h_refined[idx]
                new_val = _score(h_refined)
                if new_val < best_local:
                    best_local = new_val
                    improved = True
                else:
                    h_refined[idx] = -h_refined[idx]
            if not improved:
                break
        if best_local < best_score:
            best_score = best_local
            best_h = h_refined.copy()
    
    return best_h

# EVOLVE-BLOCK-END
