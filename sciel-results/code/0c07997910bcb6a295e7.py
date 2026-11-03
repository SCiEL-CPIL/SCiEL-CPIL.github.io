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

def _spectral_refinement(h_init, max_iter=500, tol=1e-6):
    """
    Iteratively refine the spectrum to minimize peak autoconvolution.
    
    Strategy: Work in frequency domain. The circular autoconvolution is
    IFFT(|H|^2). We iteratively reduce |H|^2 at frequencies that contribute
    most to the peak of the autoconvolution.
    
    For real h, H[k] = conj(H[N-k]), so we only need to handle k=0..N//2.
    """
    n = len(h_init)
    h = h_init.astype(float).copy()
    
    # Get initial DFT
    H = np.fft.fft(h)
    
    best_h = h.copy()
    best_score = _score(h)
    
    t_start = time.time()
    deadline = t_start + 90.0
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Compute circular autoconvolution from |H|^2
        H2 = np.abs(H) ** 2
        ac_circ = np.fft.ifft(H2).real
        
        # Find peak locations
        peak_val = np.max(np.abs(ac_circ))
        if peak_val < 1e-12:
            break
        
        # Find where the peak occurs
        peak_idx = np.argmax(np.abs(ac_circ))
        
        # The peak at index peak_idx is sum_k |H[k]|^2 * exp(2*pi*i*k*peak_idx/n)
        # To reduce the peak, we want to reduce |H[k]|^2 where the phase
        # exp(2*pi*i*k*peak_idx/n) aligns with the peak direction.
        
        # Simpler approach: reduce |H[k]|^2 proportionally to its contribution
        # to the peak. The contribution of frequency k to ac_circ[peak_idx] is
        # |H[k]|^2 * exp(2*pi*i*k*peak_idx/n). The real part contributes to
        # the real value, imaginary to imaginary.
        
        # Even simpler: iteratively clip |H|^2 to reduce peak.
        # Use a water-filling approach: reduce the largest |H[k]|^2 values.
        
        H2_real = H2.copy()
        
        # Try reducing the top few frequencies
        top_k = 10
        H2_sorted_idx = np.argsort(H2_real)[::-1]
        
        # Reduce the top frequencies
        for _ in range(top_k):
            k = H2_sorted_idx[_]
            if k == 0:
                continue  # Don't reduce DC
            H2_real[k] *= 0.95
            # Also reduce the conjugate for real signal
            if k < n:
                H2_real[(n - k) % n] = H2_real[k]
        
        # Reconstruct H from modified H2
        # We need to maintain the phase of H
        H_phase = np.angle(H)
        H_mag = np.sqrt(H2_real)
        H_new = H_mag * np.exp(1j * H_phase)
        
        # Ensure real signal: H[k] = conj(H[n-k])
        for k in range(1, n // 2 + 1):
            H_new[n - k] = np.conj(H_new[k])
        if n % 2 == 0:
            H_new[n // 2] = np.real(H_new[n // 2]) + 0j
        H_new[0] = np.real(H_new[0]) + 0j
        
        # Inverse DFT to get new h
        h_new = np.fft.ifft(H_new).real
        
        # Check score
        new_score = _score(h_new)
        if new_score < best_score:
            best_score = new_score
            best_h = h_new.copy()
            H = H_new.copy()
            h = h_new.copy()
        
        # Check convergence
        if it > 0 and abs(peak_val - np.max(np.abs(ac_circ))) < tol:
            break
    
    return best_h, best_score

def construct():
    t_start = time.time()
    deadline = t_start + 140.0
    
    best_score = np.inf
    best_h = np.ones(N) / N
    
    base_score = _score(best_h)
    if base_score < best_score:
        best_score = base_score
        best_h = best_h.copy()
    
    # Generate candidate initial sequences
    i = np.arange(N)
    candidates = []
    
    # Quadratic phase chirps
    candidates.append(np.cos(2.0 * np.pi * i**2 / N))
    candidates.append(np.cos(np.pi * i**2 / N))
    candidates.append(np.cos(2.0 * np.pi * i**2 / (N+1)))
    candidates.append(np.cos(4.0 * np.pi * i**2 / N))
    
    # Zadoff-Chu like
    candidates.append(np.cos(np.pi * i * (i + 1) / N))
    candidates.append(np.cos(np.pi * i * (i + 1) / (N+1)))
    candidates.append(np.cos(2.0 * np.pi * i * (i + 1) / N))
    
    # Evaluate candidates and pick best initial
    best_init = None
    best_init_score = np.inf
    for h_cand in candidates:
        s = np.sum(h_cand)
        if abs(s) < 1e-6:
            continue
        val = _score(h_cand)
        if val < best_init_score:
            best_init_score = val
            best_init = h_cand.copy()
    
    if best_init is not None:
        best_score = min(best_score, best_init_score)
        best_h = best_init.copy()
    
    # Spectral refinement on the best candidate
    if best_init is not None and time.time() < deadline - 20:
        h_refined, score_refined = _spectral_refinement(best_init)
        if score_refined < best_score:
            best_score = score_refined
            best_h = h_refined.copy()
    
    # Final check against baseline
    if best_score > base_score:
        best_score = base_score
        best_h = np.ones(N) / N
    
    return best_h

# EVOLVE-BLOCK-END
