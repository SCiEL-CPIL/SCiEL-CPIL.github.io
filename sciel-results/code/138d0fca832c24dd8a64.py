"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    """
    Spectral design: represent h as 0.5 + x where x is zero-mean.
    We optimize the power spectrum (amplitudes) and phases.
    1. Candidate amplitude spectra are generated.
    2. For each, phases are optimized to minimize the 4th moment of the time-domain signal
       (proxy for PAPR reduction), which allows for maximum scaling.
    3. The resulting signal is scaled to [-0.5, 0.5] and scored.
    """
    np.random.seed(42)
    
    best_h = np.full(N, 0.5)
    best_score = np.inf
    
    k = np.arange(N)
    
    # Define a set of candidate amplitude spectra
    # We want spectra that yield high autocorrelation sidelobes.
    # Low-pass filters generally have positive autocorrelation sidelobes.
    
    candidates = []
    
    # 1. Gaussian spectra with various widths
    for sigma in [1.0, 2.0, 3.0, 5.0, 10.0]:
        d = np.minimum(k, N - k) # Distance from 0 in circular sense
        amps = np.exp(-(d**2) / (2 * sigma**2))
        candidates.append(amps)
        
    # 2. Rectangular (Box) spectra
    for width in [5, 10, 20, 50]:
        amps = np.zeros(N)
        amps[:width] = 1.0
        amps[N-width:] = 1.0
        # Symmetrize
        amps = (amps + amps[::-1]) / 2.0
        candidates.append(amps)
        
    # 3. Triangular spectra
    for width in [10, 20, 50]:
        amps = np.zeros(N)
        # Centered triangle
        d = np.minimum(k, N - k)
        amps = np.maximum(0, 1 - d/width)
        candidates.append(amps)

    # 4. Raised Cosine
    for beta in [0.5, 1.0, 2.0]:
        d = np.minimum(k, N - k)
        # Approximate raised cosine shape
        amps = np.cos(np.pi * d / (N/2))**beta
        amps = np.maximum(0, amps)
        candidates.append(amps)
        
    # Remove duplicates (optional, but good for speed)
    unique_candidates = []
    for c in candidates:
        c = c - c[0] # Zero mean DC
        if np.all(c < 0): c = -c
        c = np.clip(c, 0, None)
        if not any(np.allclose(c, uc) for uc in unique_candidates):
            unique_candidates.append(c)
            
    for amps in unique_candidates:
        if np.max(amps) == 0: continue
        
        # Normalize amps to unit power for consistent comparison during phase opt
        amps_norm = amps / np.sqrt(np.sum(amps**2))
        
        # Try multiple random phase initializations
        for trial in range(3):
            phases = np.random.uniform(0, 2 * np.pi, N)
            
            # Optimize phases to minimize sum(|x|^4)
            # This is a smooth objective function that penalizes peaks.
            # x = IFFT(amps * exp(i*phases))
            # Objective: J = sum(x_i^4)
            # dJ/dphi_k = sum_i 4 x_i^3 * dx_i/dphi_k
            # dx_i/dphi_k = -Im(X_k * exp(i 2 pi k i / N)) / N ... wait
            # x_i = (1/N) sum_j X_j exp(i 2 pi j i / N)
            # dx_i/dphi_j = (1/N) X_j i exp(i 2 pi j i / N)
            # X_j = A_j exp(i phi_j)
            # dx_i/dphi_j = (i/N) A_j exp(i phi_j) exp(i 2 pi j i / N)
            
            for it in range(200):
                X = amps_norm * np.exp(1j * phases)
                x = np.fft.ifft(X).real
                
                # Gradient of sum(x^4) wrt phases
                # grad_j = sum_i 4 x_i^3 * (i/N) X_j exp(i 2 pi j i / N)
                # Let's compute this efficiently.
                # x_i^3 is a time domain signal.
                # Let Y = FFT(x^3).
                # Then sum_i x_i^3 exp(i 2 pi j i / N) = Y_j (up to scaling).
                # Specifically, if x = IFFT(X), then FFT(x) = X (with N scaling in numpy? No, numpy ifft has 1/N, fft has 1).
                # x_i = 1/N sum X_k exp(i 2 pi k i / N)
                # X_k = sum x_i exp(-i 2 pi k i / N)
                # We need S_j = sum_i x_i^3 exp(i 2 pi j i / N).
                # This is -FFT(x^3)_j if we use standard definition?
                # FFT(x^3)_m = sum_i x_i^3 exp(-i 2 pi m i / N).
                # So S_j = FFT(x^3)_{-j mod N} = FFT(x^3)_{N-j}.
                
                x3 = x**3
                X3 = np.fft.fft(x3)
                # S_j = X3[(N-j) % N]
                S = X3[np.mod(-np.arange(N), N)]
                
                # grad_j = (4i/N) A_j exp(i phi_j) S_j
                # We want to minimize J, so we move phases in direction of negative gradient.
                # dJ/dphi_j = Re( (4i/N) A_j exp(i phi_j) S_j ) ? No, J is real.
                # J = sum x^4.
                # dJ/dphi_j = sum_i 4 x_i^3 (i/N) X_j exp(i 2 pi j i / N) + c.c.
                # = (2i/N) X_j sum_i x_i^3 exp(i 2 pi j i / N) + c.c.
                # = (2i/N) X_j S_j - (2i/N) X_j^* S_j^*
                # = (4i/N) Im( X_j S_j ) ? No.
                # Let Z = X_j S_j.
                # Term is (2i/N) Z - (-2i/N) Z^* = (2i/N) (Z - Z^*) = (2i/N) (2i Im(Z)) = -4/N Im(Z).
                # So grad_j = -4/N Im( X_j S_j ).
                
                Z = X * S
                grad = -4.0 / N * np.imag(Z)
                
                # Update phases
                step_size = 0.5
                phases -= step_size * grad
                phases = phases % (2 * np.pi)
            
            # Final reconstruction with original amps (not normalized)
            X = amps * np.exp(1j * phases)
            x = np.fft.ifft(X).real
            
            # Scale to fit [-0.5, 0.5]
            max_abs = np.max(np.abs(x))
            if max_abs > 1e-10:
                scale = 0.5 / max_abs
                x = x * scale
            
            # Ensure zero mean
            x = x - np.mean(x)
            
            # Clip
            x = np.clip(x, -0.5, 0.5)
            
            # Correct sum
            x = x - np.mean(x)
            x = np.clip(x, -0.5, 0.5)
            
            h = 0.5 + x
            h = np.clip(h, 0, 1)
            
            # Compute score
            H = np.fft.fft(h)
            autocorr = np.fft.ifft(np.abs(H)**2).real
            C_vals = N/2 - autocorr
            score = np.max(C_vals)
            
            if score < best_score:
                best_score = score
                best_h = h.copy()
    
    # Final correction
    sum_h = np.sum(best_h)
    if abs(sum_h - N/2) > 1e-10:
        best_h = best_h - (sum_h - N/2) / N
        best_h = np.clip(best_h, 0, 1)
    
    return best_h
# EVOLVE-BLOCK-END
