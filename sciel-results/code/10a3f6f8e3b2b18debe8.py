"""Rewritten program for third autocorrelation inequality optimization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.fft import fft, ifft

def construct():
    """
    Returns a sequence of 400 signed step heights.
    Uses a structural approach: model h as a low-pass filtered signal in the
    circular frequency domain to minimize the peak of the circular autocorrelation.
    The key insight is that the circular autocorrelation is the IFFT of |H|^2,
    where H is the FFT of h. We shape |H|^2 to be as flat as possible (white noise
    spectrum) to minimize the peak of the autocorrelation.
    """
    deadline = time.time() + 145.0  # Safety margin before 150s limit

    # Baseline: all ones
    best_h = np.ones(N)
    best_value = 2.0

    def compute_value(h):
        """Compute the evaluation metric."""
        h_sum = np.sum(h)
        if h_sum == 0 or not np.all(np.isfinite(h)):
            return np.inf
        # Use circular convolution for speed (via FFT)
        # But the evaluator uses np.convolve which is full linear convolution
        # We need to be careful: np.convolve(h,h) has length 2N-1
        # Let's compute it directly but efficiently
        autoconv = np.convolve(h, h)
        max_conv = np.max(autoconv)
        return abs(2 * N * max_conv / (h_sum ** 2))

    # Strategy: We want to minimize max_k (h*h)_k / (sum h)^2
    # Note that (h*h)_k = sum_j h[j] h[k-j] for the full convolution
    # The maximum is at k = N-1 (the center) for symmetric-like signals
    # 
    # Key structural idea: If we make h have a very flat power spectrum,
    # the autocorrelation will be close to a delta function, minimizing the peak.
    # 
    # In the circular domain: circular autocorrelation = IFFT(|FFT(h)|^2)
    # For linear convolution, we can zero-pad to avoid wrap-around.
    #
    # Let's work in the frequency domain. Let H = FFT(h, n=2N) (zero-padded)
    # Then the linear autocorrelation is IFFT(|H|^2, n=2N)[N-1:N-1+N] (approximately)
    # Actually, linear autocorrelation r[k] = sum_j h[j] h[j+k] for k in [-(N-1), N-1]
    # This equals IFFT(FFT(h, 2N) * conj(FFT(h, 2N)), 2N)[N-1 : N-1+N]
    #
    # So we want to choose H (with H[0] fixed for the mean) to minimize
    # max over the central N entries of IFFT(|H|^2, 2N).
    #
    # A flat |H|^2 (except DC) would give a delta-like autocorrelation.
    # But we need h to be real, so H must satisfy conjugate symmetry.
    #
    # Let's try: set |H[k]|^2 = c for k != 0, and H[0] = S (sum of h)
    # Then |H|^2 has a big DC spike and flat elsewhere.
    # IFFT of this gives: (S^2/N + c*(N-1)/N) at lag 0, and -c/N at other lags.
    # Wait, that's not quite right. Let me think more carefully.
    #
    # Actually, let's just try various spectral shapes and see what works.
    # We'll parameterize the amplitude spectrum and optimize.

    # Use zero-padded FFT for linear convolution
    n_fft = 2 * N  # 800

    # Precompute: for a given amplitude spectrum A[k] and phase phi[k],
    # H[k] = A[k] * exp(i*phi[k]), with conjugate symmetry for real h.
    # We want to minimize the peak of the linear autocorrelation.

    # Let's try a simple approach: 
    # 1. Start with a flat spectrum (white noise)
    # 2. Try to reduce the peak by adjusting the spectrum

    # White noise in time domain has flat power spectrum
    # Let's generate random white noise vectors and see the value
    np.random.seed(42)
    
    # Try several random initializations
    for trial in range(200):
        if time.time() > deadline - 5.0:
            break

        # Random vector with zero mean (AC only)
        h_ac = np.random.randn(N)
        h_ac -= np.mean(h_ac)  # Remove DC
        
        # Add DC component
        dc = 1.0
        h_candidate = h_ac + dc
        
        # Ensure sum is nonzero
        h_sum = np.sum(h_candidate)
        if abs(h_sum) < 1e-10:
            continue
        
        # Normalize so that sum = N (mean = 1)
        h_candidate = h_candidate * (N / h_sum)
        
        value = compute_value(h_candidate)
        
        if value < best_value:
            best_value = value
            best_h = h_candidate.copy()

    # Now try a more structured approach: 
    # Use the fact that we can control the spectrum directly.
    # Let H be the FFT of h (zero-padded to 2N).
    # For real h, H[k] = conj(H[2N-k]).
    # We want to minimize max of IFFT(|H|^2, 2N) in the central region.
    
    # Let's parameterize by the log-amplitude spectrum.
    # Start from the best solution found so far.
    
    best_h_fft = fft(best_h, n=n_fft)
    
    # Local search in the spectral domain
    for refine_iter in range(300):
        if time.time() > deadline - 5.0:
            break
        
        np.random.seed(10000 + refine_iter)
        
        # Perturb the amplitude spectrum slightly
        # Keep DC component fixed
        perturbation_scale = 0.05 / (1 + refine_iter * 0.05)
        
        # Work with the amplitude spectrum
        amp = np.abs(best_h_fft).copy()
        phase = np.angle(best_h_fft).copy()
        
        # Perturb amplitudes for non-DC frequencies
        # Only perturb a random subset to keep it local
        indices_to_perturb = np.random.choice(range(1, n_fft // 2), size=min(50, n_fft // 2 - 1), replace=False)
        for idx in indices_to_perturb:
            amp[idx] *= (1 + perturbation_scale * np.random.randn())
            amp[n_fft - idx] = amp[idx]  # Maintain conjugate symmetry
        
        # Reconstruct H
        H_new = amp * np.exp(1j * phase)
        # Enforce conjugate symmetry
        H_new[n_fft // 2:] = np.conj(H_new[:n_fft // 2][::-1])
        # DC is real
        H_new[0] = np.abs(H_new[0])
        
        h_refined = ifft(H_new, n=n_fft)[:N].real
        
        if not np.all(np.isfinite(h_refined)):
            continue
        
        h_sum = np.sum(h_refined)
        if abs(h_sum) < 1e-10:
            continue
        
        h_refined = h_refined * (N / h_sum)
        
        value = compute_value(h_refined)
        
        if value < best_value:
            best_value = value
            best_h = h_refined.copy()
            best_h_fft = fft(best_h, n=n_fft)

    # Try a completely different structural approach:
    # Use a low-pass filter to smooth the signal, which reduces high-frequency
    # content and thus reduces the peak of the autocorrelation.
    
    if time.time() < deadline - 10.0:
        # Generate a smooth signal using a combination of low-frequency sinusoids
        for trial in range(100):
            if time.time() > deadline - 5.0:
                break
            
            np.random.seed(20000 + trial)
            
            # Choose a few random low frequencies
            n_components = np.random.randint(2, 6)
            freqs = np.random.randint(1, 10, size=n_components)
            amps = np.random.uniform(0.1, 1.0, size=n_components)
            phases = np.random.uniform(0, 2 * np.pi, size=n_components)
            
            t = np.arange(N)
            h_candidate = np.ones(N)  # DC component
            for f, a, p in zip(freqs, amps, phases):
                h_candidate += a * np.cos(2 * np.pi * f * t / N + p)
            
            h_sum = np.sum(h_candidate)
            if abs(h_sum) < 1e-10:
                continue
            
            h_candidate = h_candidate * (N / h_sum)
            
            value = compute_value(h_candidate)
            
            if value < best_value:
                best_value = value
                best_h = h_candidate.copy()

    return best_h
# EVOLVE-BLOCK-END
