"""Rewritten program for third autocorrelation inequality optimization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.fft import dct, idct

def construct():
    """
    Returns a sequence of 400 signed step heights.
    Uses DCT-based spectral optimization to minimize the peak-to-mean ratio
    of the autoconvolution by shaping the low-frequency spectral content.
    """
    deadline = time.time() + 140.0  # Safety margin before 150s limit
    
    # Start with baseline (all ones)
    best_h = np.ones(N)
    best_value = 2.0
    
    # Objective function: compute the value we want to minimize
    def compute_value(h):
        """Compute the evaluation metric."""
        if np.sum(h) == 0 or not np.all(np.isfinite(h)):
            return np.inf
        autoconv = np.convolve(h, h)
        max_conv = np.max(autoconv)
        return abs(2 * N * max_conv / (np.sum(h) ** 2))
    
    # Try different spectral shapes in DCT domain
    # h = baseline + perturbation, where perturbation has zero mean
    # In DCT domain, we can control the spectral shape
    
    # Strategy: Use DCT to create vectors with specific low-frequency emphasis
    # that minimizes the peak of the autoconvolution
    
    # Initialize DCT coefficients
    # For a vector h, DCT gives H[k] = sum_j h[j] * cos(pi*k*(j+0.5)/N)
    # The autoconvolution in time domain corresponds to |H|^2 in frequency domain (approximately)
    
    # Try various spectral amplitude profiles
    n_trials = 500
    trial_count = 0
    
    # Precompute DCT basis for speed
    # We'll work with normalized DCT-II
    
    for trial in range(n_trials):
        if time.time() > deadline - 5.0:
            break
            
        # Create a spectral amplitude profile
        # Low frequencies get more weight to reduce high-frequency peaks in autoconvolution
        
        # Random spectral shape with emphasis on low frequencies
        np.random.seed(42 + trial)
        
        # Generate random DCT coefficients with decay
        decay_rate = 0.01 + 0.05 * np.random.random()
        dct_coeffs = np.zeros(N)
        
        # DC component (k=0) sets the mean
        dct_coeffs[0] = 1.0
        
        # AC components with random phases and decaying amplitudes
        for k in range(1, N):
            # Amplitude decays with frequency
            amp = np.exp(-decay_rate * k) * np.random.random()
            # Random sign
            sign = 1.0 if np.random.random() > 0.5 else -1.0
            dct_coeffs[k] = sign * amp
        
        # Convert from DCT domain to time domain
        h_candidate = idct(dct_coeffs, type=2, norm='ortho')
        
        # Ensure finite values
        if not np.all(np.isfinite(h_candidate)):
            continue
        
        # Normalize to have sum = N (so mean = 1, like the baseline)
        h_sum = np.sum(h_candidate)
        if abs(h_sum) < 1e-10:
            continue
        h_candidate = h_candidate * (N / h_sum)
        
        # Compute the value
        value = compute_value(h_candidate)
        
        if value < best_value:
            best_value = value
            best_h = h_candidate.copy()
        
        trial_count += 1
    
    # Additional refinement: local search around best solution
    if time.time() < deadline - 10.0:
        # Try small perturbations in DCT domain
        best_dct = dct(best_h, type=2, norm='ortho')
        
        for refine_iter in range(100):
            if time.time() > deadline - 5.0:
                break
            
            # Random perturbation in DCT domain
            np.random.seed(1000 + refine_iter)
            perturbation_scale = 0.1 / (1 + refine_iter * 0.1)
            
            # Perturb AC components (keep DC fixed to maintain mean)
            perturbed_dct = best_dct.copy()
            for k in range(1, min(50, N)):  # Only perturb low frequencies
                perturbed_dct[k] += perturbation_scale * np.random.randn()
            
            h_refined = idct(perturbed_dct, type=2, norm='ortho')
            
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
                best_dct = perturbed_dct.copy()
    
    return best_h
# EVOLVE-BLOCK-END
