"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 600

# EVOLVE-BLOCK-START

def _autoconv_max_normalized(heights):
    """
    Compute the normalized maximum of the autoconvolution of the step function.
    """
    h = np.array(heights, dtype=np.float64)
    n = len(h)
    padded = np.zeros(2 * n, dtype=np.float64)
    padded[:n] = h
    F = np.fft.fft(padded)
    G = F * np.conj(F)
    conv = np.fft.ifft(G).real
    max_conv = np.max(conv[:2*n-1])
    total = np.sum(h)
    if total == 0:
        return 0.0
    dx = 1.0 / (2 * n)
    normalized_max = max_conv / (total * total * dx)
    return normalized_max

def _phase_objective(phases):
    """
    Objective: given phases, construct heights from a flat-spectrum (white noise-like) 
    function and return the normalized max autoconvolution.
    
    We use a fixed amplitude spectrum (all ones for nonzero frequencies) and vary phases.
    This ensures a flat power spectrum, which should minimize peak autoconvolution.
    """
    n = N
    # Create frequency-domain representation with flat amplitude and given phases
    # Use only the first n//2 + 1 frequencies (real signal)
    half_n = n // 2 + 1
    # Amplitudes: 1 for all nonzero freqs, 0 for DC (to center the function)
    amplitudes = np.ones(half_n, dtype=np.float64)
    amplitudes[0] = 0.0  # Remove DC component
    
    # Phases: first half_n phases from the parameter vector
    # For a real signal, we need conjugate symmetry
    phase = np.zeros(half_n, dtype=np.float64)
    phase[:len(phases)] = phases[:len(phases)]
    
    # Construct FFT
    F = np.zeros(half_n, dtype=np.complex128)
    F = amplitudes * np.exp(1j * phase)
    
    # Build full FFT with conjugate symmetry
    full_F = np.zeros(n, dtype=np.complex128)
    full_F[:half_n] = F
    # For even n: full_F[n-k] = conj(full_F[k]) for k=1..n/2-1
    # For odd n: full_F[n-k] = conj(full_F[k]) for k=1..(n-1)/2
    if n % 2 == 0:
        for k in range(1, n // 2):
            full_F[n - k] = np.conj(full_F[k])
    else:
        for k in range(1, (n - 1) // 2 + 1):
            full_F[n - k] = np.conj(full_F[k])
    
    # Inverse FFT to get heights
    h = np.fft.ifft(full_F).real
    # Ensure non-negativity
    h = np.maximum(h, 0.0)
    
    return _autoconv_max_normalized(h)

def _direct_objective(heights):
    """Direct optimization on heights with non-negativity constraint."""
    h = np.maximum(heights, 0.0)
    return _autoconv_max_normalized(h)

def construct():
    """
    Construct a nonnegative step function with 600 equal intervals on [-1/4, 1/4]
    to minimize the normalized maximum of its autoconvolution.
    """
    start_time = time.time()
    deadline = 145.0  # 5 seconds safety margin
    
    # Initial best: uniform
    best_heights = [1.0] * N
    best_score = _autoconv_max_normalized(best_heights)
    
    t = np.arange(N, dtype=np.float64)
    
    # Try raised cosine
    h_cos = 0.5 * (1.0 + np.cos(np.pi * t / (N - 1)))
    score_cos = _autoconv_max_normalized(h_cos)
    if score_cos < best_score:
        best_score = score_cos
        best_heights = list(h_cos)
    
    # Try Gaussian
    center = N / 2.0
    sigma = N / 4.0
    h_gauss = np.exp(-0.5 * ((t - center) / sigma) ** 2)
    score_gauss = _autoconv_max_normalized(h_gauss)
    if score_gauss < best_score:
        best_score = score_gauss
        best_heights = list(h_gauss)
    
    # Try random phases approach: start with random phases and optimize
    # Number of free phases: we'll use min(N//2, 300) phases
    n_phases = min(N // 2, 300)
    
    # Try several random initializations
    rng = np.random.RandomState(42)
    
    for trial in range(5):
        if time.time() > deadline - 20.0:
            break
        
        # Random initial phases
        init_phases = rng.uniform(0, 2 * np.pi, size=n_phases)
        
        # Optimize phases using L-BFGS-B
        try:
            result = minimize(
                _phase_objective,
                init_phases,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            
            if result.fun < best_score:
                # Reconstruct heights
                n = N
                half_n = n // 2 + 1
                amplitudes = np.ones(half_n, dtype=np.float64)
                amplitudes[0] = 0.0
                phase = np.zeros(half_n, dtype=np.float64)
                phase[:len(result.x)] = result.x[:len(phase)]
                
                F = amplitudes * np.exp(1j * phase)
                full_F = np.zeros(n, dtype=np.complex128)
                full_F[:half_n] = F
                if n % 2 == 0:
                    for k in range(1, n // 2):
                        full_F[n - k] = np.conj(full_F[k])
                else:
                    for k in range(1, (n - 1) // 2 + 1):
                        full_F[n - k] = np.conj(full_F[k])
                
                h = np.fft.ifft(full_F).real
                h = np.maximum(h, 0.0)
                score = _autoconv_max_normalized(h)
                if score < best_score:
                    best_score = score
                    best_heights = list(h)
        except Exception:
            pass
    
    # Try direct L-BFGS-B on heights with smoothing
    if time.time() < deadline - 15.0:
        # Start from current best and optimize directly
        h0 = np.array(best_heights, dtype=np.float64)
        # Add a small regularization to encourage smoothness
        try:
            def _smooth_objective(h):
                h_pos = np.maximum(h, 0.0)
                score = _autoconv_max_normalized(h_pos)
                # Smoothness penalty: sum of squared differences
                diff = np.diff(h_pos)
                penalty = 0.01 * np.sum(diff**2)
                return score + penalty
            
            result = minimize(
                _smooth_objective,
                h0,
                method='L-BFGS-B',
                bounds=[(0.0, None)] * N,
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-4}
            )
            
            if result.fun < best_score + 0.1:  # Allow some penalty
                h_opt = np.maximum(result.x, 0.0)
                score = _autoconv_max_normalized(h_opt)
                if score < best_score:
                    best_score = score
                    best_heights = list(h_opt)
        except Exception:
            pass
    
    return best_heights

# EVOLVE-BLOCK-END
