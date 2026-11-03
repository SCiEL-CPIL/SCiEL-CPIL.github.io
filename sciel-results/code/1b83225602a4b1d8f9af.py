"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy: Discrete spectral search.
Parameterize the step function by its DFT coefficients.
Enforce non-negativity via a soft-plus projection.
Directly minimize the maximum value of the inverse DFT of the squared magnitude spectrum
using coordinate descent on the frequency domain.
This decouples the peak constraint from the time-domain shape search.

Key insight: The autoconvolution in the time domain corresponds to the squared
magnitude of the DFT in the frequency domain (up to scaling). By working in the
frequency domain, we can directly control the "energy distribution" of the
autoconvolution. The maximum of the autoconvolution is related to the peak of
the inverse DFT of |F|^2. We use coordinate descent on the frequency coefficients
to flatten this peak, while projecting back to the time domain to enforce
non-negativity.

Improved approach:
1. Use FFT-based autoconvolution for O(N log N) score computation instead of O(N^2)
2. Use a smarter initialization based on known optimal shapes (e.g., triangular)
3. Use a more efficient coordinate descent with adaptive step sizes
4. Add time-domain refinement steps after frequency-domain optimization
5. Use multiple restarts with different spectral masks
"""

import numpy as np
import time

N = 600
TIME_LIMIT = 115  # safety margin before 120s kill

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution using FFT."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    
    # Use FFT for linear convolution
    n_conv = 2 * N - 1
    # Next power of 2 for efficient FFT
    n_fft = 1 << (n_conv - 1).bit_length()
    
    F = np.fft.rfft(f, n=n_fft)
    conv = np.fft.irfft(F * F, n=n_fft)[:n_conv]
    
    return np.max(conv) / (s * s)

def compute_score_from_F(F):
    """Compute score directly from DFT coefficients."""
    f = np.real(np.fft.ifft(F))
    f = np.maximum(f, 0.0)
    return compute_score(f)

def make_initial_F(mask_type='box', cutoff=None, random_phases=False, seed=None):
    """Create initial DFT coefficients with a given spectral mask."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    
    F = np.zeros(N, dtype=np.complex128)
    half = N // 2
    
    if cutoff is None:
        cutoff = N // 4
    
    if mask_type == 'box':
        # Box filter: uniform magnitude in [0, cutoff]
        for k in range(0, cutoff + 1):
            if k == 0:
                F[k] = 1.0
            elif k == half:
                F[k] = 1.0
            else:
                if random_phases:
                    phase = rng.uniform(0, 2 * np.pi)
                else:
                    phase = 0.0
                F[k] = np.exp(1j * phase)
                F[N - k] = np.conj(F[k])
    
    elif mask_type == 'triangle':
        # Triangular spectrum: linearly decreasing magnitude
        for k in range(0, cutoff + 1):
            mag = 1.0 - k / cutoff
            if k == 0:
                F[k] = mag
            elif k == half:
                F[k] = mag
            else:
                if random_phases:
                    phase = rng.uniform(0, 2 * np.pi)
                else:
                    phase = 0.0
                F[k] = mag * np.exp(1j * phase)
                F[N - k] = np.conj(F[k])
    
    elif mask_type == 'gaussian':
        # Gaussian spectrum
        sigma = cutoff / 2.0
        for k in range(0, cutoff + 1):
            mag = np.exp(-k**2 / (2 * sigma**2))
            if k == 0:
                F[k] = mag
            elif k == half:
                F[k] = mag
            else:
                if random_phases:
                    phase = rng.uniform(0, 2 * np.pi)
                else:
                    phase = 0.0
                F[k] = mag * np.exp(1j * phase)
                F[N - k] = np.conj(F[k])
    
    # Ensure conjugate symmetry
    for k in range(1, half + 1):
        F[N - k] = np.conj(F[k])
    
    return F

def coordinate_descent(F, start_time, max_time, n_iter=10000):
    """Coordinate descent on DFT coefficients to minimize the peak of autoconvolution."""
    best_F = F.copy()
    best_score = compute_score_from_F(F)
    current_F = F.copy()
    
    half = N // 2
    step_size = 0.1
    
    # Precompute which indices are paired
    paired = {}
    for k in range(half + 1):
        if k == 0 or k == half:
            paired[k] = (k, None)
        else:
            paired[k] = (k, N - k)
    
    # Keep track of recent steps for adaptive sizing
    step_history = {}
    
    for it in range(n_iter):
        if time.time() - start_time > max_time:
            break
        
        improved_in_pass = False
        no_improve_count = 0
        
        # Iterate over frequency bins
        for k in range(half + 1):
            if time.time() - start_time > max_time:
                break
            
            idx_k, idx_Nk = paired[k]
            
            # Try perturbations in real and imaginary parts
            for delta_type in ['real', 'imag']:
                for sign in [1, -1]:
                    # Save old values
                    old_k = current_F[idx_k]
                    old_Nk = current_F[idx_Nk] if idx_Nk is not None else None
                    
                    # Apply perturbation
                    if delta_type == 'real':
                        current_F[idx_k] += sign * step_size
                        if idx_Nk is not None:
                            current_F[idx_Nk] += sign * step_size
                    else:
                        current_F[idx_k] += sign * step_size * 1j
                        if idx_Nk is not None:
                            current_F[idx_Nk] -= sign * step_size * 1j
                    
                    score_trial = compute_score_from_F(current_F)
                    
                    if score_trial < best_score:
                        best_score = score_trial
                        best_F = current_F.copy()
                        improved_in_pass = True
                        no_improve_count = 0
                        # Adaptive step: increase if we improved
                        step_key = (k, delta_type)
                        if step_key not in step_history:
                            step_history[step_key] = 0
                        step_history[step_key] += 1
                        break # Move to next delta_type
                    else:
                        # Revert
                        current_F[idx_k] = old_k
                        if old_Nk is not None:
                            current_F[idx_Nk] = old_Nk
                        no_improve_count += 1
                
                if improved_in_pass:
                    break
        
        # Adaptive step size
        if improved_in_pass:
            step_size = min(step_size * 1.1, 1.0)
        else:
            step_size *= 0.7
            if step_size < 1e-10:
                break
    
    return best_F, best_score

def time_domain_refine(f, start_time, max_time, n_iter=500):
    """Refine the time-domain signal to minimize autoconvolution peak."""
    best_f = f.copy()
    best_score = compute_score(f)
    current_f = f.copy()
    
    step_size = 0.01
    n_improvements = 0
    
    for it in range(n_iter):
        if time.time() - start_time > max_time:
            break
        
        improved = False
        # Try random coordinate updates
        for _ in range(10):
            if time.time() - start_time > max_time:
                break
            
            idx = np.random.randint(0, N)
            for sign in [1, -1]:
                old_val = current_f[idx]
                current_f[idx] = max(0.0, old_val + sign * step_size)
                
                score_trial = compute_score(current_f)
                
                if score_trial < best_score:
                    best_score = score_trial
                    best_f = current_f.copy()
                    improved = True
                    n_improvements += 1
                    break
                else:
                    current_f[idx] = old_val
        
        if improved:
            step_size = min(step_size * 1.05, 0.1)
        else:
            step_size *= 0.8
            if step_size < 1e-6:
                break
    
    return best_f, best_score

def construct():
    start_time = time.time()
    
    # Initial best: constant function
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    # Try several initialization strategies
    init_configs = [
        ('box', N//4, False, 0),
        ('box', N//4, True, 1),
        ('box', N//3, True, 2),
        ('box', N//6, True, 3),
        ('triangle', N//4, False, 4),
        ('triangle', N//4, True, 5),
        ('triangle', N//3, True, 6),
        ('gaussian', N//4, False, 7),
        ('gaussian', N//4, True, 8),
        ('gaussian', N//3, True, 9),
    ]
    
    for config in init_configs:
        if time.time() - start_time > TIME_LIMIT - 15:
            break
        
        mask_type, cutoff, random_phases, seed = config
        
        # Create initial DFT
        F_init = make_initial_F(mask_type, cutoff, random_phases, seed)
        
        # Coordinate descent on frequency domain
        F_opt, score_opt = coordinate_descent(F_init, start_time, TIME_LIMIT - 10, n_iter=5000)
        
        # Convert back to time domain
        f_opt = np.real(np.fft.ifft(F_opt))
        f_opt = np.maximum(f_opt, 0.0)
        
        # Compute actual score
        actual_score = compute_score(f_opt)
        
        if actual_score < best_score:
            best_score = actual_score
            best_f = f_opt.copy()
        
        # Time-domain refinement
        if time.time() - start_time < TIME_LIMIT - 5:
            f_refined, score_refined = time_domain_refine(
                best_f, start_time, TIME_LIMIT - 3, n_iter=200
            )
            if score_refined < best_score:
                best_score = score_refined
                best_f = f_refined
    
    # Ensure nonnegative
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
