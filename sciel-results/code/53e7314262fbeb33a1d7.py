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
    
    # Use FFT for fast linear convolution
    # Pad to avoid circular convolution artifacts
    n_fft = 2 * N
    F = np.fft.fft(f, n=n_fft)
    conv_fft = np.fft.ifft(F * F)
    conv = np.real(conv_fft[:2*N-1])  # Linear convolution result
    
    return np.max(conv) / (s * s)

def objective_from_F(F):
    """Compute the exact score from DFT coefficients using FFT-based autoconvolution."""
    f = np.real(np.fft.ifft(F))
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    
    # Use FFT for fast linear convolution
    n_fft = 2 * N
    F_time = np.fft.fft(f, n=n_fft)
    conv_fft = np.fft.ifft(F_time * F_time)
    conv = np.real(conv_fft[:2*N-1])
    
    return np.max(conv) / (s * s)

def coordinate_descent(F, start_time, max_time, n_iter=200):
    """
    Coordinate descent on DFT coefficients to minimize the peak of autoconvolution.
    Uses adaptive step sizes and tries both real and imaginary perturbations.
    """
    best_F = F.copy()
    best_score = objective_from_F(F)
    current_F = F.copy()
    
    half = N // 2
    step_size = 0.1
    no_improve_count = 0
    max_no_improve = 50
    
    for it in range(n_iter):
        if time.time() - start_time > max_time:
            break
        
        improved_in_pass = False
        
        # Iterate over frequency bins (only need to handle k and N-k together)
        for k in range(half + 1):
            if time.time() - start_time > max_time:
                break
            
            # For each frequency bin, try small perturbations
            # We'll try several step sizes and directions
            for trial_step in [step_size, step_size * 0.5, step_size * 2.0]:
                if trial_step < 1e-10:
                    continue
                    
                # Try positive real perturbation
                old_k = current_F[k]
                old_Nk = current_F[N - k] if k != 0 and k != half else None
                
                current_F[k] += trial_step
                if k != 0 and k != half:
                    current_F[N - k] += trial_step
                
                score_trial = objective_from_F(current_F)
                
                if score_trial < best_score:
                    best_score = score_trial
                    best_F = current_F.copy()
                    improved_in_pass = True
                    no_improve_count = 0
                    break
                else:
                    # Revert
                    current_F[k] = old_k
                    if old_Nk is not None:
                        current_F[N - k] = old_Nk
                
                # Try negative real perturbation
                current_F[k] -= trial_step
                if k != 0 and k != half:
                    current_F[N - k] -= trial_step
                
                score_trial = objective_from_F(current_F)
                
                if score_trial < best_score:
                    best_score = score_trial
                    best_F = current_F.copy()
                    improved_in_pass = True
                    no_improve_count = 0
                    break
                else:
                    # Revert
                    current_F[k] = old_k
                    if old_Nk is not None:
                        current_F[N - k] = old_Nk
                
                # Try positive imaginary perturbation
                current_F[k] += trial_step * 1j
                if k != 0 and k != half:
                    current_F[N - k] -= trial_step * 1j
                
                score_trial = objective_from_F(current_F)
                
                if score_trial < best_score:
                    best_score = score_trial
                    best_F = current_F.copy()
                    improved_in_pass = True
                    no_improve_count = 0
                    break
                else:
                    # Revert
                    current_F[k] = old_k
                    if old_Nk is not None:
                        current_F[N - k] = old_Nk
                
                # Try negative imaginary perturbation
                current_F[k] -= trial_step * 1j
                if k != 0 and k != half:
                    current_F[N - k] += trial_step * 1j
                
                score_trial = objective_from_F(current_F)
                
                if score_trial < best_score:
                    best_score = score_trial
                    best_F = current_F.copy()
                    improved_in_pass = True
                    no_improve_count = 0
                    break
                else:
                    # Revert
                    current_F[k] = old_k
                    if old_Nk is not None:
                        current_F[N - k] = old_Nk
        
        if not improved_in_pass:
            no_improve_count += 1
            step_size *= 0.7
            if no_improve_count > max_no_improve or step_size < 1e-8:
                break
        else:
            step_size *= 1.1
    
    return best_F, best_score

def time_domain_refinement(f, start_time, max_time, n_iter=50):
    """
    Local refinement in time domain using coordinate descent.
    Tries small adjustments to individual time-domain values.
    """
    best_f = f.copy()
    best_score = compute_score(f)
    current_f = f.copy()
    
    step_size = 0.01
    no_improve_count = 0
    
    for it in range(n_iter):
        if time.time() - start_time > max_time:
            break
        
        improved_in_pass = False
        
        # Randomly shuffle indices to avoid bias
        indices = np.random.permutation(N)
        
        for i in indices:
            if time.time() - start_time > max_time:
                break
            
            old_val = current_f[i]
            
            # Try increasing
            current_f[i] = max(0.0, old_val + step_size)
            score_trial = compute_score(current_f)
            
            if score_trial < best_score:
                best_score = score_trial
                best_f = current_f.copy()
                improved_in_pass = True
                continue
            else:
                current_f[i] = old_val
            
            # Try decreasing
            current_f[i] = max(0.0, old_val - step_size)
            score_trial = compute_score(current_f)
            
            if score_trial < best_score:
                best_score = score_trial
                best_f = current_f.copy()
                improved_in_pass = True
                continue
            else:
                current_f[i] = old_val
        
        if not improved_in_pass:
            no_improve_count += 1
            step_size *= 0.8
            if no_improve_count > 20 or step_size < 1e-6:
                break
        else:
            step_size *= 1.05
    
    return best_f, best_score

def construct():
    start_time = time.time()
    
    # Initial best: constant function
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    np.random.seed(42)
    half = N // 2
    
    # Multiple restart strategies
    n_restarts = 15
    
    for r in range(n_restarts):
        if time.time() - start_time > TIME_LIMIT - 15:
            break
        
        # Different initialization strategies
        if r == 0:
            # Box filter with uniform magnitude
            cutoff = N // 4
            F_init = np.zeros(N, dtype=np.complex128)
            F_init[0] = N / 4.0
            for k in range(1, cutoff):
                F_init[k] = N / 4.0
                F_init[N - k] = N / 4.0
        elif r == 1:
            # Tapered box filter
            cutoff = N // 3
            F_init = np.zeros(N, dtype=np.complex128)
            F_init[0] = 1.0
            for k in range(1, cutoff):
                # Cosine taper
                mag = (N / cutoff) * 0.5 * (1 + np.cos(np.pi * k / cutoff))
                F_init[k] = mag
                F_init[N - k] = mag
        elif r == 2:
            # Random phases with decaying magnitude
            cutoff = N // 2
            F_init = np.zeros(N, dtype=np.complex128)
            F_init[0] = 1.0
            for k in range(1, cutoff):
                mag = 1.0 / (1.0 + 0.1 * k)
                phase = np.random.uniform(0, 2 * np.pi)
                F_init[k] = mag * np.exp(1j * phase)
                F_init[N - k] = np.conj(F_init[k])
        elif r == 3:
            # Gaussian-like spectrum
            F_init = np.zeros(N, dtype=np.complex128)
            for k in range(N):
                kk = min(k, N - k)
                mag = np.exp(-0.001 * kk * kk)
                F_init[k] = mag
        else:
            # Random box with random cutoff and random phases
            cutoff = np.random.randint(20, N // 2)
            F_init = np.zeros(N, dtype=np.complex128)
            F_init[0] = 1.0
            for k in range(1, cutoff):
                mag = np.random.uniform(0.5, 1.5)
                phase = np.random.uniform(0, 2 * np.pi)
                F_init[k] = mag * np.exp(1j * phase)
                F_init[N - k] = np.conj(F_init[k])
        
        # Coordinate descent on frequency domain
        F_opt, score_opt = coordinate_descent(F_init, start_time, TIME_LIMIT - 10, n_iter=100)
        
        # Convert back to time domain
        f_opt = np.real(np.fft.ifft(F_opt))
        f_opt = np.maximum(f_opt, 0.0)
        
        # Normalize to have reasonable scale
        s = np.sum(f_opt)
        if s > 1e-10:
            f_opt = f_opt * (N / s)  # Normalize so mean is 1
        
        # Compute actual score
        actual_score = compute_score(f_opt)
        
        # Time domain refinement
        if actual_score < best_score:
            f_refined, score_refined = time_domain_refinement(f_opt, start_time, TIME_LIMIT - 5, n_iter=30)
            if score_refined < actual_score:
                actual_score = score_refined
                f_opt = f_refined
        
        if actual_score < best_score:
            best_score = actual_score
            best_f = f_opt
    
    # Final time domain refinement on best solution
    if time.time() - start_time < TIME_LIMIT - 5:
        f_final, score_final = time_domain_refinement(best_f, start_time, TIME_LIMIT, n_iter=50)
        if score_final < best_score:
            best_f = f_final
    
    # Ensure nonnegative
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
