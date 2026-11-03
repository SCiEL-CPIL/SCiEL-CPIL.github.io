"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy: Discrete spectral search with FFT-accelerated scoring and
Simulated Annealing / Coordinate Descent hybrid.

We parameterize the step function by its DFT coefficients.
We enforce non-negativity via hard projection (clipping) in the time domain.
We minimize the maximum value of the autoconvolution using a hybrid of
coordinate descent and simulated annealing in the frequency domain.
FFT is used to compute the autoconvolution in O(N log N) time.
"""

import numpy as np
import time

N = 600
TIME_LIMIT = 115  # safety margin before 120s kill

def compute_score_fft(f):
    """Compute the normalized maximum of the autoconvolution using FFT."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    
    # FFT size for linear convolution
    fft_size = 2 * N
    F = np.fft.fft(f, fft_size)
    G = np.fft.ifft(F * F)
    # The autoconvolution is real, so take real part
    g = np.real(G)
    return np.max(g) / (s * s)

def fast_objective(F):
    """Compute the score from DFT coefficients using FFT."""
    f = np.real(np.fft.ifft(F))
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    fft_size = 2 * N
    F_padded = np.fft.fft(f, fft_size)
    G = np.fft.ifft(F_padded * F_padded)
    g = np.real(G)
    return np.max(g) / (s * s)

def coordinate_descent_anneal(F, start_time, max_time, n_iter=500, T0=1.0, T_min=1e-4):
    """
    Coordinate descent with Simulated Annealing on DFT coefficients.
    """
    best_F = F.copy()
    best_score = fast_objective(F)
    current_F = F.copy()
    current_score = best_score
    
    half = N // 2
    step_size = 1.0
    T = T0
    
    for it in range(n_iter):
        if time.time() - start_time > max_time:
            break
        
        # Anneal temperature
        T = max(T_min, T * 0.95)
        
        improved_in_pass = False
        # Shuffle indices to avoid bias
        indices = np.random.permutation(half + 1)
        
        for k in indices:
            # Try perturbations in real and imaginary parts
            for delta_type in ['real', 'imag']:
                # Try a few random step directions
                for _ in range(2):
                    sign = np.random.choice([1, -1])
                    delta = sign * step_size * np.random.uniform(0.5, 1.5)
                    
                    old_k = current_F[k]
                    old_Nk = current_F[N - k] if k != 0 and k != half else None
                    
                    # Apply perturbation
                    if delta_type == 'real':
                        current_F[k] += delta
                        if k != 0 and k != half:
                            current_F[N - k] += delta
                    else:
                        current_F[k] += delta * 1j
                        if k != 0 and k != half:
                            current_F[N - k] -= delta * 1j
                    
                    new_score = fast_objective(current_F)
                    
                    # Acceptance criterion: Metropolis
                    if new_score < current_score:
                        # Always accept improvement
                        current_score = new_score
                        if new_score < best_score:
                            best_score = new_score
                            best_F = current_F.copy()
                            improved_in_pass = True
                        # Increase step size slightly on success
                        step_size = min(step_size * 1.1, 10.0)
                        break # Move to next k
                    else:
                        # Accept with probability exp(-deltaE / T)
                        delta_E = new_score - current_score
                        if delta_E < 0.01: # Only consider small worsening
                            prob = np.exp(-delta_E / T)
                            if np.random.random() < prob:
                                current_score = new_score
                                # Do not update best
                                break
                        # Revert
                        current_F[k] = old_k
                        if old_Nk is not None:
                            current_F[N - k] = old_Nk
                
                if improved_in_pass:
                    break
        
        if not improved_in_pass:
            step_size *= 0.8
            if step_size < 1e-6:
                break
    
    return best_F, best_score

def construct():
    start_time = time.time()
    
    # Initial best: constant function
    best_f = np.ones(N)
    best_score = compute_score_fft(best_f)
    
    # Strategy: Multiple restarts with different spectral shapes
    # 1. Low-pass (Box in time)
    # 2. Band-pass
    # 3. Random phases (Gaussian in time)
    
    n_restarts = 10
    half = N // 2
    
    for r in range(n_restarts):
        if time.time() - start_time > TIME_LIMIT - 10:
            break
        
        F_init = np.zeros(N, dtype=np.complex128)
        
        if r == 0:
            # Start with a Gaussian in time
            t = np.linspace(-1, 1, N)
            f_init = np.exp(-2 * t * t)
            F_init = np.fft.fft(f_init)
            
        elif r == 1:
            # Start with a triangle in time
            t = np.linspace(-1, 1, N)
            f_init = np.maximum(0, 1 - np.abs(t))
            F_init = np.fft.fft(f_init)
            
        else:
            # Random low-pass filter
            cutoff = np.random.randint(20, N // 4)
            F_init[:cutoff] = np.exp(1j * np.random.uniform(0, 2 * np.pi, cutoff))
            F_init[0] = np.abs(F_init[0]) # Ensure DC is real positive
            for k in range(1, cutoff):
                F_init[N - k] = np.conj(F_init[k])
        
        # Coordinate descent with annealing
        F_opt, score_opt = coordinate_descent_anneal(F_init, start_time, TIME_LIMIT - 5, n_iter=200)
        
        # Convert back to time domain
        f_opt = np.real(np.fft.ifft(F_opt))
        f_opt = np.maximum(f_opt, 0.0)
        
        # Compute actual score
        actual_score = compute_score_fft(f_opt)
        
        if actual_score < best_score:
            best_score = actual_score
            best_f = f_opt
    
    # Ensure nonnegative
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score_fft(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
