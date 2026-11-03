"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy: Iterative Spectral Flattening with Non-Negativity Projection.
We iterate between the time and frequency domains.
1. In the frequency domain, we flatten the magnitude spectrum to reduce the
   peak of the autoconvolution (which corresponds to the inverse DFT of |F|^2).
2. In the time domain, we project the signal to be non-negative.
3. We normalize the signal to maintain a fixed sum.

This approach is computationally efficient (O(N log N) per iteration) and
effectively searches the spectral space for a signal with a "flat" power
spectrum, which minimizes the autocorrelation peak for a given energy.
"""

import numpy as np
import time

N = 600
TIME_LIMIT = 115  # safety margin before 120s kill

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    # Use FFT for faster convolution calculation
    # Linear convolution of two length N signals requires FFT size >= 2N-1
    # We use 2*N for simplicity and safety
    f_fft = np.fft.fft(f, n=2*N)
    conv_fft = f_fft * f_fft
    conv = np.real(np.fft.ifft(conv_fft))
    # The linear convolution is contained in conv[0:2*N-1]
    max_val = np.max(conv[:2*N-1])
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return max_val / (s * s)

def spectral_flatten(f, n_iters=500, alpha=0.5):
    """
    Iteratively flatten the magnitude spectrum of f while enforcing non-negativity.
    
    Args:
        f: Initial time-domain signal (non-negative).
        n_iters: Number of iterations.
        alpha: Blending factor for magnitude flattening (0=keep old, 1=fully flat).
    """
    f = f.copy()
    # Normalize to unit sum
    s = np.sum(f)
    if s > 1e-15:
        f = f / s
    
    for _ in range(n_iters):
        # 1. Go to frequency domain
        F = np.fft.fft(f)
        
        # 2. Flatten magnitude
        # We want |F|^2 to be as flat as possible.
        # Target magnitude is the geometric mean or arithmetic mean of current magnitudes?
        # Arithmetic mean of magnitudes is a good heuristic.
        mag = np.abs(F)
        target_mag = np.mean(mag)
        
        # Blend to target to avoid instability
        new_mag = (1 - alpha) * mag + alpha * target_mag
        
        # 3. Reconstruct F with new magnitude and old phase
        phase = np.angle(F)
        F_new = new_mag * np.exp(1j * phase)
        
        # 4. Back to time domain
        f_new = np.real(np.fft.ifft(F_new))
        
        # 5. Enforce non-negativity
        f_new = np.maximum(f_new, 0.0)
        
        # 6. Normalize
        s_new = np.sum(f_new)
        if s_new < 1e-15:
            # If signal dies out, restart with random noise
            f_new = np.random.rand(N)
            f_new = np.maximum(f_new, 0.0)
            s_new = np.sum(f_new)
            f_new = f_new / s_new
        else:
            f_new = f_new / s_new
        
        f = f_new
        
    return f

def local_refinement(f, start_time, max_time):
    """
    Perform local coordinate descent on the time domain to fine-tune the solution.
    This corrects for any suboptimality introduced by the spectral flattening heuristic.
    """
    f = f.copy()
    best_score = compute_score(f)
    step_size = 1e-4
    
    # Use a simple hill-climbing approach on random coordinates
    n_coords = N
    no_improve = 0
    
    while time.time() - start_time < max_time:
        improved = False
        # Shuffle indices to avoid bias
        indices = np.random.permutation(n_coords)
        
        for idx in indices:
            if time.time() - start_time > max_time:
                break
            
            old_val = f[idx]
            
            # Try increasing
            f[idx] = old_val + step_size
            score_plus = compute_score(f)
            if score_plus < best_score:
                best_score = score_plus
                improved = True
                no_improve = 0
                # Keep change
            else:
                # Try decreasing
                f[idx] = old_val - step_size
                score_minus = compute_score(f)
                if score_minus < best_score and f[idx] >= 0:
                    best_score = score_minus
                    improved = True
                    no_improve = 0
                    # Keep change
                else:
                    # Revert
                    f[idx] = old_val
            
            if improved:
                break # Restart pass with new best
        
        if not improved:
            no_improve += 1
            step_size *= 0.9
            if step_size < 1e-8 or no_improve > 20:
                break
    
    return f, best_score

def construct():
    start_time = time.time()
    
    # Initial best: constant function
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    # Use multiple restarts with different initializations
    # 1. Gaussian-like
    # 2. Box-like
    # 3. Random
    
    n_restarts = 10
    for r in range(n_restarts):
        if time.time() - start_time > TIME_LIMIT - 20:
            break
        
        if r == 0:
            # Gaussian initialization
            x = np.linspace(-1, 1, N)
            f_init = np.exp(-5 * x**2)
        elif r == 1:
            # Box initialization
            f_init = np.zeros(N)
            f_init[N//4 : 3*N//4] = 1.0
        else:
            # Random initialization
            f_init = np.random.rand(N)
        
        f_init = np.maximum(f_init, 0.0)
        
        # Run spectral flattening
        f_opt = spectral_flatten(f_init, n_iters=200, alpha=0.5)
        
        # Compute score after flattening
        score_flat = compute_score(f_opt)
        
        # Refine with local search
        f_refined, score_refined = local_refinement(f_opt, start_time, TIME_LIMIT - 5)
        
        if score_refined < best_score:
            best_score = score_refined
            best_f = f_refined
    
    # Ensure nonnegative and return list
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
