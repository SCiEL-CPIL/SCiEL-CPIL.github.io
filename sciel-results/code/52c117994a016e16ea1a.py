"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_c3_vec(signs):
    """
    Compute the upper bound C3 for a signed step function with N intervals.
    C3 = (1/3) * max |autocorrelation| where autocorrelation is computed
    on the discrete grid.
    
    For a signed step function s[i] in {-1, 1}, the autoconvolution
    (correlation) at lag k is sum_i s[i]*s[i+k].
    The L-infinity norm of the autoconvolution is max_k |sum_i s[i]*s[i+k]|.
    The upper bound C3 is (1/3) * max_k |sum_i s[i]*s[i+k]|.
    """
    # Compute all lags of autocorrelation using FFT for speed
    n_fft = 1 << (2 * N - 1).bit_length()
    f = np.zeros(n_fft, dtype=np.float64)
    f[:N] = signs
    F = np.fft.rfft(f)
    acf = np.fft.irfft(F * np.conj(F), n=n_fft)
    # acf[0:N] are the positive lags, acf[-N+1:] are the negative lags
    # The full autocorrelation sequence
    full_acf = np.concatenate([acf[N-1:0:-1], acf[:N]])
    return np.max(np.abs(full_acf)) / 3.0

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial random signed step function
    np.random.seed(42)
    best_signs = np.random.choice([-1, 1], size=N).astype(np.float64)
    best_c3 = _compute_c3_vec(best_signs)
    
    # Local search with multi-flip moves
    # Precompute the autocorrelation for efficient updates
    def compute_acf(signs):
        n_fft = 1 << (2 * N - 1).bit_length()
        f = np.zeros(n_fft, dtype=np.float64)
        f[:N] = signs
        F = np.fft.rfft(f)
        acf = np.fft.irfft(F * np.conj(F), n=n_fft)
        full_acf = np.concatenate([acf[N-1:0:-1], acf[:N]])
        return full_acf
    
    acf = compute_acf(best_signs)
    
    # Local search: try single and multi-flip moves
    while time.time() < deadline:
        improved = False
        
        # Try single flips
        for i in range(N):
            # Flip sign at position i
            old_val = best_signs[i]
            best_signs[i] = -old_val
            
            # Update autocorrelation efficiently
            # When we flip s[i], the change in autocorrelation at lag k is:
            # delta_acf[k] = -2 * old_val * (sum of s[i+k] for appropriate k)
            # This is complex to do incrementally, so use FFT for simplicity
            # but only for the current candidate
            
            new_acf = compute_acf(best_signs)
            new_c3 = np.max(np.abs(new_acf)) / 3.0
            
            if new_c3 < best_c3 - 1e-12:
                best_c3 = new_c3
                acf = new_acf
                improved = True
            else:
                best_signs[i] = old_val
        
        # Try double flips
        if not improved:
            # Try pairs of flips
            for i in range(N):
                if time.time() > deadline - 5.0:
                    break
                for j in range(i + 1, N):
                    old_i = best_signs[i]
                    old_j = best_signs[j]
                    best_signs[i] = -old_i
                    best_signs[j] = -old_j
                    
                    new_acf = compute_acf(best_signs)
                    new_c3 = np.max(np.abs(new_acf)) / 3.0
                    
                    if new_c3 < best_c3 - 1e-12:
                        best_c3 = new_c3
                        acf = new_acf
                        improved = True
                    else:
                        best_signs[i] = old_i
                        best_signs[j] = old_j
                
                if time.time() > deadline - 5.0:
                    break
        
        # If no improvement, try a random restart with perturbation
        if not improved:
            # Perturb the current best with random flips
            num_flips = np.random.randint(1, 10)
            flip_indices = np.random.choice(N, size=num_flips, replace=False)
            candidate = best_signs.copy()
            candidate[flip_indices] *= -1
            
            cand_acf = compute_acf(candidate)
            cand_c3 = np.max(np.abs(cand_acf)) / 3.0
            
            if cand_c3 < best_c3 - 1e-12:
                best_signs = candidate
                best_c3 = cand_c3
                acf = cand_acf
    
    # Return the optimal signs as a sequence
    return best_signs
# EVOLVE-BLOCK-END
