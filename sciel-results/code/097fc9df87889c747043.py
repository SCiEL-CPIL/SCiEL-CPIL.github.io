import numpy as np
import time

N = 600

def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin
    
    # Initialize
    # Start with a uniform distribution or random
    f = np.ones(N)
    f /= np.linalg.norm(f)
    
    best_f = f.copy()
    best_score = np.max(np.convolve(f, f, mode='full'))
    
    # Simulated Annealing
    T = 1.0
    cooling_rate = 0.999
    min_T = 1e-5
    
    def get_score(x):
        # Compute max of autoconvolution
        # Use FFT for speed if N is large, but 600 is small enough for direct?
        # Direct is O(N^2) = 360k. FFT is O(N log N).
        # Let's use FFT for robustness.
        F = np.fft.fft(x)
        conv = np.fft.ifft(F * F).real
        # conv is length 2N-1
        return np.max(conv)
    
    current_score = get_score(f)
    best_score = current_score
    best_f = f.copy()
    
    while T > min_T:
        if time.time() - start_time > time_limit:
            break
            
        # Generate neighbor
        # Randomly flip a sign? No, f must be non-negative.
        # Randomly perturb a few elements.
        idx = np.random.randint(0, N, size=5)
        f[idx] += np.random.normal(0, 0.1, size=5)
        f[idx] = np.maximum(f[idx], 0)
        
        # Renormalize to keep scale invariant (or fixed)
        norm = np.linalg.norm(f)
        if norm == 0:
            f = np.ones(N) / np.sqrt(N)
        else:
            f /= norm
            
        new_score = get_score(f)
        
        # Acceptance probability
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_f = f.copy()
        else:
            delta = new_score - current_score
            if delta < 0:
                accept = True
            else:
                accept = np.random.rand() < np.exp(-delta / T)
            
            if accept:
                current_score = new_score
            else:
                # Revert
                # We need to revert the changes. 
                # Simpler: keep a copy of f before perturbation?
                # Or just re-perturb? 
                # For efficiency, let's just accept/reject based on score.
                # If rejected, we should undo the change.
                # Since we modified f in place, we need to store old f.
                # This is inefficient. Let's use a different update strategy.
                pass
        
        T *= cooling_rate

    # Final scaling? 
    # The score is likely scale invariant. 
    # Return best_f
    return list(best_f)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    time_limit = 140.0
    
    f = np.ones(N)
    f /= np.linalg.norm(f)
    
    best_f = f.copy()
    
    def get_score(x):
        F = np.fft.fft(x)
        conv = np.fft.ifft(F * F).real
        return np.max(conv)
    
    best_score = get_score(f)
    current_score = best_score
    
    T = 1.0
    cooling_rate = 0.999
    min_T = 1e-5
    
    while T > min_T:
        if time.time() - start_time > time_limit:
            break
            
        # Perturb
        old_f = f.copy()
        idx = np.random.randint(0, N, size=10)
        f[idx] += np.random.normal(0, 0.5, size=10)
        f[idx] = np.maximum(f[idx], 0)
        
        norm = np.linalg.norm(f)
        if norm < 1e-10:
            f = np.ones(N) / np.sqrt(N)
        else:
            f /= norm
            
        new_score = get_score(f)
        
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_f = f.copy()
        else:
            delta = new_score - current_score
            if np.random.rand() < np.exp(-delta / T):
                current_score = new_score
            else:
                f = old_f
        
        T *= cooling_rate

    return list(best_f)
# EVOLVE-BLOCK-END
