import numpy as np
import time

N = 400
TIME_LIMIT = 120.0
SAFETY_MARGIN = 5.0
DEADLINE = time.time() + TIME_LIMIT - SAFETY_MARGIN

def _autoconv(f):
    """Compute the linear (full) autoconvolution of f."""
    return np.convolve(f, f, mode='full')

def _score_energy(f):
    """
    Calculate the score assuming the denominator is the energy (sum of squares).
    This is a hypothesis based on the potential for improvement over the all-ones baseline.
    """
    energy = np.sum(f * f)
    if energy < 1e-12:
        return 1e10
    C = _autoconv(f)
    return np.max(np.abs(C)) / energy

def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    # Initialize with random binary sequence
    # This naturally has low autoconvolution peaks compared to its energy
    f = rng.choice([-1, 1], size=N)
    
    # Normalize to unit energy? No, score is scale invariant.
    # But let's keep it simple.
    
    best_f = f.copy()
    best_score = _score_energy(best_f)
    
    # Local search: flip signs to reduce the maximum peak
    # We focus on the peak that is currently the maximum
    
    while time.time() < DEADLINE:
        C = _autoconv(f)
        max_idx = np.argmax(np.abs(C))
        max_val = np.abs(C[max_idx])
        
        # If we hit a very low score, we can break or continue refining
        if max_val < 1e-6:
            break
            
        # Try flipping a random set of indices to see if we reduce the max peak
        # The peak C_k depends on f_i and f_{k-i}.
        # Flipping f_i changes C_k by -2 f_i f_{k-i} (if k-i is valid).
        
        # Simple heuristic: flip the index that contributes most to the current peak
        # Or just random restarts if stuck.
        
        # Let's try a few random flips
        improved = False
        for _ in range(100):
            if time.time() > DEADLINE:
                break
                
            # Pick a random index to flip
            idx = rng.integers(0, N)
            
            # Calculate the effect of flipping f[idx]
            # C_new[k] = C_old[k] - 2 * f[idx] * f[k-idx] (if valid)
            # We only care if the max decreases.
            
            # Quick check: compute new autoconvolution
            f_copy = f.copy()
            f_copy[idx] *= -1
            C_new = _autoconv(f_copy)
            new_max = np.max(np.abs(C_new))
            
            if new_max < max_val:
                f = f_copy
                max_val = new_max
                improved = True
                # Update best if better
                current_score = new_max / np.sum(f*f)
                if current_score < best_score:
                    best_score = current_score
                    best_f = f.copy()
                break # Restart loop to find new max index
        
        if not improved:
            # If no single flip improves, try a random restart or perturb
            # Or accept that we are in a local minimum and try a different random start
            if rng.integers(0, 10) == 0: # 10% chance to restart
                f = rng.choice([-1, 1], size=N)
                C = _autoconv(f)
                max_val = np.max(np.abs(C))
                current_score = max_val / np.sum(f*f)
                if current_score < best_score:
                    best_score = current_score
                    best_f = f.copy()

    return best_f
