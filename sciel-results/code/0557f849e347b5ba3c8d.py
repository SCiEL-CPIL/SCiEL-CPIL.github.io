# EVOLVE-BLOCK-START
import numpy as np
from scipy.signal import fftconvolve

def _optimize_signs(signs):
    """
    Optimizes the signs using efficient O(N) autoconvolution updates.
    Returns the optimized signs and the max autoconvolution value.
    """
    signs = signs.copy()
    s_float = signs.astype(np.float64)
    ac = fftconvolve(s_float, s_float, mode='full')
    current_max_ac = np.max(np.abs(ac))
    
    improved = True
    max_passes = 200
    passes = 0
    
    while improved and passes < max_passes:
        improved = False
        passes += 1
        
        for i in range(N):
            s_i = signs[i]
            
            # Calculate s_shifted = s[n-i] for valid n
            # n ranges from i to i+N-1
            # n-i ranges from 0 to N-1
            # So s_shifted is just s[0:N] shifted?
            # Wait.
            # ac index n.
            # s_shifted[n] = s[n-i].
            # Valid n for ac: 0 to 2N-1.
            # Valid n-i for s: 0 to N-1.
            # So n must be in [i, i+N-1].
            
            start_idx = i
            end_idx = i + N
            
            # Clip to ac bounds [0, 2N-1]
            s_idx_start = max(0, start_idx)
            s_idx_end = min(2*N, end_idx)
            
            # Corresponding indices in s:
            # n goes from s_idx_start to s_idx_end - 1
            # n-i goes from s_idx_start - i to s_idx_end - 1 - i
            s_slice = signs[s_idx_start - i : s_idx_end - i]
            
            # Apply update: ac -= 4 * s_i * s_slice
            # Note: s_slice is int, s_i is int
            # We update ac (float)
            
            # We need to be careful with the delta term
            # ac[2*i] += 4.0
            
            # To check improvement, we can compute the new max.
            # But updating the whole ac and checking max is O(N).
            
            # Let's store the old values to revert if needed?
            # Or just revert mathematically.
            
            # Save the segment to revert? No, revert math is exact.
            
            ac[s_idx_start:s_idx_end] -= 4.0 * s_i * s_slice.astype(np.float64)
            ac[2*i] += 4.0
            
            new_max_ac = np.max(np.abs(ac))
            
            if new_max_ac < current_max_ac:
                current_max_ac = new_max_ac
                signs[i] = -s_i
                improved = True
            else:
                # Revert
                ac[s_idx_start:s_idx_end] += 4.0 * s_i * s_slice.astype(np.float64)
                ac[2*i] -= 4.0
                
    return signs, current_max_ac

def construct():
    """
    Constructs the optimal signed step function.
    Uses multiple random restarts and efficient hill-climbing.
    """
    best_signs = None
    best_metric = np.inf
    
    # Use a few random restarts to escape local minima
    for seed in [42, 123, 999, 2024, 777]:
        np.random.seed(seed)
        init_signs = np.random.choice([-1, 1], size=N).astype(np.int32)
        
        opt_signs, metric = _optimize_signs(init_signs)
        
        if metric < best_metric:
            best_metric = metric
            best_signs = opt_signs
            
    return best_signs
# EVOLVE-BLOCK-END
