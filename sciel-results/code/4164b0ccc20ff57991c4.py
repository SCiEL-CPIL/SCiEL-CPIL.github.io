"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    # Assume linear shifts (zero-padded)
    # Strategy: Boost edges to maximize tail correlations,
    # which determines the minimum correlation and thus the max overlap.
    
    h = np.full(N, 0.5)
    
    # Boost first and last elements to 1
    if N >= 2:
        h[0] = 1.0
        h[-1] = 1.0
        
        # Adjust the remaining elements to maintain sum = N/2
        # Sum is now 2 + (N-2)*0.5 = N/2 + 1
        # We need to subtract 1 from the sum of h[1:-1]
        if N > 2:
            adjustment = 1.0 / (N - 2)
            h[1:-1] -= adjustment
            
            # Ensure bounds [0, 1]
            h[1:-1] = np.clip(h[1:-1], 0.0, 1.0)
            
            # Re-normalize to exactly N/2 if clipping occurred
            current_sum = np.sum(h)
            target_sum = N / 2.0
            diff = target_sum - current_sum
            if abs(diff) > 1e-9:
                # Distribute difference evenly
                h += diff / N
                h = np.clip(h, 0.0, 1.0)
                
                # Final safety check
                final_sum = np.sum(h)
                if abs(final_sum - target_sum) > 1e-9:
                    # Force exact sum by adjusting the last element
                    h[-1] += target_sum - final_sum
                    h = np.clip(h, 0.0, 1.0)
    
    return h
# EVOLVE-BLOCK-END
