import numpy as np
import time

N = 600

def construct():
        # Deadline guard
        start_time = time.time()
        deadline = 140.0
        
        # Construct a comb function: 1 on even indices, 0 on odd indices.
        # This creates 300 disjoint blocks of width 1 interval.
        # The autoconvolution peaks are separated by 2 intervals (4*delta time? No, 2*delta shift).
        # Wait, shift for self-peak of block i is 0 relative to block i.
        # But in global time, peak is at 2*x_i.
        # x_i = i * 2 * delta.
        # Peak at 2 * i * 2 * delta = 4 * i * delta.
        # Spacing of peaks 4 * delta.
        # Peak width 2 * delta.
        # Well separated.
        
        h = np.zeros(N)
        h[0::2] = 1.0
        
        return h.tolist()
