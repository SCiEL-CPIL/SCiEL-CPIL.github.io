import time
import numpy as np

DIMENSIONS = [2, 2, 9]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Try to build a rank 23 solution
    # Split p=9 into p1=4 and p2=5
    # For each block, use Strassen-like decomposition
    
    def build_strassen_block(k_start, k_end, block_size):
        # Returns U, V, W for a 2x2 x block_size block
        # Indices for U: 4
        # Indices for V: 2 * block_size
        # Indices for W: block_size * 2
        
        # Strassen for 2x2x2 uses 7 ranks
        # We can tile this
        
        pass

    # Fallback to current rank 32 if construction fails or takes too long
    # ... (code from current program)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
