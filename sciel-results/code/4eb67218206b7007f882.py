import time
import numpy as np

N = 17
MIN_DIST = 6
MAX_WORD = 1 << N
MASKS = []

def _init_masks():
    global MASKS
    for m in range(MAX_WORD):
        if m.bit_count() < MIN_DIST:
            MASKS.append(m)

def construct():
    _init_masks()
    best_code = []
    best_size = 0
    
    # Precompute masks as numpy array
    masks_np = np.array(MASKS, dtype=np.uint32)
    
    start_time = time.time()
    
    # Limit iterations
    max_restarts = 10000
    
    for i in range(max_restarts):
        if time.time() - start_time > 55:
            break
            
        # Initialize availability
        is_avail = np.ones(MAX_WORD, dtype=bool)
        code = []
        
        # Greedy loop
        while True:
            if time.time() - start_time > 55.5:
                break
                
            # Find available candidates
            avail_idx = np.nonzero(is_avail)[0]
            if len(avail_idx) == 0:
                break
                
            # Sample candidates for MRV heuristic
            sample_size = min(50, len(avail_idx))
            if sample_size == 0:
                break
                
            samples = np.random.choice(avail_idx, sample_size, replace=False)
            
            best_u = -1
            best_deg = float('inf')
            
            for u in samples:
                # Calculate degree: number of available neighbors
                neighbors = u ^ masks_np
                deg = np.sum(is_avail[neighbors])
                
                if deg < best_deg:
                    best_deg = deg
                    best_u = u
                    
            if best_u == -1:
                break
                
            code.append(int(best_u))
            
            # Mark u and its neighbors as unavailable
            neighbors = best_u ^ masks_np
            is_avail[neighbors] = False
            is_avail[best_u] = False
            
        if len(code) > best_size:
            best_size = len(code)
            best_code = code.copy()

    return best_code
