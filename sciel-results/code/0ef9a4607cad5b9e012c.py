"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    n = 256
    start_time = time.time()
    time_limit = 55.0  # Safety margin
    
    # 1. Precompute all Arithmetic Progressions (APs)
    # An AP is any contiguous sub-sequence of indices in a residue class mod d.
    index_to_aps = [[] for _ in range(n)]
    ap_id_counter = 0
    
    for d in range(1, n + 1):
        for r in range(d):
            indices = list(range(r, n, d))
            L = len(indices)
            if L == 0:
                continue
            
            # Generate all contiguous sub-sequences
            for start_idx in range(L):
                for end_idx in range(start_idx, L):
                    ap_id = ap_id_counter
                    ap_id_counter += 1
                    
                    # Map this AP to its constituent global indices
                    for k in range(start_idx, end_idx + 1):
                        global_idx = indices[k]
                        index_to_aps[global_idx].append(ap_id)
                        
    num_aps = ap_id_counter
    
    # Convert to NumPy arrays for fast fancy indexing
    index_to_aps_np = [np.array(lst, dtype=np.int32) if lst else np.array([], dtype=np.int32) for lst in index_to_aps]
    
    # 2. Initialize
    rng = random.Random(42)
    signs = np.array([rng.choice([-1, 1]) for _ in range(n)], dtype=np.int32)
    
    # Initialize AP sums
    ap_sums = np.zeros(num_aps, dtype=np.int32)
    
    # Efficient initialization: add sign[i] to all APs containing i
    for i in range(n):
        ids = index_to_aps_np[i]
        if len(ids) > 0:
            ap_sums[ids] += signs[i]
            
    # Calculate initial metrics
    current_max_disc = np.max(np.abs(ap_sums))
    
    best_signs = signs.copy()
    best_max_disc = current_max_disc
    
    # 3. Simulated Annealing
    T = 10.0
    cooling_rate = 0.999
    min_T = 0.1
    
    # Pre-calculate step counts or just loop until time limit
    while time.time() - start_time < time_limit:
        i = rng.randrange(n)
        ids = index_to_aps_np[i]
        
        if len(ids) == 0:
            continue
            
        # Calculate change in Sum of Squares (proxy for max discrepancy)
        old_sums = ap_sums[ids]
        delta = -2 * signs[i]
        new_sums = old_sums + delta
        
        # Delta E = sum(new^2) - sum(old^2)
        # = sum((old+delta)^2 - old^2) = sum(2*old*delta + delta^2)
        # = 2*delta*sum(old) + len(ids)*delta^2
        # This is faster than summing squares twice
        
        sum_old = np.sum(old_sums)
        delta_E = 2 * delta * sum_old + len(ids) * delta * delta
        
        accept = False
        if delta_E < 0:
            accept = True
        else:
            if T > 0:
                # Check for overflow in exp
                if delta_E / T < 700:
                    accept = (rng.random() < np.exp(-delta_E / T))
        
        if accept:
            signs[i] = -signs[i]
            ap_sums[ids] = new_sums
            
            # Update best
            # np.max is the bottleneck, but necessary for exact score
            current_max_disc = np.max(np.abs(ap_sums))
            
            if current_max_disc < best_max_disc:
                best_max_disc = current_max_disc
                best_signs = signs.copy()
                
        T *= cooling_rate
        if T < min_T:
            T = min_T
            
    return list(best_signs)

# EVOLVE-BLOCK-END
