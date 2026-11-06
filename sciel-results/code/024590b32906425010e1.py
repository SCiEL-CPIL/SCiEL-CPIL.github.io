import time
import random
import numpy as np

def construct():
    start_time = time.time()
    time_limit = 55.0
    
    N = 256
    
    # 1. Precompute ALL APs
    ap_indices = []
    # aps_containing[i] will store list of AP indices that include index i (0-indexed)
    aps_containing = [[] for _ in range(N)]
    
    # To speed up generation, we can iterate d, s, and compute indices directly
    # But we need to store them to build aps_containing
    # Number of APs is ~45k. Storing lists of ints is fine.
    
    for d in range(1, N + 1):
        for s in range(1, N + 1):
            # Max length
            L_max = (N - s) // d + 1
            if L_max < 1:
                continue
            # Generate all lengths
            # For each length L, the AP is s, s+d, ..., s+(L-1)d
            # Note: The AP of length L is a subset of the AP of length L+1.
            # We need to include ALL of them.
            
            # Optimization: Instead of generating lists for every L,
            # we can just note that the AP of length L ends at s+(L-1)d.
            # But we need the list of indices for the dot product / update.
            
            # Let's just generate them.
            # For a fixed d and s, the base indices are base = [s-1, s-1+d, ...]
            # AP of length L uses base[0:L]
            
            base_indices = []
            curr = s - 1
            while curr < N:
                base_indices.append(curr)
                curr += d
                
            # Now for L in 1..len(base_indices):
            # The AP is base_indices[:L]
            
            for L in range(1, len(base_indices) + 1):
                idx_list = base_indices[:L]
                ap_idx = len(ap_indices)
                ap_indices.append(idx_list)
                for idx in idx_list:
                    aps_containing[idx].append(ap_idx)
    
    num_aps = len(ap_indices)
    
    # 2. Initialization
    rng = random.Random(42)
    signs = np.array([rng.choice([-1, 1]) for _ in range(N)], dtype=np.int8)
    
    # Calculate initial sums
    # sums[r] = sum(signs[i] for i in ap_indices[r])
    sums = np.zeros(num_aps, dtype=np.int16)
    for r, idx_list in enumerate(ap_indices):
        # Small lists, loop is fine
        s_val = 0
        for idx in idx_list:
            s_val += signs[idx]
        sums[r] = s_val
        
    current_max = np.max(np.abs(sums))
    
    # 3. Simulated Annealing
    T = 1.0
    cooling = 0.9999 # Slow cooling
    
    while time.time() - start_time < time_limit:
        i = rng.randint(0, N - 1)
        
        old_val = signs[i]
        delta = 2 * old_val
        signs[i] = -old_val
        
        # Update sums
        for r in aps_containing[i]:
            sums[r] += delta
            
        new_max = np.max(np.abs(sums))
        
        if new_max < current_max or (T > 0 and rng.random() < np.exp(-(new_max - current_max) / T)):
            current_max = new_max
        else:
            # Revert
            signs[i] = old_val
            for r in aps_containing[i]:
                sums[r] -= delta
                
        T *= cooling
        if T < 0.01:
            T = 0.01 # Floor to allow some exploration? Or break?
            # Let's keep it small but non-zero to escape local minima if possible,
            # though with integer max, small T means only equal or better moves.
            
    return [int(x) for x in signs]
