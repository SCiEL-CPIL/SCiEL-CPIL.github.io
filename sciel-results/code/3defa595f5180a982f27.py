"""Vectorized QUBO-style penalty annealing for minimizing monochromatic 4-APs on [1,256]."""
import numpy as np
import time

def construct():
    N = 256
    rng = np.random.default_rng(42)
    
    # Precompute all 4-term arithmetic progressions
    aps_list = []
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            idx = [a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1]
            aps_list.append(idx)
    
    aps = np.array(aps_list, dtype=np.int32)
    num_aps = len(aps)
    
    # Initial random coloring
    x = rng.integers(0, 2, size=N).astype(np.int8)
    
    # Helper to count mono APs efficiently from scratch
    def count_mono_aps_full(x):
        vals = x[aps]
        return int(((vals[:, 0] == vals[:, 1]) & 
                    (vals[:, 1] == vals[:, 2]) & 
                    (vals[:, 2] == vals[:, 3])).sum())
    
    current_energy = count_mono_aps_full(x)
    best_x = x.copy()
    best_energy = current_energy
    
    # Simulated Annealing parameters
    T = 5.0
    T_min = 0.01
    cooling = 0.9995
    start_time = time.time()
    max_time = 110.0
    
    # Precompute flattened indices for bincount
    var_indices_flat = aps.ravel()
    
    while T > T_min:
        if time.time() - start_time > max_time:
            break
        
        # 1. Compute current values for all APs
        vals = x[aps]
        
        # 2. Compute "others equal 0" and "others equal 1" for each position
        # oeq0[k, pos] is 1 if the other 3 vars in AP k are all 0
        # We can compute this using boolean logic on vals
        
        # For pos 0: others are 1,2,3
        oeq0_0 = (vals[:, 1] == 0) & (vals[:, 2] == 0) & (vals[:, 3] == 0)
        oeq0_1 = (vals[:, 0] == 0) & (vals[:, 2] == 0) & (vals[:, 3] == 0)
        oeq0_2 = (vals[:, 0] == 0) & (vals[:, 1] == 0) & (vals[:, 3] == 0)
        oeq0_3 = (vals[:, 0] == 0) & (vals[:, 1] == 0) & (vals[:, 2] == 0)
        
        oeq1_0 = (vals[:, 1] == 1) & (vals[:, 2] == 1) & (vals[:, 3] == 1)
        oeq1_1 = (vals[:, 0] == 1) & (vals[:, 2] == 1) & (vals[:, 3] == 1)
        oeq1_2 = (vals[:, 0] == 1) & (vals[:, 1] == 1) & (vals[:, 3] == 1)
        oeq1_3 = (vals[:, 0] == 1) & (vals[:, 1] == 1) & (vals[:, 2] == 1)
        
        # Stack into arrays of shape (num_aps, 4)
        oeq0 = np.stack([oeq0_0, oeq0_1, oeq0_2, oeq0_3], axis=1).astype(np.float32)
        oeq1 = np.stack([oeq1_0, oeq1_1, oeq1_2, oeq1_3], axis=1).astype(np.float32)
        
        # 3. Compute delta for flipping each variable
        # If x[i] == 0, flipping to 1:
        #   New mono if others are 1 (oeq1)
        #   Old mono if others are 0 (oeq0)
        #   Delta = oeq1 - oeq0
        # If x[i] == 1, flipping to 0:
        #   New mono if others are 0 (oeq0)
        #   Old mono if others are 1 (oeq1)
        #   Delta = oeq0 - oeq1
        
        # We need to sum these contributions over all APs for each variable index.
        # var_indices_flat has the variable index for each (AP, Pos) pair.
        # oeq1.ravel() and oeq0.ravel() have the corresponding weights.
        
        delta_if_0 = np.bincount(var_indices_flat, weights=(oeq1 - oeq0).ravel(), minlength=N)
        delta_if_1 = np.bincount(var_indices_flat, weights=(oeq0 - oeq1).ravel(), minlength=N)
        
        # Select delta based on current x
        delta_all = np.where(x == 0, delta_if_0, delta_if_1)
        
        # 4. Selection and Acceptance
        # Pick a random variable
        i = rng.integers(0, N)
        d = delta_all[i]
        
        if d < 0:
            x[i] = 1 - x[i]
            current_energy += d
            if current_energy < best_energy:
                best_energy = current_energy
                best_x = x.copy()
        else:
            if d == 0:
                x[i] = 1 - x[i]
            else:
                # Accept with probability exp(-d/T)
                # Optimization: if d/T is large, exp is 0
                if d < T * 5: # Approx threshold where exp becomes negligible
                    if rng.random() < np.exp(-d / T):
                        x[i] = 1 - x[i]
                        current_energy += d
        
        T *= cooling
    
    return [int(v) for v in best_x]
