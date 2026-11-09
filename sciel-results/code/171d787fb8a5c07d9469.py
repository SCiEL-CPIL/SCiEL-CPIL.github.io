"""QUBO-based penalty annealing for minimizing monochromatic 4-APs on [1,256]."""
import numpy as np
import time

def construct():
    N = 256
    rng = np.random.default_rng(42)
    
    # Precompute all 4-term arithmetic progressions (a, a+d, a+2d, a+3d) with d>=1, a+3d <= N
    # Indexing: integers 1..256 map to indices 0..255
    aps = []
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            idx = [a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1]
            aps.append(idx)
    
    aps = np.array(aps, dtype=np.int32)
    num_aps = len(aps)
    
    # Precompute incidence: for each variable index, which APs does it belong to?
    # This allows O(degree) delta energy calculation instead of O(total_aps)
    incidence = [[] for _ in range(N)]
    for ap_idx in range(num_aps):
        for var_idx in aps[ap_idx]:
            incidence[var_idx].append(ap_idx)
    
    # Initial random coloring
    x = rng.integers(0, 2, size=N).astype(np.int8)
    
    # Helper to count mono APs efficiently from scratch
    def count_mono_aps_full(x):
        vals = x[aps]
        return int(((vals[:, 0] == vals[:, 1]) & 
                    (vals[:, 1] == vals[:, 2]) & 
                    (vals[:, 2] == vals[:, 3])).sum())
    
    # Calculate initial energy
    current_energy = count_mono_aps_full(x)
    best_x = x.copy()
    best_energy = current_energy
    
    # Simulated Annealing parameters
    T = 10.0
    T_min = 0.01
    cooling = 0.999
    start_time = time.time()
    max_time = 110.0 # Leave buffer for evaluation overhead
    
    while T > T_min:
        if time.time() - start_time > max_time:
            break
        
        # Pick a random index to flip
        i = rng.integers(0, N)
        
        # Calculate change in energy if we flip x[i]
        delta_e = 0
        old_val = x[i]
        new_val = 1 - old_val
        
        for ap_idx in incidence[i]:
            a, b, c, d = aps[ap_idx]
            
            # Get current values
            v0 = x[a]
            v1 = x[b]
            v2 = x[c]
            v3 = x[d]
            
            # Was it mono?
            was_mono = (v0 == v1) and (v1 == v2) and (v2 == v3)
            
            # Apply flip to i
            if a == i:
                v0 = new_val
            elif b == i:
                v1 = new_val
            elif c == i:
                v2 = new_val
            elif d == i:
                v3 = new_val
            
            # Is it mono now?
            is_mono_now = (v0 == v1) and (v1 == v2) and (v2 == v3)
            
            if was_mono and not is_mono_now:
                delta_e -= 1
            elif not was_mono and is_mono_now:
                delta_e += 1
        
        # Acceptance criterion
        if delta_e < 0:
            x[i] = new_val
            current_energy += delta_e
            if current_energy < best_energy:
                best_energy = current_energy
                best_x = x.copy()
        else:
            if delta_e == 0:
                x[i] = new_val
            else:
                prob = np.exp(-delta_e / T)
                if rng.random() < prob:
                    x[i] = new_val
                    current_energy += delta_e
        
        T *= cooling
    
    return [int(v) for v in best_x]
