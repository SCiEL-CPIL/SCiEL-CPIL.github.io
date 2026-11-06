import numpy as np
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    # Safety margin: Evaluator kills at 150s. We use 145s.
    deadline = start_time + 145.0
    
    N = 256
    
    # 1. Precompute all 4-term arithmetic progressions (0-indexed)
    # A 4-AP is (a, a+d, a+2d, a+3d) with 0 <= a, a+3d < N
    # d >= 1
    ap_list = []
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        for a in range(0, max_a):
            ap_list.append((a, a+d, a+2*d, a+3*d))
            
    if not ap_list:
        return [0] * N
        
    aps = np.array(ap_list, dtype=np.int32)
    num_aps = len(aps)
    
    # 2. Precompute sparse adjacency for fast delta calculation
    # For each index i, we need to know:
    # - Which APs it belongs to
    # - For each such AP, what is its position (0, 1, 2, or 3)
    # - The indices of the other 3 elements in that AP
    
    # index_to_aps[i] = list of tuples (ap_idx, pos, other1, other2, other3)
    index_to_aps = [[] for _ in range(N)]
    
    for ap_idx in range(num_aps):
        a, b, c, d = aps[ap_idx]
        index_to_aps[a].append((ap_idx, 0, b, c, d))
        index_to_aps[b].append((ap_idx, 1, a, c, d))
        index_to_aps[c].append((ap_idx, 2, a, b, d))
        index_to_aps[d].append((ap_idx, 3, a, b, c))
    
    # Convert to lists of lists for faster access, or keep as is.
    # For numpy efficiency in the inner loop, we might just stick to Python lists 
    # but optimize the logic. 
    # Actually, for N=256, the degree is small. 
    # Let's precompute the "neighbors" in a way that allows fast vectorized or looped check.
    
    # We will maintain the current coloring `c` as a numpy array of int8.
    # We will maintain the current score `S`.
    
    # To calculate delta for flipping index i:
    # For each AP j containing i:
    #   Let others be the 3 other indices.
    #   Let val_i = c[i]
    #   Let new_val_i = 1 - val_i
    #   Let others_vals = [c[o1], c[o2], c[o3]]
    #   Check if AP was mono before: (val_i == o1 == o2 == o3)
    #   Check if AP is mono after:   (new_val_i == o1 == o2 == o3)
    #   delta_j = (is_mono_after ? 1 : 0) - (is_mono_before ? 1 : 0)
    #   total_delta += delta_j
    
    # We can precompute for each index i, the list of (ap_idx, pos, others_tuple).
    # But checking mono status requires accessing c[others].
    
    # Let's optimize the inner loop.
    # Precompute for each i:
    #   neighbors[i] = list of (ap_idx, pos, o1, o2, o3)
    
    # We can speed up the mono check by precomputing the "state" of the other 3?
    # No, c changes.
    
    # However, we can precompute the list of neighbors as numpy arrays for each i?
    # That might be overkill. Python loops over small lists are okay, but we need MANY steps.
    
    # Let's try to vectorize the delta calculation for a SINGLE flip i.
    # Neighbors of i: list of tuples.
    # We can extract arrays: ap_idxs, o1s, o2s, o3s.
    # Then:
    #   vals_o1 = c[o1s]
    #   vals_o2 = c[o2s]
    #   vals_o3 = c[o3s]
    #   others_all_equal = (vals_o1 == vals_o2) & (vals_o2 == vals_o3)
    #   
    #   # Before flip:
    #   # Mono if val_i == vals_o1 (and others_all_equal)
    #   is_mono_before = others_all_equal & (val_i == vals_o1)
    #   
    #   # After flip:
    #   # Mono if (1-val_i) == vals_o1 (and others_all_equal)
    #   is_mono_after = others_all_equal & ((1 - val_i) == vals_o1)
    #   
    #   delta = np.sum(is_mono_after.astype(np.int32) - is_mono_before.astype(np.int32))
    
    # This is O(degree) per flip, which is very fast.
    
    # Let's precompute these arrays for each i to avoid tuple unpacking in the loop.
    # neighbor_data[i] = (o1s, o2s, o3s) as numpy arrays.
    
    neighbor_data = []
    for i in range(N):
        if not index_to_aps[i]:
            neighbor_data.append(None)
            continue
            
        # Extract others
        o1s = []
        o2s = []
        o3s = []
        for ap_idx, pos, o1, o2, o3 in index_to_aps[i]:
            o1s.append(o1)
            o2s.append(o2)
            o3s.append(o3)
            
        neighbor_data.append((np.array(o1s, dtype=np.int32), 
                              np.array(o2s, dtype=np.int32), 
                              np.array(o3s, dtype=np.int32)))
        
    # Simulated Annealing
    rng = np.random.default_rng()
    
    best_score = np.inf
    best_coloring = None
    
    # Parameters
    T_start = 10.0
    T_end = 0.1
    cooling_rate = 0.999 # Adjust based on steps
    
    # We will run multiple restarts.
    # Each restart runs SA for a fixed number of steps or until T is low.
    
    num_restarts = 20
    steps_per_restart = 5000 # Estimate
    
    # To ensure we don't time out, we will track time and break if needed.
    
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Initialize coloring
        # Random start
        c = rng.integers(0, 2, size=N, dtype=np.int8)
        
        # Calculate initial score
        # We can use the vectorized full check for initialization
        ca = c[aps[:, 0]]
        cb = c[aps[:, 1]]
        cc = c[aps[:, 2]]
        cd = c[aps[:, 3]]
        is_mono = (ca == cb) & (cb == cc) & (cc == cd)
        current_score = np.sum(is_mono)
        
        if current_score < best_score:
            best_score = current_score
            best_coloring = c.copy()
            
        T = T_start
        # We will run for a number of steps, but check time periodically.
        # steps_per_restart is a target, but we might do fewer if time runs out.
        
        for step in range(steps_per_restart):
            if time.time() > deadline - 2:
                break
                
            # Pick a random index to flip
            i = rng.integers(0, N)
            
            # Calculate delta
            data = neighbor_data[i]
            if data is None:
                # No APs contain i, delta is 0. (Impossible for N=256, but safe)
                delta = 0
            else:
                o1s, o2s, o3s = data
                vals_o1 = c[o1s]
                vals_o2 = c[o2s]
                vals_o3 = c[o3s]
                
                others_all_equal = (vals_o1 == vals_o2) & (vals_o2 == vals_o3)
                
                val_i = c[i]
                new_val_i = 1 - val_i
                
                # Mono before: others_all_equal AND val_i == vals_o1
                is_mono_before = others_all_equal & (val_i == vals_o1)
                
                # Mono after: others_all_equal AND new_val_i == vals_o1
                is_mono_after = others_all_equal & (new_val_i == vals_o1)
                
                delta = np.sum(is_mono_after.astype(np.int32) - is_mono_before.astype(np.int32))
            
            # Acceptance probability
            if delta < 0:
                # Always accept improvements
                accept = True
            else:
                # Accept with probability exp(-delta / T)
                if T > 1e-9:
                    prob = np.exp(-delta / T)
                    accept = rng.random() < prob
                else:
                    accept = False
            
            if accept:
                c[i] = 1 - c[i]
                current_score += delta
                
                if current_score < best_score:
                    best_score = current_score
                    best_coloring = c.copy()
            
            # Cool down
            T *= cooling_rate
            
            # Optional: If T is very low, we might restart or break to save time for other restarts
            if T < T_end:
                break

    # Final check
    if best_coloring is None:
        # Fallback
        return [0] * N
        
    return best_coloring.tolist()
# EVOLVE-BLOCK-END
