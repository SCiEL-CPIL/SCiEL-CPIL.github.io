import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s limit to be safe (kill at 150, buffer 40)
    
    # 1. Generate Weight-4 vectors
    v4_list = []
    for indices in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(indices, signs):
                v[idx] = s
            v4_list.append(v)
    V4 = np.array(v4_list, dtype=np.int8)
    
    # 2. Generate Weight-2 vectors
    v2_list = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = 2 * si
                v[j] = 2 * sj
                v2_list.append(v)
    V2 = np.array(v2_list, dtype=np.int8)
    
    # Combine them
    # Note: V2 has norm squared 8? No, I used 2*si, so values are +/-2. Norm squared = 4+4=8.
    # Wait, the task says "Normalizing each vector to norm 2 certifies...".
    # The condition is "minimum pairwise distance is at least the maximum vector norm".
    # If I mix norms, the "maximum vector norm" applies to the whole set.
    # If I use V4 (norm 2) and V2_scaled (norm 2), they are comparable.
    # My V2 above has values +/-2, norm sqrt(8) = 2.82.
    # V4 has values +/-1, norm 2.
    # Max norm is 2.82.
    # Distance between V4 and V2_scaled:
    # d^2 = 4 + 8 - 2 dot.
    # We need d >= 2.82 => d^2 >= 8.
    # 12 - 2 dot >= 8 => 4 >= 2 dot => dot <= 2.
    # If V4=(1,1,1,1,0..) and V2=(2,2,0..), dot = 4. Conflict.
    # This mixing is tricky.
    
    # Let's stick to **same norm** vectors to simplify the constraint to a pure MIS.
    # The weight-4 vectors (norm 2) are the best candidates for high count.
    # The weight-2 vectors (norm sqrt(2)) can be scaled to norm 2 by multiplying by sqrt(2)? No, must be integer.
    # If we use integer vectors, we can't easily mix norms without changing the "max norm" threshold for all.
    # If we use only weight-4 vectors, max norm is 2. Min dist >= 2.
    # This is the standard "kissing number" setup with integer vectors.
    
    # Let's optimize the MIS on V4.
    V = V4
    n_vecs = V.shape[0]
    
    # Compute dot products
    dots = V @ V.T
    
    # Conflict if dot > 2
    # We want MIS in graph where edge exists if dots[i,j] > 2.
    
    best_size = 0
    best_indices = []
    
    # Use a more effective heuristic: 
    # 1. Random order greedy (current)
    # 2. "Low Degree First" greedy: At each step, pick the node with fewest connections to remaining available nodes.
    #    This is expensive to update dynamically.
    # 3. Random Restart with "Shuffled Greedy" is usually good.
    #    Let's increase restarts and use a faster inner loop.
    
    # Pre-compute conflict masks? 
    # conflict_mask[i] is a boolean array of length N.
    # Memory: 3360 * 3360 * 1 byte = ~11 MB. Acceptable.
    conflict_masks = (dots > 2).astype(np.int8)
    # Exclude self? dots[i,i]=4>2, so self is in mask. 
    # When we pick u, we remove u and neighbors.
    # So we remove conflict_masks[u] from available.
    
    for attempt in range(500):
        if time.time() > deadline:
            break
            
        order = np.random.permutation(n_vecs)
        available = np.ones(n_vecs, dtype=bool)
        selected = []
        
        for u in order:
            if not available[u]:
                continue
                
            selected.append(u)
            
            # Remove u and its neighbors
            # conflict_masks[u] includes u (since dots[u,u]=4>2)
            available[conflict_masks[u] == 1] = False
            
        if len(selected) > best_size:
            best_size = len(selected)
            best_indices = selected
            # Early stop if we find a very good one?
            # Theoretical max for this set is unknown, but > 3000 is good.
            if best_size > 3500:
                break

    if best_size == 0:
        # Fallback
        return V.astype(np.int64)

    return V[best_indices].astype(np.int64)
