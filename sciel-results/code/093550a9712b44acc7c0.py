import time
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 55.0 # 5s margin
    
    N_max = 256
    vertices = list(range(2, N_max + 1))
    N = len(vertices)
    
    # Precompute conflicts
    # conflict[i] is a bitmask where bit j is 1 if vertices[i] and vertices[j] conflict
    # We use 0-indexed for the list 'vertices'
    conflicts = [0] * N
    for i in range(N):
        vi = vertices[i]
        mask = 0
        for j in range(N):
            if i == j: continue
            if lcm(vi, vertices[j]) <= 256:
                mask |= (1 << j)
        conflicts[i] = mask
        
    # Sort by weight (1/x) descending -> x ascending? No, 1/x descending means x ascending.
    # Wait. 1/2 > 1/3 > 1/4 ...
    # So sorting by weight descending is sorting by value ascending.
    # vertices are already 2..256. So index 0 is 2, index 1 is 3...
    # This is already sorted by weight descending!
    
    weights = [1.0 / v for v in vertices]
    
    # Greedy initialization
    best_sum = 0.0
    best_set = []
    avail_mask = (1 << N) - 1
    # Greedy: pick highest weight (lowest index) available
    current_mask = avail_mask
    greedy_set = []
    greedy_sum = 0.0
    for i in range(N):
        if (current_mask >> i) & 1:
            greedy_set.append(i)
            greedy_sum += weights[i]
            current_mask &= ~conflicts[i]
            
    best_sum = greedy_sum
    best_set = greedy_set[:]
    
    # B&B
    import sys
    sys.setrecursionlimit(10000)
    
    # Precompute suffix sums of weights? 
    # The bound is sum of weights of available vertices.
    # Calculating this on the fly by iterating bits is O(N/word_size) which is fast.
    
    def bit_count_weight(mask):
        # Sum weights of set bits in mask
        s = 0.0
        m = mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            idx = lsb.bit_length() - 1
            s += weights[idx]
            m ^= lsb
        return s

    # To speed up bound calculation, we can precompute weight sums for chunks?
    # Or just rely on the fact that N=255 is small.
    # bit_count_weight is called at every node. 
    # Number of nodes might be large.
    # Optimization: Pass the current potential sum?
    # When we include j, new potential = old_potential - weights[j] - sum(weights[k] for k in neighbors of j in avail).
    # When we exclude j, new potential = old_potential - weights[j].
    # This avoids re-summing from scratch.
    
    # Let's modify search to pass 'potential'
    
    def search(i, current_sum, current_potential, avail_mask, current_set):
        nonlocal best_sum, best_set, deadline
        
        if time.time() > deadline:
            return
        
        if current_sum + current_potential <= best_sum:
            return
            
        if avail_mask == 0:
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = current_set[:]
            return
        
        # Find lowest set bit in avail_mask (highest weight)
        lsb = avail_mask & -avail_mask
        j = lsb.bit_length() - 1
        
        # Branch 1: Include j
        # New avail: remove j and its neighbors from avail_mask
        # neighbors are conflicts[j]
        # new_avail = avail_mask & ~conflicts[j]
        # Note: conflicts[j] includes j itself? 
        # In my precomputation, I skipped i==j. So bit j is 0 in conflicts[j].
        # So we must manually clear bit j.
        new_avail_include = avail_mask & ~conflicts[j] & ~(1 << j)
        
        # Calculate new potential
        # Potential decreases by weights[j] and weights of all neighbors of j that were in avail_mask
        # removed_mask = avail_mask & conflicts[j]  (neighbors in avail)
        # plus j itself
        removed_sum = weights[j]
        rem_neighbors = avail_mask & conflicts[j]
        # Sum weights of rem_neighbors
        # We can use a helper or just iterate
        # For efficiency, we can precompute weight_sum for any mask? No, too many masks.
        # Just iterate bits of rem_neighbors.
        m = rem_neighbors
        while m:
            lsb_r = m & -m
            idx_r = lsb_r.bit_length() - 1
            removed_sum += weights[idx_r]
            m ^= lsb_r
            
        new_potential_include = current_potential - removed_sum
        
        # Recurse
        new_set = current_set[:]
        new_set.append(j)
        search(j+1, current_sum + weights[j], new_potential_include, new_avail_include, new_set)
        
        # Branch 2: Exclude j
        # New avail: just remove j
        new_avail_exclude = avail_mask & ~(1 << j)
        new_potential_exclude = current_potential - weights[j]
        
        search(j+1, current_sum, new_potential_exclude, new_avail_exclude, current_set)
        
    # Initial potential is sum of all weights
    initial_potential = sum(weights)
    initial_mask = (1 << N) - 1
    search(0, 0.0, initial_potential, initial_mask, [])
    
    # Convert indices back to values
    result = [vertices[i] for i in best_set]
    
    # Verify
    # for a in result:
    #   for b in result:
    #     if a != b and lcm(a,b) <= 256: return [] # Should not happen
    
    return result
