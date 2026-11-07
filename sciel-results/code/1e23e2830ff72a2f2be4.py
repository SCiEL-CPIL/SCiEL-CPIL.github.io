import math
import time

def construct():
    N = 256
    deadline = time.time() + 85  # safety margin before 90s kill
    
    # Weights: vertex i (0-indexed) has weight i+1
    weights = [i + 1 for i in range(N)]
    
    # Build conflict graph: neighbors[i] is bitset of j where gcd(i+1, j+1) > 1
    neighbors = [0] * N
    for i in range(N):
        vi = i + 1
        mask = 0
        for j in range(N):
            if i != j and math.gcd(vi, j + 1) > 1:
                mask |= (1 << j)
        neighbors[i] = mask
    
    # For each vertex, also store the set of vertices that can coexist (non-neighbors + self)
    # can_coexist[i] = all vertices j where j==i or gcd(i+1,j+1)==1
    can_coexist = [0] * N
    all_mask = (1 << N) - 1
    for i in range(N):
        can_coexist[i] = all_mask ^ neighbors[i]
    
    best_sum = [0]
    best_set = [0]
    
    # Branch and bound
    # State: available mask, current sum
    def branch_and_bound(available, current_sum, depth=0):
        if time.time() > deadline:
            return
        
        if available == 0:
            if current_sum > best_sum[0]:
                best_sum[0] = current_sum
                best_set[0] = 0  # We'll reconstruct separately if needed
            return
        
        # Upper bound: sum of weights of all available vertices
        # Quick bound: if current_sum + sum of all available <= best, prune
        # For a tighter bound, use a greedy coloring bound
        # Simple bound first: sum of weights in available
        bound = current_sum
        tmp = available
        while tmp:
            lsb = tmp & (-tmp)
            idx = lsb.bit_length() - 1
            bound += weights[idx]
            tmp ^= lsb
        
        if bound <= best_sum[0]:
            return
        
        # Select vertex with highest degree in available subgraph (most constraints)
        # to maximize pruning
        best_v = -1
        best_deg = -1
        tmp = available
        while tmp:
            lsb = tmp & (-tmp)
            idx = lsb.bit_length() - 1
            deg = bin(neighbors[idx] & available).count('1')
            if deg > best_deg:
                best_deg = deg
                best_v = idx
            tmp ^= lsb
        
        if best_v == -1:
            return
        
        # Branch 1: include best_v
        new_available = available & can_coexist[best_v]
        new_available &= ~(1 << best_v)  # remove best_v itself
        branch_and_bound(new_available, current_sum + weights[best_v], depth + 1)
        
        # Branch 2: exclude best_v
        new_available2 = available & ~(1 << best_v)
        branch_and_bound(new_available2, current_sum, depth + 1)
    
    # Start with all vertices available
    branch_and_bound(all_mask, 0)
    
    # Reconstruct the optimal set by running a greedy selection guided by the optimal sum
    # Since we need the actual set, let's do a second pass or track it
    # For efficiency, let's just find a valid set with sum close to best_sum
    # Actually, let's modify to track the actual set
    
    # Better approach: track the actual set in the B&B
    best_sum2 = [0]
    best_mask2 = [0]
    
    def branch_and_bound2(available, current_sum, current_mask):
        if time.time() > deadline:
            return
        
        if available == 0:
            if current_sum > best_sum2[0]:
                best_sum2[0] = current_sum
                best_mask2[0] = current_mask
            return
        
        bound = current_sum
        tmp = available
        while tmp:
            lsb = tmp & (-tmp)
            idx = lsb.bit_length() - 1
            bound += weights[idx]
            tmp ^= lsb
        
        if bound <= best_sum2[0]:
            return
        
        best_v = -1
        best_deg = -1
        tmp = available
        while tmp:
            lsb = tmp & (-tmp)
            idx = lsb.bit_length() - 1
            deg = bin(neighbors[idx] & available).count('1')
            if deg > best_deg:
                best_deg = deg
                best_v = idx
            tmp ^= lsb
        
        if best_v == -1:
            return
        
        # Include best_v
        new_available = available & can_coexist[best_v] & ~(1 << best_v)
        branch_and_bound2(new_available, current_sum + weights[best_v], current_mask | (1 << best_v))
        
        # Exclude best_v
        new_available2 = available & ~(1 << best_v)
        branch_and_bound2(new_available2, current_sum, current_mask)
    
    branch_and_bound2(all_mask, 0, 0)
    
    # Extract the set
    result_mask = best_mask2[0]
    result = []
    for i in range(N):
        if result_mask & (1 << i):
            result.append(i + 1)
    
    # Ensure 1 is included if possible (it's coprime with everything)
    if 1 not in result:
        result.append(1)
    
    return [int(x) for x in result]
