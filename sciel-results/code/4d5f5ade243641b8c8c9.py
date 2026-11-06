import math
import time

def construct():
    """
    Returns the largest subset of [1, 255] containing 255 such that
    gcd(a, b) > 1 for all distinct pairs a, b in the subset.
    
    Uses Bron-Kerbosch with bitmasks to find the maximum clique
    in the graph where vertices are numbers in [1,255] with gcd(x,255)>1,
    and edges connect pairs with gcd > 1. The clique must contain 255.
    """
    start_time = time.time()
    DEADLINE = 80.0  # safety margin before 90s kill

    # 255 = 3 * 5 * 17
    # Candidate set: all x in [1,255] with gcd(x, 255) > 1
    # (must be divisible by 3, 5, or 17)
    candidates = [x for x in range(1, 256) if math.gcd(x, 255) > 1]
    n = len(candidates)
    
    # Find index of 255
    idx_255 = candidates.index(255)
    
    # Build adjacency bitmasks: adj[i] has bit j set if gcd(candidates[i], candidates[j]) > 1
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    def popcount(x):
        return bin(x).count('1')
    
    # Initial best: all multiples of 3 (a guaranteed clique containing 255)
    best_size = 0
    best_mask = 0
    for i in range(n):
        if candidates[i] % 3 == 0:
            best_mask |= (1 << i)
    best_size = popcount(best_mask)
    
    # Bron-Kerbosch with pivot for maximum clique
    # We need clique containing idx_255, so start with {idx_255} in clique
    # and candidates = neighbors of idx_255
    
    best_clique_mask = [best_mask]  # use list for mutability in closure
    
    def bk(clique_mask, P, X):
        nonlocal best_size, best_clique_mask
        
        if time.time() > DEADLINE:
            return False  # signal timeout
        
        # Bound: current clique size + remaining candidates can't beat best
        if popcount(clique_mask) + popcount(P) <= best_size:
            return True
        
        if P == 0:
            if popcount(clique_mask) > best_size:
                best_size = popcount(clique_mask)
                best_clique_mask[0] = clique_mask
            return True
        
        # Choose pivot: vertex in P|X with maximum neighbors in P
        PX = P | X
        pivot = -1
        max_deg = -1
        tmp = PX
        while tmp:
            lsb = tmp & (-tmp)
            v = lsb.bit_length() - 1
            deg = popcount(P & adj[v])
            if deg > max_deg:
                max_deg = deg
                pivot = v
            tmp ^= lsb
        
        # Branch on vertices in P that are NOT adjacent to pivot
        if pivot >= 0:
            branch_set = P & ~adj[pivot]
        else:
            branch_set = P
        
        tmp = branch_set
        while tmp:
            lsb = tmp & (-tmp)
            v = lsb.bit_length() - 1
            tmp ^= lsb
            
            new_clique = clique_mask | lsb
            new_P = P & adj[v]
            new_X = X & adj[v]
            
            if not bk(new_clique, new_P, new_X):
                return False
            
            P &= ~lsb
            X |= lsb
        
        return True
    
    # Start BK: clique = {idx_255}, P = neighbors of 255, X = 0
    init_clique = (1 << idx_255)
    init_P = adj[idx_255]
    
    bk(init_clique, init_P, 0)
    
    # Greedy local search polish: try to add vertices one by one
    # Check if adding a vertex maintains the clique property
    result_mask = best_clique_mask[0]
    
    # Try each candidate not in result_mask
    for i in range(n):
        if result_mask & (1 << i):
            continue
        # Check if vertex i is adjacent to all vertices in result_mask
        # i must be adjacent to 255 (guaranteed by candidate filter)
        # and to all other vertices in the clique
        # adj[i] has bits for all neighbors; we need (result_mask & ~adj[i]) == 0
        # i.e., all vertices in result_mask must be neighbors of i
        if (result_mask & ~adj[i]) == 0:
            result_mask |= (1 << i)
    
    # Second pass after additions
    for i in range(n):
        if result_mask & (1 << i):
            continue
        if (result_mask & ~adj[i]) == 0:
            result_mask |= (1 << i)
    
    # Convert mask back to list of values
    result = []
    for i in range(n):
        if result_mask & (1 << i):
            result.append(candidates[i])
    
    # Ensure 255 is in the result
    if 255 not in result:
        result.append(255)
    
    return result
