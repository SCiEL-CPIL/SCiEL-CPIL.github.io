import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin

    # Candidates: i in [1, 254] such that gcd(i, 255) > 1
    # 255 = 3 * 5 * 17
    candidates = []
    for i in range(1, 255):
        if i % 3 == 0 or i % 5 == 0 or i % 17 == 0:
            candidates.append(i)
    
    n = len(candidates)
    
    # Build adjacency bitmasks
    # adj[i] is an integer where bit j is 1 if gcd(candidates[i], candidates[j]) > 1
    adj = [0] * n
    for i in range(n):
        ci = candidates[i]
        mask = 0
        for j in range(n):
            if i != j and math.gcd(ci, candidates[j]) > 1:
                mask |= (1 << j)
        adj[i] = mask
        
    # All vertices mask
    all_mask = (1 << n) - 1
    
    best_clique_size = 0
    best_clique_mask = 0
    
    def bron_kerbosch(R, P, X):
        nonlocal best_clique_size, best_clique_mask
        if time.time() > deadline:
            return
        
        if not P and not X:
            r_size = R.bit_count()
            if r_size > best_clique_size:
                best_clique_size = r_size
                best_clique_mask = R
            return
        
        if not P:
            return
        
        # Pivot selection: choose vertex in P | X with max degree in P
        union_mask = P | X
        if union_mask == 0:
            return
            
        pivot = -1
        max_deg = -1
        u = union_mask
        while u:
            lsb = u & -u
            v = lsb.bit_length() - 1
            u ^= lsb
            deg = (P & adj[v]).bit_count()
            if deg > max_deg:
                max_deg = deg
                pivot = v
                
        # Candidates: vertices in P not adjacent to pivot
        if pivot != -1:
            candidates_mask = P & ~adj[pivot]
        else:
            candidates_mask = P
            
        # Iterate through candidates
        u = candidates_mask
        while u:
            lsb = u & -u
            v = lsb.bit_length() - 1
            u ^= lsb
            
            # Recurse
            new_R = R | lsb
            new_P = P & adj[v]
            new_X = X & adj[v]
            
            bron_kerbosch(new_R, new_P, new_X)
            
            # Update P and X
            P &= ~lsb
            X |= lsb
            
    bron_kerbosch(0, all_mask, 0)
    
    # Reconstruct result
    result_indices = [i for i in range(n) if (best_clique_mask >> i) & 1]
    result = [candidates[i] for i in result_indices]
    result.append(255)
    
    return result
