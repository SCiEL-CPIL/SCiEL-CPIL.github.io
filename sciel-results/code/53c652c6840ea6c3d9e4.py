import math

def construct():
    candidates = [x for x in range(1, 256) if math.gcd(x, 255) > 1]
    n = len(candidates)
    num_to_idx = {num: i for i, num in enumerate(candidates)}
    
    adj = [0] * n
    for i in range(n):
        ai = candidates[i]
        for j in range(i + 1, n):
            aj = candidates[j]
            if math.gcd(ai, aj) > 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    best_clique = []
    
    # Seed with multiples of 3
    clique_3 = [x for x in candidates if x % 3 == 0]
    if len(clique_3) > len(best_clique):
        best_clique = clique_3
        
    # Also try multiples of 5? Size 51.
    # Multiples of 17? Size 15.
    # 85 is the largest single-factor clique.
    
    idx_255 = num_to_idx[255]
    initial_R = [255]
    initial_P = adj[idx_255]
    
    def expand(R, P):
        nonlocal best_clique
        
        # Bound
        if len(R) + P.bit_count() <= len(best_clique):
            return
            
        if len(R) > len(best_clique):
            best_clique = list(R)
            
        if P == 0:
            return
            
        # Heuristic: pick vertex with highest degree in P
        best_v = -1
        best_deg = -1
        temp_P = P
        while temp_P:
            lsb = temp_P & -temp_P
            idx = lsb.bit_length() - 1
            deg = (P & adj[idx]).bit_count()
            if deg > best_deg:
                best_deg = deg
                best_v = idx
            temp_P ^= lsb
            
        if best_v == -1:
            return
            
        P_new = P & ~(1 << best_v)
        
        # Branch 1: Include best_v
        R.append(candidates[best_v])
        expand(R, P_new & adj[best_v])
        R.pop()
        
        # Branch 2: Exclude best_v
        expand(R, P_new)

    expand(initial_R, initial_P)
    
    # Ensure 255 is in the result
    if 255 not in best_clique:
        best_clique.append(255)
        
    return sorted(best_clique)
