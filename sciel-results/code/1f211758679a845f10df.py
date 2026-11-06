import math

# EVOLVE-BLOCK-START
def construct():
    # Candidate vertices: numbers in [1, 255] that share a factor with 255
    # 255 = 3 * 5 * 17
    candidates = [i for i in range(1, 256) if math.gcd(i, 255) > 1]
    
    # Map candidates to indices for faster graph operations
    # idx: value -> index
    idx_map = {v: i for i, v in enumerate(candidates)}
    n = len(candidates)
    
    # Build adjacency matrix as bitsets (integers) for efficiency
    # adj[i] is an integer with bit j set if candidates[i] and candidates[j] are connected
    adj = [0] * n
    for i in range(n):
        vi = candidates[i]
        for j in range(i + 1, n):
            vj = candidates[j]
            if math.gcd(vi, vj) > 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    # Find the index of 255
    idx_255 = idx_map[255]
    
    # We want the max clique containing 255.
    # Initialize Bron-Kerbosch with R={255}, P=neighbors of 255 in candidates\{255}, X=empty
    # Since all candidates share a factor with 255, P is all other candidates.
    
    # Represent P and X as bitsets
    # Bit i is set if candidate i is in the set
    all_bits = (1 << n) - 1
    # P initially contains all candidates except 255
    P = all_bits & ~(1 << idx_255)
    X = 0
    
    max_clique_size = 1  # At least 255
    best_clique = [255]
    
    # Helper to get neighbors of a vertex as a bitmask
    # We need N(v) intersected with a given set
    # adj[v] already contains neighbors.
    
    def bronk(R_size, P_mask, X_mask):
        nonlocal max_clique_size, best_clique
        
        if P_mask == 0 and X_mask == 0:
            if R_size > max_clique_size:
                max_clique_size = R_size
                # Reconstruct clique? 
                # For just the size, we don't need to reconstruct every time.
                # But we need to return the set. 
                # Let's store the best R set.
                # To avoid copying large sets, we can just track the size and 
                # at the end, run a targeted search or store the path.
                # Given the small size, storing the set is fine.
                pass
        
        if P_mask == 0:
            return
        
        # Pivot selection
        # Choose u from P U X that maximizes |P ∩ N(u)|
        # Iterate through bits in P | X
        PX = P_mask | X_mask
        if PX == 0:
            return
            
        best_u = -1
        max_count = -1
        temp = PX
        while temp:
            # Extract lowest set bit
            lsb = temp & -temp
            u = lsb.bit_length() - 1
            temp -= lsb
            
            # Count bits in P ∩ N(u)
            # N(u) is adj[u]
            count = bin(P_mask & adj[u]).count('1')
            if count > max_count:
                max_count = count
                best_u = u
        
        if best_u == -1:
            return
            
        # Iterate over v in P \ N(best_u)
        # P \ N(best_u) = P & ~adj[best_u]
        candidates_for_expansion = P_mask & ~adj[best_u]
        
        temp = candidates_for_expansion
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            temp -= lsb
            
            # Recurse
            # New P = P ∩ N(v)
            new_P = P_mask & adj[v]
            # New X = X ∩ N(v)
            new_X = X_mask & adj[v]
            
            bronk(R_size + 1, new_P, new_X)
            
            # Update P and X for the next iteration
            P_mask = P_mask & ~lsb
            X_mask = X_mask | lsb
            
    # We need to actually store the best clique to return it.
    # Let's modify the function to track the best set.
    
    best_set = {255}
    
    def bronk_v2(R_set, P_mask, X_mask):
        nonlocal max_clique_size, best_set
        
        if P_mask == 0 and X_mask == 0:
            if len(R_set) > max_clique_size:
                max_clique_size = len(R_set)
                best_set = set(R_set)
            return
        
        if P_mask == 0:
            return
        
        # Pivot
        PX = P_mask | X_mask
        if PX == 0:
            return
            
        best_u = -1
        max_count = -1
        temp = PX
        while temp:
            lsb = temp & -temp
            u = lsb.bit_length() - 1
            temp -= lsb
            count = bin(P_mask & adj[u]).count('1')
            if count > max_count:
                max_count = count
                best_u = u
        
        if best_u == -1:
            return
            
        candidates_for_expansion = P_mask & ~adj[best_u]
        temp = candidates_for_expansion
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            temp -= lsb
            
            new_R = R_set | {candidates[v]}
            new_P = P_mask & adj[v]
            new_X = X_mask & adj[v]
            
            bronk_v2(new_R, new_P, new_X)
            
            P_mask = P_mask & ~lsb
            X_mask = X_mask | lsb
            
    bronk_v2({255}, P, X)
    
    return list(best_set)

# EVOLVE-BLOCK-END
