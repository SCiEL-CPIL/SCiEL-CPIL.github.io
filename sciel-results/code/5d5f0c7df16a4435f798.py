# EVOLVE-BLOCK-START
import random
import itertools
import time

def construct():
    # Use Paley graph on 37 vertices as seed
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    
    # Initialize adjacency as list of bitsets
    n = q
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % q in residues:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    best_n = n
    best_adj = adj[:]
    
    # Helper to check if a set of vertices (bitmask) contains a clique of size k
    def has_clique_in_mask(mask, k, adj_list, n):
        if k == 1:
            return mask != 0
        if k == 2:
            # Check if any edge exists within mask
            # Iterate through bits in mask
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                neighbors_in_mask = adj_list[idx] & mask
                if neighbors_in_mask:
                    return True
                m ^= lsb
            return False
        
        # For k >= 3, use recursive search or combinations
        # For efficiency with small n, combinations of bits
        verts = []
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            verts.append(idx)
            m ^= lsb
            
        if len(verts) < k:
            return False
            
        for combo in itertools.combinations(verts, k):
            # Check if combo forms a clique
            is_clique = True
            for i in range(k):
                for j in range(i+1, k):
                    if not (adj_list[combo[i]] >> combo[j]) & 1:
                        is_clique = False
                        break
                if not is_clique:
                    break
            if is_clique:
                return True
        return False

    def has_independent_in_mask(mask, k, adj_list, n):
        # Independent set in G is clique in complement
        # Complement adjacency: ~adj_list[i] & ((1<<n)-1) & ~(1<<i)
        # We can just check if has_clique_in_mask on complement
        # But building complement mask for every call is expensive.
        # Instead, check if combo is independent: no edges between any pair.
        verts = []
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            verts.append(idx)
            m ^= lsb
            
        if len(verts) < k:
            return False
            
        for combo in itertools.combinations(verts, k):
            is_indep = True
            for i in range(k):
                for j in range(i+1, k):
                    if (adj_list[combo[i]] >> combo[j]) & 1:
                        is_indep = False
                        break
                if not is_indep:
                    break
            if is_indep:
                return True
        return False

    def is_valid_graph(adj_list, n):
        # Check for K5
        # We only need to check if the whole graph has K5 or I5
        # But we are growing, so we only need to check new constraints?
        # No, adding a vertex can create new K5s involving the new vertex.
        # Old K5s are preserved.
        # So we just need to check:
        # 1. Does N(new) contain K4?
        # 2. Does V\N(new) contain I4?
        # If we maintain validity, we only check these.
        pass

    # Growth Loop
    # We want to add vertices.
    # For each new vertex, we need to find a neighborhood N.
    # Constraints:
    # 1. N must not contain K4.
    # 2. V\N must not contain I4.
    
    # Heuristic: Try to find such N.
    # We can iterate through random subsets or use a greedy approach.
    
    for step in range(50): # Try to add up to 50 vertices
        found = False
        # Try multiple random attempts to find a valid N
        for attempt in range(1000):
            # Sample a random subset N of size approx n/2
            # To speed up, we can start with a random vertex and expand/contract
            # Or just random sample
            
            # Random sample size between n/3 and 2n/3
            size = random.randint(max(1, n//3), min(n-1, 2*n//3))
            N_mask = 0
            candidates = random.sample(range(n), size)
            for v in candidates:
                N_mask |= (1 << v)
            
            # Check constraints
            # 1. N_mask has no K4
            if has_clique_in_mask(N_mask, 4, adj, n):
                continue
                
            # 2. V\N_mask has no I4
            V_mask = (1 << n) - 1
            non_N_mask = V_mask & ~N_mask
            if has_independent_in_mask(non_N_mask, 4, adj, n):
                continue
                
            # If both pass, we can add the vertex
            # New vertex index is n
            # Its adjacency is N_mask
            # Update adj list
            adj.append(N_mask)
            for i in range(n):
                if (N_mask >> i) & 1:
                    adj[i] |= (1 << n)
            
            n += 1
            best_n = n
            best_adj = adj[:]
            found = True
            break
            
        if not found:
            # Try a more targeted approach:
            # Pick a random vertex u, set N = {u}
            # Then try to add/remove vertices from N to satisfy constraints
            # This is complex. If random fails, we might stop.
            # But we can try smaller N or larger N.
            
            # Try N = empty set (vertex is isolated)
            # V\N = V. V must have no I4.
            # If V has I4, this fails.
            
            # Try N = V (vertex is universal)
            # N must have no K4.
            
            # If we can't find one, break
            break
            
    # Reconstruct edges
    edges = []
    for i in range(best_n):
        m = best_adj[i]
        while m:
            lsb = m & -m
            j = lsb.bit_length() - 1
            if i < j:
                edges.append((i, j))
            m ^= lsb
            
    return best_n, edges
# EVOLVE-BLOCK-END
