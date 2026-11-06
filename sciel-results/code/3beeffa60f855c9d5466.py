import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Leave 10s margin
    
    # Try to find a valid graph for n from 45 down to 38
    # If we find one, return it.
    # If time runs out, return the best valid one found (or seed if none)
    
    best_n = 37
    best_edges = None
    
    # Seed graph for fallback
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    seed_edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
    best_edges = seed_edges
    
    for n in range(45, 37, -1):
        if time.time() > deadline:
            break
            
        # Try multiple random restarts for this n
        for restart in range(50):
            if time.time() > deadline:
                break
                
            # Initialize random graph
            edges = set()
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.5:
                        edges.add((i, j))
            
            # Repair loop
            # We need to check for K5 and Independent 5.
            # If found, fix it.
            
            # To speed up, we can limit the number of repair steps
            for step in range(1000):
                if time.time() > deadline:
                    break
                    
                # Check for K5
                k5 = _find_clique(n, edges, 5)
                if k5:
                    # Remove a random edge from the clique
                    # The clique is a list of 5 vertices
                    # Edges in clique: (v0,v1), (v0,v2), etc.
                    # Pick a random pair from the 5 vertices
                    i, j = random.sample(k5, 2)
                    if i > j: i, j = j, i
                    if (i, j) in edges:
                        edges.remove((i, j))
                    # Continue loop to check again
                    continue
                    
                # Check for Independent Set of size 5
                # An independent set in G is a clique in G_complement
                # Or we can just check non-edges
                indep = _find_independent_set(n, edges, 5)
                if indep:
                    # Add a random edge within the independent set
                    i, j = random.sample(indep, 2)
                    if i > j: i, j = j, i
                    edges.add((i, j))
                    continue
                
                # No violations found
                # Valid graph found
                if n > best_n:
                    best_n = n
                    best_edges = list(edges)
                break
                
            if best_n >= n:
                break
                
    return best_n, best_edges

def _find_clique(n, edges, k):
    # Find a clique of size k
    # Simple recursive search
    # This might be slow if done naively, but n is small (~45)
    # We can use a heuristic: try to build a clique of size k
    # Start with all vertices, prune
    
    # Actually, for k=5, we can just iterate?
    # No, let's use a simple backtracking.
    # To make it fast, we can just pick a random subset of size k and check?
    # No, that's inefficient.
    
    # Let's use a standard clique finding algorithm.
    # Since n is small, we can just check all subsets of size k if we are lucky,
    # but $\binom{45}{5}$ is large.
    
    # Better:
    # 1. Pick a vertex v.
    # 2. Find clique of size k-1 in N(v).
    
    # To avoid full search every time, we can just check if one exists.
    # We can use a simple recursive function.
    
    def _find_clique_in_set(vertices, k, adj):
        if k == 0:
            return []
        if not vertices:
            return None
        if len(vertices) < k:
            return None
            
        # Pick a random vertex to start (or first)
        # To be fast, pick random
        v = random.choice(vertices)
        neighbors = [u for u in vertices if u != v and (v, u) in adj or (u, v) in adj]
        # Note: edges are stored as (min, max)
        
        # We need to check if v is connected to all others in the found clique
        # So we look for clique of size k-1 in neighbors of v that are also in vertices
        
        # Optimization: only consider neighbors of v
        # Wait, the clique must be formed by v and k-1 vertices from N(v)
        
        # Let's try to find k-1 clique in N(v) \cap vertices
        # But we must ensure the resulting set is a clique.
        
        # Recursive call
        # We need to pass the adjacency restricted to N(v)
        
        # This recursive approach might be slow in Python if not careful.
        # Let's try a simpler heuristic:
        # Sample random subsets of size k?
        # If the graph is dense or sparse, it might work.
        # But it's unreliable.
        
        # Let's stick to the recursive one but optimize.
        # Actually, for n=45, a simple BFS/DFS for K5 is fast enough if we prune.
        
        pass

    # Let's implement a specific fast K5 finder
    # 1. List all edges.
    # 2. For each edge (u, v), find common neighbors w.
    # 3. For each w, find common neighbors x of u, v, w.
    # 4. For each x, check if there is y connected to u, v, w, x.
    
    # This is O(E * d^2) roughly.
    
    # Let's implement this specific logic.
    
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Precompute common neighbors for all pairs? No, too expensive.
    
    # Iterate over pairs of vertices (u, v) that are connected.
    # For each such pair, get common neighbors.
    # For each common neighbor w, get common neighbors of u, v, w.
    # For each such x, check if there is a y connected to u, v, w, x.
    
    # This is still potentially heavy.
    
    # Alternative: Just check random subsets of size 5?
    # If the graph is "close" to valid, violations are likely to be found quickly by random sampling?
    # Not necessarily.
    
    # Let's use a library if available? No, standard library only.
    
    # Let's try the random sampling first. It's O(1) per check.
    # If we sample 1000 random 5-sets and find a violation, great.
    # If not, we might miss it.
    
    # Let's combine:
    # 1. Random sampling for violations.
    # 2. If found, fix.
    # 3. If not found after many samples, assume valid? Risky.
    
    # Let's implement the structured search for K5.
    # It's safer.
    
    # K5 search:
    # For each vertex v:
    #   For each pair (u, w) in N(v):
    #     If (u, w) in edges:
    #       For each x in N(v) \cap N(u) \cap N(w):
    #         For each y in N(v) \cap N(u) \cap N(w) \cap N(x):
    #           If (x, y) in edges: return [v, u, w, x, y]
    
    # This is bounded by n * (d^2) * (d^2) * d. With d~20, this is 45 * 400 * 400 * 20 ~ 144M ops. Too slow in Python.
    
    # We need a faster check.
    # Bitmasks!
    # Represent adjacency as bitmasks (integers).
    # n <= 45 fits in a 64-bit integer.
    
    adj_masks = [0] * n
    for u, v in edges:
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)
        
    def _find_clique_bitmask(n, adj_masks, k):
        # Find clique of size k
        # Use recursive search with bitmasks
        # This is very fast in Python if implemented correctly
        
        # We can just iterate through vertices?
        # Or use a simple backtracking.
        
        # Let's use a simple heuristic:
        # Try to find a clique of size k.
        
        # We can use the fact that if a clique of size k exists, 
        # we can find it by checking subsets.
        
        # Let's try to find K5 specifically.
        
        # Iterate v from 0 to n-1
        # For each v, we need K4 in N(v).
        # To find K4 in N(v):
        #   Iterate u in N(v)
        #   For each u, we need K3 in N(u) \cap N(v)
        #   To find K3:
        #      Iterate w in N(u) \cap N(v)
        #      For each w, check if N(w) \cap N(u) \cap N(v) has any vertex x.
        #      If yes, we have K4 {v, u, w, x}.
        #      Then check if N(x) \cap N(v) \cap N(u) \cap N(w) has any vertex y.
        #      If yes, we have K5 {v, u, w, x, y}.
        
        # This is much faster.
        
        for v in range(n):
            # Get neighbors of v as a bitmask
            mask_v = adj_masks[v]
            
            # Iterate u in neighbors of v
            # We can iterate through bits in mask_v
            mask_u_candidates = mask_v
            while mask_u_candidates:
                u = (mask_u_candidates & -mask_u_candidates).bit_length() - 1
                mask_u_candidates &= mask_u_candidates - 1
                
                # We need K3 in N(u) \cap N(v)
                # mask_u is neighbors of u
                # common_u_v = adj_masks[u] & mask_v
                common_u_v = adj_masks[u] & mask_v
                
                # Iterate w in common_u_v
                mask_w_candidates = common_u_v
                while mask_w_candidates:
                    w = (mask_w_candidates & -mask_w_candidates).bit_length() - 1
                    mask_w_candidates &= mask_w_candidates - 1
                    
                    # We need a vertex x in N(w) \cap N(u) \cap N(v)
                    # common_u_v_w = common_u_v & adj_masks[w]
                    common_u_v_w = common_u_v & adj_masks[w]
                    
                    if common_u_v_w:
                        # We have a K4 {v, u, w, x} for any x in common_u_v_w
                        # Now we need to check if any such x has a neighbor y in N(x) \cap N(v) \cap N(u) \cap N(w)
                        # Actually, we need y to be connected to v, u, w, x.
                        # So y must be in N(v) \cap N(u) \cap N(w) \cap N(x)
                        
                        # Let's pick one x from common_u_v_w
                        # To be thorough, we should check all? 
                        # Or just one? If we find one, we return.
                        
                        # Let's iterate x in common_u_v_w
                        mask_x_candidates = common_u_v_w
                        while mask_x_candidates:
                            x = (mask_x_candidates & -mask_x_candidates).bit_length() - 1
                            mask_x_candidates &= mask_x_candidates - 1
                            
                            # Check for y in N(x) \cap N(v) \cap N(u) \cap N(w)
                            # mask_y_candidates = adj_masks[x] & mask_v & adj_masks[u] & adj_masks[w]
                            # Note: mask_v is N(v). adj_masks[u] is N(u).
                            # We already know x is in N(u) and N(v) and N(w).
                            # So we just need y in N(x) that is also in N(v) \cap N(u) \cap N(w).
                            # Which is exactly common_u_v_w (excluding x itself? No, x is in common_u_v_w).
                            # Wait, y must be distinct from x.
                            # And y must be connected to x.
                            
                            mask_y_candidates = adj_masks[x] & common_u_v_w
                            # We must exclude x itself from the candidate set for y?
                            # adj_masks[x] does not contain x.
                            # common_u_v_w contains x.
                            # So mask_y_candidates will not contain x.
                            
                            if mask_y_candidates:
                                # Found y
                                # y is in mask_y_candidates
                                # We can just return the clique
                                y = (mask_y_candidates & -mask_y_candidates).bit_length() - 1
                                return [v, u, w, x, y]
                                
        return None

    def _find_independent_set_bitmask(n, adj_masks, k):
        # Find independent set of size k
        # This is equivalent to finding a clique of size k in the complement graph.
        # Complement adjacency mask: ~adj_masks[i] & ((1 << n) - 1)
        # But we must exclude i itself.
        
        # Let's just reuse the clique finder on the complement.
        
        comp_masks = [0] * n
        full_mask = (1 << n) - 1
        for i in range(n):
            comp_masks[i] = full_mask & ~adj_masks[i] & ~(1 << i)
            
        return _find_clique_bitmask(n, comp_masks, k)

    # Use these functions in the main loop
    # Replace _find_clique and _find_independent_set with these bitmask versions
    
    # Update the construct function to use these
    # I will rewrite the whole construct function to be clean.

# EVOLVE-BLOCK-START
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Leave 10s margin
    
    best_n = 37
    best_edges = None
    
    # Seed graph for fallback
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    seed_edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
    best_edges = seed_edges
    
    def _find_clique_bitmask(n, adj_masks, k):
        # Find clique of size k
        if k == 0:
            return []
        if k == 1:
            return [0] # Any vertex
        if k == 2:
            for i in range(n):
                for j in range(i+1, n):
                    if (adj_masks[i] & (1 << j)):
                        return [i, j]
            return None
        if k == 3:
            for i in range(n):
                mask_i = adj_masks[i]
                # Find edge in N(i)
                mask_j_candidates = mask_i
                while mask_j_candidates:
                    j = (mask_j_candidates & -mask_j_candidates).bit_length() - 1
                    mask_j_candidates &= mask_j_candidates - 1
                    # Find k in N(i) \cap N(j)
                    mask_k_candidates = mask_i & adj_masks[j]
                    if mask_k_candidates:
                        k = (mask_k_candidates & -mask_k_candidates).bit_length() - 1
                        return [i, j, k]
            return None
        if k == 4:
            for i in range(n):
                mask_i = adj_masks[i]
                mask_j_candidates = mask_i
                while mask_j_candidates:
                    j = (mask_j_candidates & -mask_j_candidates).bit_length() - 1
                    mask_j_candidates &= mask_j_candidates - 1
                    # Find K2 in N(i) \cap N(j)
                    common_ij = mask_i & adj_masks[j]
                    mask_k_candidates = common_ij
                    while mask_k_candidates:
                        k = (mask_k_candidates & -mask_k_candidates).bit_length() - 1
                        mask_k_candidates &= mask_k_candidates - 1
                        # Find l in N(i) \cap N(j) \cap N(k)
                        common_ijk = common_ij & adj_masks[k]
                        if common_ijk:
                            l = (common_ijk & -common_ijk).bit_length() - 1
                            return [i, j, k, l]
            return None
        if k == 5:
            for v in range(n):
                mask_v = adj_masks[v]
                mask_u_candidates = mask_v
                while mask_u_candidates:
                    u = (mask_u_candidates & -mask_u_candidates).bit_length() - 1
                    mask_u_candidates &= mask_u_candidates - 1
                    # Find K3 in N(v) \cap N(u)
                    common_vu = mask_v & adj_masks[u]
                    mask_w_candidates = common_vu
                    while mask_w_candidates:
                        w = (mask_w_candidates & -mask_w_candidates).bit_length() - 1
                        mask_w_candidates &= mask_w_candidates - 1
                        # Find x in N(v) \cap N(u) \cap N(w)
                        common_vuw = common_vu & adj_masks[w]
                        mask_x_candidates = common_vuw
                        while mask_x_candidates:
                            x = (mask_x_candidates & -mask_x_candidates).bit_length() - 1
                            mask_x_candidates &= mask_x_candidates - 1
                            # Find y in N(v) \cap N(u) \cap N(w) \cap N(x)
                            common_vuwx = common_vuw & adj_masks[x]
                            if common_vuwx:
                                y = (common_vuwx & -common_vuwx).bit_length() - 1
                                return [v, u, w, x, y]
            return None
        return None

    def _find_independent_set_bitmask(n, adj_masks, k):
        comp_masks = [0] * n
        full_mask = (1 << n) - 1
        for i in range(n):
            comp_masks[i] = full_mask & ~adj_masks[i] & ~(1 << i)
        return _find_clique_bitmask(n, comp_masks, k)

    for n in range(45, 37, -1):
        if time.time() > deadline:
            break
            
        for restart in range(100):
            if time.time() > deadline:
                break
                
            edges = set()
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.5:
                        edges.add((i, j))
            
            adj_masks = [0] * n
            for u, v in edges:
                adj_masks[u] |= (1 << v)
                adj_masks[v] |= (1 << u)
                
            for step in range(500):
                if time.time() > deadline:
                    break
                    
                k5 = _find_clique_bitmask(n, adj_masks, 5)
                if k5:
                    i, j = random.sample(k5, 2)
                    if i > j: i, j = j, i
                    if (i, j) in edges:
                        edges.remove((i, j))
                        adj_masks[i] &= ~(1 << j)
                        adj_masks[j] &= ~(1 << i)
                    continue
                    
                indep = _find_independent_set_bitmask(n, adj_masks, 5)
                if indep:
                    i, j = random.sample(indep, 2)
                    if i > j: i, j = j, i
                    edges.add((i, j))
                    adj_masks[i] |= (1 << j)
                    adj_masks[j] |= (1 << i)
                    continue
                    
                if n > best_n:
                    best_n = n
                    best_edges = list(edges)
                break
                
            if best_n >= n:
                break
                
    return best_n, best_edges
# EVOLVE-BLOCK-END
