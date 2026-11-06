import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    def time_left():
        return deadline - time.time()
        
    def generate_paley(q):
        if q % 4 != 1:
            return None
        squares = {x * x % q for x in range(1, q)}
        adj = [0] * q
        for i in range(q):
            for j in range(i + 1, q):
                if (j - i) % q in squares:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    def find_k4s(adj, n):
        k4s = []
        # For each edge (u, v), check common neighbors
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    # Common neighbors of u and v
                    common = adj[u] & adj[v]
                    # Remove u and v from common? They aren't in each other's adj in simple graph logic usually, 
                    # but adj[u] has bit v set. So common has bit v? No, adj[v] has bit u set.
                    # adj[u] & adj[v] contains vertices connected to both.
                    # u is not connected to u. v is not connected to v.
                    # So common is correct.
                    # We need to check if any two vertices in common are connected.
                    # Iterate through pairs in common
                    common_list = [i for i in range(n) if (common >> i) & 1]
                    for i in range(len(common_list)):
                        x = common_list[i]
                        for j in range(i + 1, len(common_list)):
                            y = common_list[j]
                            if (adj[x] >> y) & 1:
                                # u, v, x, y form K4
                                k4s.append((u, v, x, y))
        return k4s

    def max_independent_set_size(adj, n):
        # Greedy approximation for speed
        # We just need to know if it's >= 6
        # Use a simple greedy: pick vertex with min degree, remove it and neighbors
        # Repeat
        # To be faster, we can use a heuristic or just check if a specific set is independent.
        # For repair, we need the actual set to break it.
        
        remaining = list(range(n))
        indep_set = []
        while remaining:
            # Pick vertex with min degree in remaining subgraph
            min_deg = n
            min_v = -1
            for v in remaining:
                deg = bin(adj[v] & (sum(1 << u for u in remaining) if len(remaining) < 64 else (1<<n)-1)).count('1')
                # Optimization: use bit_count if available
                if hasattr(adj[v], 'bit_count'):
                    mask = 0
                    for u in remaining:
                        mask |= (1 << u)
                    deg = (adj[v] & mask).bit_count()
                else:
                    deg = bin(adj[v]).count('1') # Approximate, might be wrong if we don't mask
                    # Let's stick to a simpler check for small n
                if deg < min_deg:
                    min_deg = deg
                    min_v = v
            
            if min_v == -1: break
            
            indep_set.append(min_v)
            # Remove min_v and its neighbors from remaining
            neighbors = [u for u in remaining if (adj[min_v] >> u) & 1]
            to_remove = set(neighbors)
            to_remove.add(min_v)
            remaining = [v for v in remaining if v not in to_remove]
            
            if len(indep_set) >= 6:
                return indep_set[:6]
        
        return indep_set

    def is_valid(adj, n):
        # Check K4
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    common = adj[u] & adj[v]
                    common_list = [i for i in range(n) if (common >> i) & 1]
                    for i in range(len(common_list)):
                        x = common_list[i]
                        for j in range(i + 1, len(common_list)):
                            y = common_list[j]
                            if (adj[x] >> y) & 1:
                                return False
        # Check Independent Set
        # Use a faster check: try to find an independent set of size 6
        # This is NP-hard, but for n=37, we can use a heuristic or branch and bound
        # For now, use the greedy one to get a candidate, then verify
        # Actually, we need to be sure.
        # Let's use a simple backtracking for independent set of size 6
        # If n is small, it's fast.
        
        def has_independent_set_of_size_k(adj, n, k):
            # Branch and bound
            # Sort vertices by degree?
            # Simple recursion
            # To speed up, use bitmasks
            # We want to select k vertices such that no two are adjacent.
            
            # Use a heuristic: try to build the set
            # If we fail, it might still exist.
            # Given time constraints, we might rely on the repair process finding violations.
            # But for final validation, we need to be exact.
            # However, the task says "checked exactly".
            # I will implement a fast exact checker for alpha >= 6 using branch and bound with bitmasks.
            
            # Precompute non-adjacency masks?
            # non_adj[i] has bits set for j not adjacent to i (and j != i)
            non_adj = [0] * n
            for i in range(n):
                for j in range(n):
                    if i != j and not ((adj[i] >> j) & 1):
                        non_adj[i] |= (1 << j)
                        
            # We want to find a clique in the complement graph of size 6.
            # Complement graph edges are non_adj.
            
            def backtrack_clique(current_clique, candidates_mask):
                if len(current_clique) == k:
                    return True
                if not candidates_mask:
                    return False
                # Pruning: if not enough candidates left
                if candidates_mask.bit_count() < k - len(current_clique):
                    return False
                
                # Pick a candidate
                # Heuristic: pick one with max degree in candidates
                # For speed, just pick the first one
                c = (candidates_mask & -candidates_mask).bit_length() - 1
                # Remove c from candidates for next step
                new_candidates = candidates_mask & ~(1 << c)
                
                # Intersect with non_adj[c] to ensure independence in G (clique in Gbar)
                new_candidates &= non_adj[c]
                
                if backtrack_clique(current_clique + [c], new_candidates):
                    return True
                
                # Backtrack: try other candidates
                # This is a simple DFS. It might be slow.
                # But for n=37, k=6, it should be okay with good pruning.
                # To improve, we can iterate through all candidates
                # Let's rewrite to iterate
                return False
            
            # Better backtracking:
            def find_clique(candidates_mask, depth, k):
                if depth == k:
                    return True
                if candidates_mask == 0:
                    return False
                if candidates_mask.bit_count() < k - depth:
                    return False
                
                # Pick a vertex v from candidates
                # We must pick one. Let's pick the first one.
                # Then we must include it or exclude it?
                # Standard clique finding:
                # Pick v.
                # 1. Include v: recurse on (candidates & non_adj[v])
                # 2. Exclude v: recurse on (candidates & ~(1<<v))
                
                v = (candidates_mask & -candidates_mask).bit_length() - 1
                v_bit = 1 << v
                
                # Include v
                if find_clique(candidates_mask & non_adj[v], depth + 1, k):
                    return True
                
                # Exclude v
                if find_clique(candidates_mask & ~v_bit, depth, k):
                    return True
                
                return False
            
            return find_clique((1 << n) - 1, 0, k)
        
        if has_independent_set_of_size_k(adj, n, 6):
            return False
        return True

    # Main Logic
    best_n = 0
    best_adj = None
    best_edges = None
    
    # Try sizes from 37 down to 35
    for n in [37, 36, 35]:
        if time_left() < 10:
            break
            
        # Initialize with Paley graph if possible, else random
        if n == 37:
            adj = generate_paley(37)
        elif n == 36:
            # Take PG(37) and remove a vertex?
            adj37 = generate_paley(37)
            if adj37:
                # Remove vertex 36
                adj = [0] * 36
                for i in range(36):
                    for j in range(36):
                        if (adj37[i] >> j) & 1:
                            adj[i] |= (1 << j)
            else:
                adj = [0] * 36
                for i in range(36):
                    for j in range(i+1, 36):
                        if random.random() < 0.5:
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
        else: # n=35
            adj37 = generate_paley(37)
            if adj37:
                adj = [0] * 35
                for i in range(35):
                    for j in range(35):
                        if (adj37[i] >> j) & 1:
                            adj[i] |= (1 << j)
            else:
                adj = [0] * 35
                for i in range(35):
                    for j in range(i+1, 35):
                        if random.random() < 0.5:
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
        
        # Repair Phase 1: Remove K4s
        # Iterate until no K4s
        iterations = 0
        while time_left() > 5:
            k4s = find_k4s(adj, n)
            if not k4s:
                break
            
            # Pick a random K4 and remove a random edge from it
            u, v, x, y = k4s[0]
            edges_to_consider = [(u,v), (u,x), (u,y), (v,x), (v,y), (x,y)]
            # Pick one to remove
            ru, rv = random.choice(edges_to_consider)
            adj[ru] &= ~(1 << rv)
            adj[rv] &= ~(1 << ru)
            iterations += 1
            if iterations > 1000:
                break
        
        if time_left() < 5:
            continue
            
        # Check if K4-free
        if find_k4s(adj, n):
            continue
            
        # Repair Phase 2: Reduce Alpha
        # Check for independent set of size 6
        # We use the exact checker in is_valid, but we need the set to repair.
        # Let's use a heuristic to find a large independent set to break
        # Greedy Max Independent Set
        def get_large_independent_set(adj, n):
            remaining = list(range(n))
            indep = []
            while remaining:
                # Pick min degree
                min_deg = n
                min_v = -1
                mask = 0
                for u in remaining:
                    mask |= (1 << u)
                for v in remaining:
                    deg = (adj[v] & mask).bit_count()
                    if deg < min_deg:
                        min_deg = deg
                        min_v = v
                indep.append(min_v)
                neighbors = [u for u in remaining if (adj[min_v] >> u) & 1]
                to_remove = set(neighbors)
                to_remove.add(min_v)
                remaining = [v for v in remaining if v not in to_remove]
            return indep

        max_iter = 500
        for _ in range(max_iter):
            if time_left() < 5:
                break
            
            indep = get_large_independent_set(adj, n)
            if len(indep) < 6:
                break
                
            # We have an independent set of size >= 6
            # Try to add an edge within this set to break it
            # We want to add an edge (u,v) in indep such that it doesn't create a K4
            added = False
            random.shuffle(indep)
            for i in range(len(indep)):
                for j in range(i+1, len(indep)):
                    u = indep[i]
                    v = indep[j]
                    # Check if adding (u,v) creates a K4
                    # Common neighbors of u and v
                    common = adj[u] & adj[v]
                    # Check if any pair in common is connected
                    creates_k4 = False
                    common_list = [k for k in range(n) if (common >> k) & 1]
                    for a in range(len(common_list)):
                        for b in range(a+1, len(common_list)):
                            if (adj[common_list[a]] >> common_list[b]) & 1:
                                creates_k4 = True
                                break
                        if creates_k4:
                            break
                            
                    if not creates_k4:
                        # Add edge
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        added = True
                        break
                if added:
                    break
            
            if not added:
                # Cannot add any edge in this independent set without creating K4
                # Try a different independent set? Or give up on this n
                break

        # Final Validation
        if is_valid(adj, n):
            best_n = n
            best_adj = adj
            break

    if best_n == 0:
        # Fallback to seed
        return {'n': 17, 'edges': [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in {x * x % 17 for x in range(1, 17)}]}
        
    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
                
    return {'n': best_n, 'edges': edges}

construct()
