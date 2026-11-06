import time
import random
import sys

def construct():
    start_time = time.time()
    time_limit = 115.0  # Leave 5s buffer for evaluation overhead
    
    # We aim for n=35. The seed provides a valid n=17.
    # If we can't find 35 quickly, we might fallback, but 35 is the record.
    # We will try to find a valid graph on n=35.
    
    def is_valid_graph(n, adj_bitmasks):
        """
        Checks if graph is K4-free and has no independent set of size 6.
        adj_bitmasks[i] is an int where bit j is 1 if i~j.
        """
        # 1. Check for K4
        # A K4 exists if there is an edge (u,v) such that the common neighbors 
        # contain an edge.
        for u in range(n):
            for v in range(u + 1, n):
                if (adj_bitmasks[u] >> v) & 1:
                    common = adj_bitmasks[u] & adj_bitmasks[v]
                    # common contains vertices w such that w~u and w~v
                    # We need to check if any pair in common is an edge.
                    # Iterate through set bits in common
                    c = common
                    while c:
                        # Get lowest set bit
                        lsb = c & -c
                        k = lsb.bit_length() - 1
                        c ^= lsb
                        # Check if k is connected to any other in common?
                        # Actually, if common has an edge, we have K4.
                        # Let's just check if (common & adj_bitmasks[k]) has any bits.
                        if common & adj_bitmasks[k]:
                            return False, "K4"
        
        # 2. Check for Independent Set of size 6
        # An independent set in G is a clique in the complement G'.
        # Complement adjacency: comp_adj[i] = full_mask ^ adj_bitmasks[i] ^ (1 << i)
        # We need to find a clique of size 6 in G'.
        
        def has_clique_k(adj_complement, k):
            """
            Backtracking check for clique of size k in graph represented by adj_bitmasks.
            """
            if k == 0:
                return True
            if k == 1:
                return any(adj_complement[i] != 0 or True for i in range(len(adj_complement))) # Always true if k=1 and n>=1
            
            # Recursive helper
            def backtrack(candidates, size):
                if size == 0:
                    return True
                if len(candidates) < size:
                    return False
                
                # Heuristic: pick vertex with max degree in candidates? 
                # For simplicity and speed with bitmasks, we can just iterate.
                # Better: Order candidates by degree in the induced subgraph to prune faster.
                # But for small k=6, simple branching is often fast enough.
                
                # Try including the first candidate
                v = candidates[0]
                # New candidates must be neighbors of v in the complement graph
                # AND be in the remaining candidates
                new_candidates = [u for u in candidates[1:] if (adj_complement[v] >> u) & 1]
                
                if backtrack(new_candidates, size - 1):
                    return True
                
                # Try excluding v
                if backtrack(candidates[1:], size):
                    return True
                
                return False
            
            return backtrack(list(range(len(adj_complement))), k)

        # Build complement adjacency bitmasks
        # For n=35, full_mask = (1 << 35) - 1
        full_mask = (1 << n) - 1
        comp_adj = []
        for i in range(n):
            # Non-neighbors are bits in full_mask that are 0 in adj[i]
            # Exclude self-loop
            non_neighbors = full_mask ^ adj_bitmasks[i]
            non_neighbors &= ~(1 << i)
            comp_adj.append(non_neighbors)
            
        if has_clique_k(comp_adj, 6):
            return False, "Ind6"
            
        return True, ""

    def get_k4_edges(n, adj_bitmasks):
        """Returns one K4 if found as a set of edges, else None."""
        for u in range(n):
            for v in range(u + 1, n):
                if (adj_bitmasks[u] >> v) & 1:
                    common = adj_bitmasks[u] & adj_bitmasks[v]
                    c = common
                    # We need two vertices k, l in common such that k~l
                    # Just pick the first two in common and check? 
                    # No, we need to find ANY edge in common.
                    while c:
                        lsb = c & -c
                        k = lsb.bit_length() - 1
                        c ^= lsb
                        # Check if k is connected to any other vertex in the original 'common' set
                        # Note: 'common' still contains k. 
                        # We check if (common & adj_bitmasks[k]) is non-zero.
                        # Since k is in common, adj_bitmasks[k] might have bits in common.
                        # If it has, say l, then (u,v,k,l) is a K4.
                        if common & adj_bitmasks[k]:
                            # Found K4: u, v, k, and some l in (common & adj_bitmasks[k])
                            # We just need to remove one edge. Let's return the edge (u, v) or (k, l).
                            # Returning (u,v) is safe.
                            return [(u, v), (u, k), (u, l), (v, k), (v, l), (k, l)]
        return None

    def get_ind6_vertices(n, adj_bitmasks):
        """Returns one independent set of size 6 as a list of vertices, else None."""
        full_mask = (1 << n) - 1
        comp_adj = []
        for i in range(n):
            non_neighbors = full_mask ^ adj_bitmasks[i]
            non_neighbors &= ~(1 << i)
            comp_adj.append(non_neighbors)
            
        found_clique = []
        
        def backtrack(candidates, size):
            if size == 0:
                return True
            if len(candidates) < size:
                return False
            
            v = candidates[0]
            new_candidates = [u for u in candidates[1:] if (comp_adj[v] >> u) & 1]
            
            found_clique.append(v)
            if backtrack(new_candidates, size - 1):
                return True
            found_clique.pop()
            
            if backtrack(candidates[1:], size):
                return True
                
            return False
            
        if backtrack(list(range(n)), 6):
            return found_clique
        return None

    # Main Search Logic
    best_n = 17
    best_edges_list = []
    # Seed edges from the current program (n=17, quadratic residues mod 17)
    n_seed = 17
    squares = {x * x % 17 for x in range(1, 17)}
    seed_edges = set((i, j) for i in range(n_seed) for j in range(i + 1, n_seed) if (j - i) % 17 in squares)
    best_edges_list = [[u, v] for u, v in seed_edges]
    
    # Try to find n=35
    target_n = 35
    random.seed(42)
    
    for attempt in range(50): # Try a few random restarts if local search gets stuck
        if time.time() - start_time > time_limit:
            break
            
        n = target_n
        adj = [0] * n
        edges_set = set()
        
        # Initialize random graph
        p = 0.5
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    edges_set.add((i, j))
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Local Search
        steps = 0
        max_steps = 1000
        stuck_counter = 0
        
        while steps < max_steps and time.time() - start_time < time_limit:
            steps += 1
            
            # Check K4
            k4 = get_k4_edges(n, adj)
            if k4:
                # Remove a random edge from the K4
                u, v = random.choice(k4)
                edges_set.discard((min(u, v), max(u, v)))
                # Update adj
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                stuck_counter = 0
                continue
                
            # Check Ind 6
            ind6 = get_ind6_vertices(n, adj)
            if ind6:
                # Add a random edge between two vertices in the independent set
                u, v = random.sample(ind6, 2)
                if (min(u, v), max(u, v)) not in edges_set:
                    edges_set.add((min(u, v), max(u, v)))
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    stuck_counter = 0
                    continue
                else:
                    # This shouldn't happen if ind6 is truly independent, but just in case
                    stuck_counter += 1
                    if stuck_counter > 10:
                        break # Restart
            else:
                # No K4 and No Ind6. We found a valid graph!
                # Verify strictly one last time? 
                # The checks above are exact.
                best_n = n
                best_edges_list = [[u, v] for u, v in edges_set]
                return {'n': best_n, 'edges': best_edges_list}
                
        # If loop finishes without finding valid graph, try next restart with slightly different p or seed
        # We can vary p slightly
        p = 0.5 + (attempt % 5) * 0.01 - 0.02

    # If we didn't find 35, return the best we have (seed n=17)
    # It is highly likely we find 35.
    return {'n': best_n, 'edges': best_edges_list}
