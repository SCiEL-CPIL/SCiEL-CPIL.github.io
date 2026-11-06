import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s buffer for the 120s limit (or 150s wall clock if applicable, but 115 is safe)

    def check_c4_free(adj, n):
        # adj is list of int bitmasks
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] & adj[v]) & ((adj[u] & adj[v]) - 1):
                    return False
        return True

    def has_independent_set(adj, n, k):
        # Returns True if there is an independent set of size k.
        # Independent set in G is a clique in complement.
        # We use a simple branch and bound on the complement.
        # For speed, we represent complement adjacency as bitmasks.
        
        # Build complement adjacency bitmasks
        comp_adj = [0] * n
        for u in range(n):
            # All vertices except u and its neighbors in G
            all_mask = (1 << n) - 1
            comp_adj[u] = all_mask & ~(adj[u] | (1 << u))
            
        # Sort vertices by degree in complement descending to find cliques faster?
        # Or just use a simple recursive search with pruning.
        
        # To check if alpha >= k, we look for a clique of size k in comp.
        # We can prune if remaining candidates < k - current_size.
        
        # Optimization: If any vertex has degree < k-1 in comp, it can't be in a clique of size k.
        # But we need a clique of size k anywhere.
        
        # Let's use a simple DFS with a limit on nodes visited to avoid timeouts.
        # If it times out, we assume "no" (risky) or "yes" (conservative for validity)?
        # If we assume "no" and there IS an independent set, we return an INVALID graph.
        # We must be SURE there is NO independent set.
        # So if the search is inconclusive, we must treat it as "has independent set" (invalid).
        
        visited_count = 0
        max_nodes = 2000000 # 2M nodes cap for this check
        
        # Greedy coloring bound for max clique is better, but let's try a simpler prune first.
        # We'll use a set of available vertices.
        
        def dfs(available, current_size):
            nonlocal visited_count
            visited_count += 1
            if visited_count > max_nodes:
                return True # Assume failure (has independent set) to be safe
        
            if current_size >= k:
                return True
            
            if not available:
                return False
            
            # Prune: if number of available vertices + current_size < k, fail
            if current_size + bin(available).count('1') < k:
                return False
            
            # Pick a vertex with max degree in available to branch
            # Or just pick the first one? Max degree usually helps find cliques faster.
            # Let's pick v with max degree in available.
            best_v = -1
            max_deg = -1
            temp = available
            while temp:
                v = (temp & -temp).bit_length() - 1
                temp &= temp - 1
                deg = bin(comp_adj[v] & available).count('1')
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
            
            if best_v == -1:
                return False
            
            v = best_v
            # Branch 1: Include v
            # New available: neighbors of v in comp that are in available
            new_avail = available & comp_adj[v]
            if dfs(new_avail, current_size + 1):
                return True
            
            # Branch 2: Exclude v
            available &= ~(1 << v)
            return dfs(available, current_size)
            
        return dfs((1 << n) - 1, 0)

    def generate_random_c4_free(n, seed):
        random.seed(seed)
        adj = [0] * n
        edges = []
        
        # Generate all possible edges and shuffle
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            # Check if adding (u,v) creates a C4
            # A C4 is formed if there is a w such that w is neighbor of u and v, 
            # and there is another common neighbor? No.
            # C4: u-w-x-v-u. 
            # Adding (u,v) creates a C4 if there exists w, x such that 
            # w is neighbor of u, x is neighbor of v, and w is neighbor of x? 
            # Actually, the condition for C4-free is: for any u, v, |N(u) \cap N(v)| <= 1.
            # If we add (u,v), the new common neighbors of u and v are N(u) \cap N(v).
            # If |N(u) \cap N(v)| >= 1, then adding (u,v) might not create a C4 immediately involving u,v as opposite vertices?
            # Wait. The C4 condition is on the final graph.
            # If we add (u,v), we need to ensure that for all pairs (a,b) in the graph, |N(a) \cap N(b)| <= 1.
            # Adding (u,v) changes N(u) and N(v).
            # It increases |N(u) \cap N(x)| for x in N(v)?
            # Let's check the standard incremental C4-free construction:
            # To add (u,v), we require that N(u) \cap N(v) is empty? No, that's for triangle-free.
            # For C4-free: If we add (u,v), we create a C4 if there exists a path of length 3 between u and v in the existing graph.
            # i.e., if there is w, x such that u-w-x-v.
            # This is equivalent to checking if there is a common neighbor of u and v? No.
            # If u-w and w-x and x-v exist.
            # Let's just check the condition: For all w, |N(u) \cap N(w)| <= 1 and |N(v) \cap N(w)| <= 1.
            # And specifically, the new edge (u,v) must not form a C4.
            # A C4 involving (u,v) would be u-v-x-w-u.
            # This requires v-x, x-w, w-u.
            # So we need to check if there is a path of length 3 between u and v.
            # This is slower.
            
            # Simpler check for incremental addition:
            # The condition "No two vertices have two common neighbours" must hold for ALL pairs.
            # Adding (u,v) affects pairs involving u or v.
            # 1. Pair (u, v): Common neighbors are N(u) \cap N(v). If size > 1, invalid.
            # 2. Pair (u, w) for w in N(v): New common neighbor is v. 
            #    Old common neighbors were (N(u) \cap N(w)) \setminus {v} (since v wasn't in N(u) yet).
            #    So we need |(N(u) \cap N(w)) \setminus {v}| <= 0? No.
            #    Total common neighbors of u and w after adding (u,v) will be (N(u) \cap N(w)) \cup {v} (if v in N(w)).
            #    Since v is in N(w) (as w in N(v) implies v in N(w)), v becomes a common neighbor.
            #    So we need |(N(u) \cap N(w))| <= 0? i.e., N(u) \cap N(w) must be empty?
            #    If N(u) \cap N(w) is not empty, say contains z, then u-z-w-v-u is a C4?
            #    Edges: u-z, z-w, w-v, v-u. Yes.
            #    So, for any w in N(v), N(u) \cap N(w) must be empty.
            #    Similarly, for any w in N(u), N(v) \cap N(w) must be empty.
            
            safe = True
            # Check condition 1: N(u) \cap N(v) must be empty?
            # If N(u) \cap N(v) is not empty, let z be common.
            # Then u-z-v-u is a triangle, not a C4.
            # But does it violate the "2 common neighbors" rule for some pair?
            # The rule is: for ANY pair a,b, |N(a) \cap N(b)| <= 1.
            # If we add (u,v), and z is in N(u) \cap N(v), then z is a common neighbor of u and v.
            # If there is another common neighbor z', then |N(u) \cap N(v)| >= 2, which is invalid.
            # So N(u) \cap N(v) must have size <= 1.
            # But we also need to check other pairs.
            
            # Let's stick to the path of length 3 check which is robust.
            # Path u-w-x-v.
            # w must be in N(u). x must be in N(v). And w-x must be an edge.
            # So, for any w in N(u), N(w) \cap N(v) must be empty?
            # If w in N(u) and x in N(v) and w-x is edge, then u-w-x-v is a path of length 3.
            # Adding u-v closes the C4.
            # So, for all w in N(u), N(w) \cap N(v) must be empty.
            # And for all x in N(v), N(x) \cap N(u) must be empty (symmetric).
            
            # Let's check:
            # For each w in N(u):
            #   if N(w) & N(v) != 0: return False
            # This covers u-w-x-v.
            
            nu = adj[u]
            nv = adj[v]
            
            # Iterate neighbors of u
            temp = nu
            while temp:
                w = (temp & -temp).bit_length() - 1
                temp &= temp - 1
                if adj[w] & nv:
                    safe = False
                    break
            
            if not safe:
                continue
                
            # Add edge
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edges.append([u, v])
            
        return adj, edges

    # Try to find a valid graph
    # We know n=39 is possible.
    # We will try n=43 down to 39.
    
    for n in range(43, 38, -1):
        if time.time() > deadline:
            break
            
        # Try a few seeds
        # For larger n, random dense graphs might be too hard to make C4-free and alpha<11.
        # C4-free graphs are sparse. Max edges ~ n^2/4.
        # Alpha bound is the tricky part.
        
        found = False
        for seed in range(50):
            if time.time() > deadline:
                break
                
            adj, edges = generate_random_c4_free(n, seed)
            
            # Verify C4-free (should be true by construction, but double check for safety)
            if not check_c4_free(adj, n):
                continue
                
            # Check independent set
            # If has_independent_set returns False, then alpha < 11.
            if not has_independent_set(adj, n, 11):
                return {'n': n, 'edges': edges}
                
        # If we didn't find one for this n, try smaller n
        # We don't keep the best invalid graph because we must return a VALID one.
        
    # Fallback: Return a known valid graph.
    # The seed program likely returned n=30 or similar.
    # Let's construct a simple C4-free graph with small alpha.
    # Disjoint union of triangles?
    # n=30, 10 triangles.
    # C4-free? Yes, components are K3. No C4 in K3.
    # Alpha? In each K3, alpha=1. Total alpha = 10.
    # 10 < 11. So this is valid.
    
    edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        edges += [[a, b], [a, c], [b, c]]
        
    return {'n': 30, 'edges': edges}
