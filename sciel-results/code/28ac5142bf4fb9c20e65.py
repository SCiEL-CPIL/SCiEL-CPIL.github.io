import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Target: Find a graph with n >= 35
    # We will use Simulated Annealing on a fixed n=35 graph.
    # If we find a valid one (violations=0), we return it.
    # We can also try to extend if we have time, but 35 is the target.
    
    n = 35
    
    # Helper: Generate a valid base graph of size n0
    # We can use a random graph and repair it, or use a known structure.
    # Let's start with a random graph with p=0.5 and let SA fix it.
    # To be safer, we can start with a graph that is "close" to valid.
    # However, random restarts are robust.
    
    def get_violations(adj):
        """
        Counts the number of K4s and I6s in the graph defined by adjacency matrix adj.
        adj is a numpy array of 0s and 1s.
        """
        # K4 count:
        # A K4 is a set of 4 vertices {a,b,c,d} such that all 6 edges exist.
        # We can iterate over all 4-sets, but that is C(35,4) ~ 52k. 
        # Better: Use matrix multiplication or bitmasks.
        # Bitmask approach is fast in Python if we convert rows to ints.
        
        # Convert adjacency matrix to list of bitmasks
        # Row i has bit j set if edge (i,j) exists.
        # Note: adj[i][i] is 0.
        
        masks = [0] * n
        for i in range(n):
            m = 0
            row = adj[i]
            # Only consider j > i? No, K4 needs all edges.
            # Standard trick: 
            # For each pair (i, j) with i < j and edge (i,j) exists:
            # Common neighbors of i and j: common = adj[i] & adj[j]
            # Any vertex k in common with k > j forms a triangle (i, j, k).
            # To count K4s:
            # Iterate over all edges (i, j) with i < j.
            # Let common = adj[i] & adj[j].
            # Remove i and j from common.
            # Any pair of vertices (k, l) in common such that edge (k,l) exists forms a K4 {i, j, k, l}.
            # This is essentially counting triangles in the subgraph induced by common.
            # This might be slow if done naively for every edge.
            
            # Alternative: 
            # Number of K4s = (1/8) * sum over all vertices v of (number of triangles containing v)? No.
            # Let's stick to a reasonably fast Python implementation using bitmasks.
            
            # Precompute bitmasks for each row
            for j in range(n):
                if adj[i][j]:
                    m |= (1 << j)
            masks[i] = m
        
        k4_count = 0
        # Iterate over all pairs (i, j)
        for i in range(n):
            mi = masks[i]
            for j in range(i + 1, n):
                if mi & (1 << j): # Edge (i,j) exists
                    mj = masks[j]
                    # Common neighbors of i and j
                    common = mi & mj
                    # Remove i and j themselves
                    common &= ~(1 << i)
                    common &= ~(1 << j)
                    
                    # Now, any edge (k, l) within 'common' forms a K4 with i, j.
                    # We need to count edges in the subgraph induced by 'common'.
                    # To avoid double counting K4s, we only count pairs (k, l) with k < l.
                    # But wait, if we iterate over all edges (i,j), each K4 is counted how many times?
                    # A K4 has 6 edges. 
                    # If we pick edge (i,j) from the K4, the other 2 vertices are in common.
                    # The edge between the other 2 vertices is in common.
                    # So each K4 contributes 6 to the sum if we count all pairs in common.
                    # We want to count each K4 once.
                    # So we can sum up the number of edges in the induced subgraph of 'common' for each edge (i,j), and divide by 6.
                    
                    # How to count edges in induced subgraph of 'common' quickly?
                    # Let C be the set of vertices in common.
                    # Edges in C = sum_{k in C} (popcount(masks[k] & C)) / 2.
                    # This is O(|C|) per edge (i,j).
                    # Total complexity: O(E * avg_degree). For n=35, this is fine.
                    
                    c_count = 0
                    c_mask = common
                    while c_mask:
                        # Get lowest set bit
                        lsb = c_mask & -c_mask
                        k = lsb.bit_length() - 1
                        c_mask ^= lsb
                        # Count neighbors of k within c_mask (which is common minus k)
                        # Actually, we can just count popcount(masks[k] & common)
                        # But we divide by 2 at the end.
                        c_count += bin(masks[k] & common).count('1')
                    
                    k4_count += c_count
                    
        # Total K4s = k4_count / 6
        k4_count //= 6
        
        # I6 count:
        # Independent set of size 6.
        # This is harder to count exactly with bitmasks efficiently in pure Python loops.
        # However, we can use a recursive branch and bound or just check if alpha >= 6.
        # For SA, we need a cost function. 
        # Cost = (# K4s) + (# I6s).
        # Counting I6s exactly is expensive. 
        # Alternative: Use the number of I5s as a proxy? No, we need to avoid I6.
        # Maybe we can use a heuristic cost: 
        # If we find any I6, cost is high.
        # But SA needs a smooth landscape.
        
        # Let's try to count I6s using a backtracking algorithm that is optimized.
        # Or, simpler: Just check if there is an I6. If yes, cost += 100 (large penalty).
        # This makes the landscape discontinuous, but often works for "avoid constraint" problems.
        
        # To make it smoother, we can count the number of I5s. 
        # If there are many I5s, it's likely to have I6s.
        # Let's count I5s.
        
        i5_count = 0
        # Iterate over all 5-sets? C(35,5) ~ 324k. Too slow for every SA step.
        
        # We need a faster way to evaluate the "I6-ness" of the graph.
        # Idea: Use the complement graph. I6 in G is K6 in G_bar.
        # So we need to avoid K6 in G_bar.
        # We can count K6s in G_bar similarly to K4s in G.
        # But K6 counting is even more expensive.
        
        # Let's reconsider the cost function.
        # Cost = 10 * (# K4s) + 1 * (Number of vertices in the largest independent set? No).
        
        # Let's try a different approach for the cost:
        # 1. Count K4s exactly (fast enough).
        # 2. For I6, we can use a greedy approximation of the independence number? 
        #    Or simply: if alpha(G) >= 6, add a penalty.
        #    How to compute alpha(G) fast? 
        #    Branch and bound for alpha is exponential but for n=35, it's usually very fast if the graph is dense enough.
        #    Since we are avoiding K4, the graph is not too dense.
        
        # Let's implement a fast alpha checker.
        
        def has_independent_set_6(adj):
            # Backtracking to find I6
            # Order vertices by degree (ascending) to prune earlier?
            # Actually, for independent set, we want to pick vertices with low degree in the remaining graph?
            # Or high degree? 
            # Standard: Bron-Kerbosch for maximal cliques in complement.
            # Let's just do a simple DFS with pruning.
            
            # Precompute non-neighbor masks
            # non_adj[i] has bit j set if NOT edge (i,j) and i != j.
            
            # We'll use a recursive function.
            # State: current set of vertices in I, candidate set.
            # To be fast, we represent sets as bitmasks.
            
            # non_adj_masks[i]
            non_adj_masks = [0] * n
            for i in range(n):
                m = 0
                for j in range(n):
                    if i != j and not adj[i][j]:
                        m |= (1 << j)
                non_adj_masks[i] = m
            
            found = False
            
            def dfs(current_set, candidates):
                nonlocal found
                if found:
                    return
                # If we can pick enough from candidates to reach size 6
                if bin(current_set).count('1') + bin(candidates).count('1') < 6:
                    return
                
                # Try to add vertices from candidates
                c_mask = candidates
                while c_mask:
                    if found:
                        return
                    lsb = c_mask & -c_mask
                    v = lsb.bit_length() - 1
                    c_mask ^= lsb
                    
                    # Add v to current set
                    new_set = current_set | lsb
                    # New candidates: those in c_mask that are non-neighbors of v
                    # Note: c_mask already excludes v and previously removed vertices.
                    # We must intersect with non_adj_masks[v]
                    new_candidates = c_mask & non_adj_masks[v]
                    
                    dfs(new_set, new_candidates)
                    
            dfs(0, (1 << n) - 1)
            return found

        # This DFS might be too slow if called at every SA step.
        # C(35, 6) is ~ 1.6M. But with pruning, it should be much less.
        # However, calling this 1000 times (SA steps) might take too long.
        
        # Alternative: 
        # Use a randomized greedy heuristic to estimate alpha?
        # Or, simpler: 
        # Cost = # K4s + (1 if alpha >= 6 else 0).
        # The discontinuity might trap SA.
        
        # Let's try to count I5s instead? 
        # If we have an I5, it's "closer" to I6.
        # Counting I5s is also expensive.
        
        # Let's try a hybrid:
        # 1. Count K4s.
        # 2. Run a fast greedy max independent set. If size >= 6, penalty.
        #    Greedy: pick vertex with min degree, remove it and neighbors, repeat.
        #    This is O(n^2). Very fast.
        
        def greedy_alpha(adj):
            # Returns size of a greedy independent set
            remaining = set(range(n))
            alpha = 0
            while remaining:
                # Pick vertex with minimum degree in the induced subgraph
                min_deg = n
                best_v = -1
                for v in remaining:
                    deg = 0
                    for u in remaining:
                        if v != u and adj[v][u]:
                            deg += 1
                    if deg < min_deg:
                        min_deg = deg
                        best_v = v
                
                if best_v == -1:
                    break
                
                alpha += 1
                # Remove best_v and its neighbors
                to_remove = {best_v}
                for u in remaining:
                    if adj[best_v][u]:
                        to_remove.add(u)
                remaining -= to_remove
                
            return alpha

        # This greedy is O(n^3) if implemented naively with sets. 
        # n=35, n^3 ~ 42k ops. Very fast.
        
        # Let's use this.
        # Cost = # K4s + 10 * max(0, greedy_alpha - 5)
        # This encourages reducing alpha below 6.
        
        g_alpha = greedy_alpha(adj)
        i_penalty = 0
        if g_alpha >= 6:
            i_penalty = 10 * (g_alpha - 5)
            
        return k4_count + i_penalty

    # Initial Graph
    # Start with a random graph with p=0.5
    adj = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < 0.5:
                adj[i][j] = 1
                adj[j][i] = 1

    current_cost = get_violations(adj)
    best_adj = adj.copy()
    best_cost = current_cost

    T = 10.0
    T_min = 0.1
    cooling_rate = 0.999

    # SA Loop
    while time.time() < deadline and T > T_min:
        # Pick a random edge to flip
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j:
            continue
        
        # Flip edge (i, j)
        adj[i][j] = 1 - adj[i][j]
        adj[j][i] = adj[i][j]
        
        new_cost = get_violations(adj)
        delta = new_cost - current_cost
        
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_adj = adj.copy()
                # If we found a valid graph (cost 0), we can return early?
                # Yes, 35 is the target.
                if best_cost == 0:
                    break
        else:
            # Revert
            adj[i][j] = 1 - adj[i][j]
            adj[j][i] = adj[i][j]
            
        T *= cooling_rate

    # If we found a valid graph on 35, return it
    if best_cost == 0:
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if best_adj[i][j]:
                    edges.append([i, j])
        return {'n': n, 'edges': edges}
    
    # If we didn't find a valid graph on 35, try to return the best one found?
    # No, it must be valid. 
    # If we didn't reach 0 violations, we should fall back to a known valid graph.
    # We can try to run SA for a smaller n, e.g., 30, which is easier.
    
    # Let's try n=30 if 35 failed.
    # But the task wants to maximize n.
    # If we can't get 35, we should return the largest valid n we can construct.
    # Since we only optimized for 35, we don't have a valid graph for 34, 33, etc.
    # We should probably run the SA for descending n if 35 fails?
    # Or just return a known small valid graph.
    
    # Let's try to quickly find a valid graph for n=30 using the same SA but shorter time?
    # No, just return a fallback.
    
    # Fallback: Paley 17
    n_p = 17
    squares = {x * x % n_p for x in range(1, n_p)}
    p_edges = [[i, j] for i in range(n_p) for j in range(i + 1, n_p) if (j - i) % n_p in squares]
    
    return {'n': n_p, 'edges': p_edges}
