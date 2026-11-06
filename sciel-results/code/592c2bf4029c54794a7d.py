import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s margin for safety
    
    # We want to find a graph with n vertices, no K4, no independent set of 6.
    # Strategy: Simulated Annealing / Randomized Local Search on the adjacency matrix.
    # We start with a random graph or a Paley graph and flip edges to minimize "violations".
    # Violation score:
    #   - 1000 * (number of K4s)  [Hard constraint, but we can allow some during search if we penalize heavily]
    #   - 1000 * (number of independent 6-sets)
    #   - Actually, for n>30, counting all K4s and I6s is expensive.
    #   - Better: Track specific violations.
    #   - Even better: Use a "repair" strategy or a specific fitness function.
    
    # Let's try a different angle: 
    # The known best is 35. We aim for 32-35.
    # Start with a random graph with density p ~ 0.5 (Paley is self-complementary, density ~0.5).
    # Or start with Paley(29) and add random vertices, then optimize the whole graph.
    
    # Let's use a "Randomized Hill Climbing" on the edge set.
    # Fitness = - (100 * K4_count + 100 * I6_count)
    # We want to minimize violations.
    
    # Efficient violation counting:
    # K4 count: For each edge (u,v), count common neighbors. If common neighbors contain an edge, that's a K4.
    # Actually, just count cliques of size 4.
    # I6 count: Count independent sets of size 6.
    
    # For n=32, counting all K4s and I6s every step is too slow.
    # We need a faster heuristic or a more targeted search.
    
    # Alternative: 
    # Start with a valid graph on n=29 (Paley).
    # Try to add vertices one by one, but use a "backtracking" or "local search" on the neighbor set of the NEW vertex, 
    # AND allow occasional "flips" of edges in the existing graph to make room for the new vertex.
    
    # Let's stick to the extension idea but make it much more robust with local search on the existing graph.
    
    def generate_paley(q):
        n = q
        squares = set(x * x % q for x in range(1, q))
        adj = [set() for _ in range(n)]
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (j - i) % q in squares:
                    adj[i].add(j)
                    adj[j].add(i)
                    edges.append([i, j])
        return adj, edges, n

    def count_k4s(adj, n):
        count = 0
        for i in range(n):
            ni = adj[i]
            for j in ni:
                if j <= i: continue
                nij = adj[i] & adj[j]
                for k in nij:
                    if k <= j: continue
                    nijk = adj[i] & adj[j] & adj[k]
                    for l in nijk:
                        if l > k:
                            count += 1
        return count

    def count_i6s(adj, n):
        # Independent set of size 6 in G = Clique of size 6 in complement
        # For n=30+, this is expensive. 
        # Let's use a recursive count with pruning.
        # Or, simpler: just check if alpha >= 6. We don't need the exact count, just >0.
        # But for annealing, we need a gradient.
        # Let's just check existence for the final validation, and use a heuristic for the loop.
        
        # Heuristic: Max Independent Set size.
        # Greedy MIS is fast but not exact.
        # Let's use a faster exact check for small n, but for n=32 it might be slow.
        
        # Let's try a different fitness:
        # Fitness = (number of edges in a random sample of 6-sets that are NOT independent)
        # No, that's weak.
        
        # Let's go back to the extension strategy but with a "relaxation" phase.
        pass

    # Actually, the previous program's extension strategy failed because it got stuck.
    # Let's try a "Randomized Greedy Extension" with multiple restarts.
    
    best_n = 0
    best_edges = []
    
    # Try to build graphs of size n_target
    # Start with n=30, 31, 32...
    
    # We'll use a helper to check if a graph is valid (no K4, no I6)
    def is_valid(adj, n):
        if count_k4s(adj, n) > 0:
            return False
        
        # Check for independent set of 6
        # Use a backtracking search for independent set of size 6
        # If found, return False
        
        # Optimization: Sort vertices by degree?
        # Simple recursive search
        def has_i6(idx, current_size, current_set):
            if current_size == 6:
                return True
            if idx >= n:
                return False
            if current_size + (n - idx) < 6:
                return False
            
            # Try not taking idx
            if has_i6(idx + 1, current_size, current_set):
                return True
            
            # Try taking idx
            # Must be independent from all in current_set
            ok = True
            for v in current_set:
                if v in adj[idx]:
                    ok = False
                    break
            if ok:
                current_set.append(idx)
                if has_i6(idx + 1, current_size + 1, current_set):
                    return True
                current_set.pop()
            
            return False
        
        if has_i6(0, 0, []):
            return False
        
        return True

    # We will try to extend a Paley graph.
    # Paley(29) is valid.
    # We will try to add vertices one by one.
    # For each new vertex, we find a subset S of existing vertices to connect to.
    # Constraints on S:
    # 1. S must be triangle-free (no K4 with new vertex)
    # 2. S must intersect every independent 5-set (no I6 with new vertex)
    
    # This is the same as before, but we will use a more robust solver for S.
    # We will use a Randomized Local Search on S.
    
    def find_valid_neighbor_set(adj, n, time_limit):
        """
        Find a set S of vertices in {0..n-1} such that:
        1. S is triangle-free in G.
        2. S intersects every independent set of size 5 in G.
        """
        
        # Precompute triangles and independent 5-sets
        triangles = []
        for i in range(n):
            ni = adj[i]
            for j in ni:
                if j <= i: continue
                nij = adj[i] & adj[j]
                for k in nij:
                    if k > j:
                        triangles.append((i, j, k))
        
        # Independent 5-sets
        indep_5s = []
        def gen_i5(start, current):
            if len(current) == 5:
                indep_5s.append(tuple(current))
                return
            if len(current) + (n - start) < 5:
                return
            for v in range(start, n):
                ok = True
                for u in current:
                    if v in adj[u]:
                        ok = False
                        break
                if ok:
                    gen_i5(v + 1, current + [v])
        
        gen_i5(0, [])
        
        # Now we need to find S.
        # Start with S = all vertices.
        # Remove vertices to break triangles.
        # Add vertices to hit independent 5-sets.
        
        # This is a hitting set problem with forbidden triples.
        # We can use a simple greedy + local search.
        
        # Greedy:
        # 1. Start with S = empty? No, S needs to be large enough to hit all I5s.
        #    Actually, S can be small.
        # 2. Start with S = random subset?
        
        # Let's try a "Start with all, remove to fix triangles" approach again, but with backtracking if it fails.
        
        # Actually, a better approach for S:
        # S must be a "Transversal" of the hypergraph of I5s, and "Independent" in the hypergraph of triangles.
        
        # Let's try random restarts of a local search on S.
        
        def score_s(S):
            # Score = (number of triangles fully in S) + (number of I5s disjoint from S)
            # We want score 0.
            tri_count = 0
            for t in triangles:
                if t[0] in S and t[1] in S and t[2] in S:
                    tri_count += 1
            
            i5_miss = 0
            for i5 in indep_5s:
                if not any(v in S for v in i5):
                    i5_miss += 1
            
            return tri_count + i5_miss
        
        best_s = None
        best_score = float('inf')
        
        # Random restarts
        for _ in range(200):
            if time.time() > time_limit:
                break
            
            # Random initial S
            # Bias towards larger sets? Or smaller?
            # Let's start with S = all vertices, then remove random ones to break triangles.
            S = set(range(n))
            
            # Greedy removal to break triangles
            # Shuffle triangles
            tris = triangles[:]
            random.shuffle(tris)
            for t in tris:
                if t[0] in S and t[1] in S and t[2] in S:
                    # Remove one
                    v = random.choice(t)
                    S.remove(v)
            
            # Now local search: flip vertices in/out of S to reduce score
            current_s = score_s(S)
            if current_s == 0:
                return S
            
            # Hill climbing
            for _ in range(100):
                if time.time() > time_limit:
                    break
                
                # Pick a random vertex to flip
                v = random.randint(0, n - 1)
                
                if v in S:
                    S.remove(v)
                else:
                    S.add(v)
                
                new_s = score_s(S)
                
                if new_s < current_s:
                    current_s = new_s
                    if current_s == 0:
                        return S
                else:
                    # Revert
                    if v in S:
                        S.remove(v)
                    else:
                        S.add(v)
            
            if current_s < best_score:
                best_score = current_s
                best_s = set(S)
        
        return best_s if best_score == 0 else None

    # Main loop: Try to extend
    adj, edges, n = generate_paley(29)
    
    # Verify base
    if not is_valid(adj, n):
        # Fallback to smaller if Paley(29) fails (shouldn't)
        adj, edges, n = generate_paley(17)
    
    best_n = n
    best_edges = edges[:]
    
    # Try to extend up to 35
    for target_n in range(n + 1, 36):
        if time.time() > deadline - 5.0:
            break
        
        # Try to find a valid neighbor set for the new vertex
        # We allow multiple attempts with different random seeds/strategies
        found = False
        attempts = 0
        max_attempts = 50
        
        while not found and attempts < max_attempts:
            attempts += 1
            time_limit = time.time() + (deadline - 5.0 - time.time()) / (max_attempts - attempts)
            S = find_valid_neighbor_set(adj, n, time_limit)
            if S is not None:
                # Add vertex
                new_idx = n
                adj.append(set(S))
                for v in S:
                    adj[v].add(new_idx)
                    edges.append([new_idx, v])
                n += 1
                best_n = n
                best_edges = edges[:]
                found = True
                break
        
        if not found:
            break
    
    return {'n': best_n, 'edges': best_edges}

