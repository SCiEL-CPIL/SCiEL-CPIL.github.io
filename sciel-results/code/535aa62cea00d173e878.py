import time
import random
import itertools
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s margin for safety
    
    # Start with Paley graph on 29 vertices
    q = 29
    n = q
    squares = set(x * x % q for x in range(1, q))
    
    # Build adjacency as a set of frozensets for fast lookup
    adj = [set() for _ in range(n)]
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % q in squares:
                adj[i].add(j)
                adj[j].add(i)
                edges.append([i, j])
    
    def has_k4(adj, n):
        """Check if graph contains K4. Returns True if K4 exists."""
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
                            return True
        return False
    
    def has_independent_set_6(adj, n):
        """Check if graph contains independent set of size 6. Returns True if exists."""
        # Use complement graph: independent set in G = clique in complement
        # But simpler: just check all 6-sets for n=29 (475k combos, each O(15) check)
        # Actually for speed, use a smarter approach: 
        # Find maximal independent sets and check if any has size >= 6
        
        # Greedy approach with backtracking for small n
        # For n=29, we can do a bounded search
        
        # Better: use the fact that we need to check if alpha(G) >= 6
        # We'll use a recursive branch and bound
        
        visited = [False] * n
        best = [0]
        
        def branch_and_bound(idx, current_set):
            if len(current_set) >= 6:
                return True
            if idx >= n:
                return False
            
            # Pruning: even if we take all remaining, can we reach 6?
            remaining = n - idx
            if len(current_set) + remaining < 6:
                return False
            
            # Option 1: don't include idx
            if branch_and_bound(idx + 1, current_set):
                return True
            
            # Option 2: include idx if it's independent from all in current_set
            can_include = True
            for v in current_set:
                if v in adj[idx]:
                    can_include = False
                    break
            
            if can_include:
                if branch_and_bound(idx + 1, current_set + [idx]):
                    return True
            
            return False
        
        return branch_and_bound(0, [])
    
    # Verify initial graph is valid
    if has_k4(adj, n) or has_independent_set_6(adj, n):
        # Shouldn't happen for Paley(29), but fallback
        # Try smaller Paley
        q = 17
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
    
    def is_valid(adj, n):
        return not has_k4(adj, n) and not has_independent_set_6(adj, n)
    
    # Simulated annealing: try to improve the graph
    # Fitness: we want to maintain validity. Since n is fixed, we can't increase n easily.
    # Alternative strategy: Try to extend by adding vertices using smarter methods.
    
    # Actually, let's try a different approach: 
    # Use the time to try multiple random extensions and keep the best.
    
    best_n = n
    best_edges = edges[:]
    
    # Try to extend from n to n+1, n+2, etc. using random neighbor sets
    # But do it more intelligently: use the structure of the graph
    
    # Precompute triangles and independent 5-sets for the base graph
    def get_triangles(adj, n):
        tris = []
        for i in range(n):
            for j in adj[i]:
                if j <= i: continue
                common = adj[i] & adj[j]
                for k in common:
                    if k > j:
                        tris.append((i, j, k))
        return tris
    
    def get_indep_5(adj, n):
        """Get all independent sets of size 5."""
        result = []
        # Use recursive generation
        def gen(start, current):
            if len(current) == 5:
                result.append(tuple(current))
                return
            remaining = n - start
            if len(current) + remaining < 5:
                return
            for v in range(start, n):
                # Check if v is independent from all in current
                ok = True
                for u in current:
                    if v in adj[u]:
                        ok = False
                        break
                if ok:
                    gen(v + 1, current + [v])
        
        gen(0, [])
        return result
    
    # Try extending step by step
    current_adj = [set(s) for s in adj]
    current_edges = edges[:]
    current_n = n
    
    for target_n in range(current_n + 1, 36):
        if time.time() > deadline - 10.0:
            break
        
        # Get constraints for adding new vertex
        triangles = get_triangles(current_adj, current_n)
        indep_5 = get_indep_5(current_adj, current_n)
        
        # We need a subset S of {0..current_n-1} such that:
        # 1. No triangle in current_adj[S] (i.e., S contains no triangle vertices)
        # 2. S intersects every independent 5-set
        
        # This is a hitting set problem with triangle-free constraint.
        # Use randomized local search to find such S.
        
        found = False
        attempts = 0
        max_attempts = 500
        
        while not found and attempts < max_attempts and time.time() < deadline - 10.0:
            attempts += 1
            
            # Start with S = all vertices, then remove to break triangles
            # Then ensure hitting set property
            S = set(range(current_n))
            
            # Break triangles: for each triangle, remove at least one vertex
            # Greedy: for each triangle, pick a random vertex to remove
            for tri in triangles:
                if tri[0] in S and tri[1] in S and tri[2] in S:
                    # Remove one of them
                    v = random.choice(tri)
                    S.remove(v)
            
            # Now check hitting set: S must intersect every indep_5
            # If not, add vertices to fix
            valid_hit = True
            for i5 in indep_5:
                if not any(v in S for v in i5):
                    valid_hit = False
                    # Need to add at least one vertex from i5 to S
                    # But adding might create triangles
                    # Try adding a random one
                    candidates = [v for v in i5 if v not in S]
                    if candidates:
                        # Check if adding any of them creates a triangle
                        for v in candidates:
                            # Check if v with its neighbors in S forms a triangle
                            creates_tri = False
                            for u in current_adj[v]:
                                if u in S:
                                    for w in current_adj[u]:
                                        if w in S and w != v:
                                            # Check if v-u-w is a triangle
                                            if v in current_adj[w]:
                                                creates_tri = True
                                                break
                            if not creates_tri:
                                S.add(v)
                                break
                        else:
                            # All candidates create triangles, so this is hard
                            valid_hit = False
                            break
                    else:
                        valid_hit = False
                        break
            
            if valid_hit:
                # Verify S is triangle-free
                tri_free = True
                for tri in triangles:
                    if tri[0] in S and tri[1] in S and tri[2] in S:
                        tri_free = False
                        break
                
                if tri_free:
                    found = True
        
        if found:
            # Add new vertex with neighbor set S
            new_idx = current_n
            current_adj.append(set(S))
            for v in S:
                current_adj[v].add(new_idx)
            for v in S:
                current_edges.append([new_idx, v])
            current_n += 1
            best_n = current_n
            best_edges = current_edges[:]
        else:
            break
    
    return {'n': best_n, 'edges': best_edges}
