"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _s2_masks(n, adj):
    # s2[i] = bitmask of vertices at distance exactly 2 from i (via one edge to a neighbor)
    s2 = [0] * n
    for i in range(n):
        m = 0
        a = adj[i]
        while a:
            lsb = a & -a
            j = lsb.bit_length() - 1
            m |= adj[j]
            a ^= lsb
        m &= ~adj[i]
        m &= ~(1 << i)
        s2[i] = m
    return s2

def _try_add(adj, s2, deg, n, u, v):
    # Returns (ok, changed_s2_indices, changed_s2_values)
    if u == v or ((adj[u] >> v) & 1):
        return False, [], []
    # C4 check: common neighbors of u and v must be < 2
    if (adj[u] & adj[v]).bit_count() >= 2:
        return False, [], []
    # Update S2 for u and v
    old_su = s2[u]
    old_sv = s2[v]
    # When adding (u,v):
    # For u: neighbors of v become distance-2 (excluding u, v, and existing neighbors of u)
    new_su = old_su | (adj[v] & ~(1 << u) & ~(1 << v) & ~adj[u])
    # For v: neighbors of u become distance-2
    new_sv = old_sv | (adj[u] & ~(1 << u) & ~(1 << v) & ~adj[v])
    # Also, for any neighbor x of u (x != v), v becomes distance-2 from x? 
    # Actually, S2[x] includes neighbors of neighbors. 
    # Adding (u,v) means:
    # - v is now neighbor of u. So for any x in N(u) (x != v), v is distance 2 from x? No, x-u-v is length 2. 
    #   So v should be added to S2[x].
    # - u is now neighbor of v. So for any y in N(v) (y != u), u is distance 2 from y.
    #   So u should be added to S2[y].
    
    # We need to update S2 for all neighbors of u and v.
    # This is expensive if done naively, but n is small.
    # However, the prompt's "S2-mask updates" implies we maintain S2 for O(1) or O(deg) updates.
    # Let's do the full update for correctness, but it might be slow.
    # Alternative: Only check C4 via common neighbors. The S2 is mainly for speedup in *finding* candidates?
    # The current program just checks (adj[u] & adj[v]).bit_count() < 2. That is O(1) bit op.
    # The "S2-mask" idea from the lesson says "Bitmask-based greedy edge addition with S2-mask updates allows fast C4-checking".
    # But (adj[u] & adj[v]) IS the fast C4 check. S2 is useful if we want to find *which* edges to remove or add to fix things?
    # Or maybe S2 helps in verifying if a removal is safe?
    
    # Let's stick to the fast (adj[u] & adj[v]) check for addition.
    # For removal, we don't need S2, just remove from adj.
    
    # So, do we even need S2?
    # The lesson says "S2-mask updates allows fast C4-checking".
    # Actually, the C4 check IS fast with adj bitmasks.
    # Maybe S2 is used to find *candidates* for edges to add that won't form C4?
    # If we maintain S2, we know that if v is in S2[u], then adding (u,v) creates a C4?
    # No. If v is in S2[u], there exists w such that u-w-v is a path.
    # If we add (u,v), does it create a C4?
    # A C4 is u-x-y-v-u.
    # If we add (u,v), we create a C4 if there is a path of length 3 between u and v in the OLD graph.
    # Path length 3: u - a - b - v.
    # This means b is in S2[u] AND a is in N(b) AND a is in N(v)?
    # Or more simply: |N(u) ∩ N(v)| >= 2.
    # The condition |N(u) ∩ N(v)| >= 2 is equivalent to the existence of a path of length 2 between u and v.
    # Wait. If |N(u) ∩ N(v)| >= 2, say w1, w2 are common neighbors.
    # Then u-w1-v-w2-u is a C4.
    # So adding (u,v) creates a C4 IF AND ONLY IF |N(u) ∩ N(v)| >= 2.
    # This is exactly what we check.
    # So S2 is not strictly necessary for the check, but maintaining it might allow faster *candidate generation*?
    # E.g., if we want to find a vertex v such that (adj[u] & adj[v]).bit_count() < 2.
    # This is hard to do with S2 directly.
    
    # Let's just use the adj bitmask check. It's O(1) and very fast in Python.
    
    # Apply change
    adj[u] |= (1 << v)
    adj[v] |= (1 << u)
    deg[u] += 1
    deg[v] += 1
    
    # We don't return S2 changes if we aren't using them for the check.
    # But the lesson mentions S2. Let's see if we can use S2 to speed up something else.
    # Maybe for *removal*? No.
    # Maybe for *finding* a vertex to add?
    # If we maintain S2, we can quickly see if a vertex is "close".
    # But the check is already fast.
    
    # Let's just return success.
    return True, [], []

def _try_remove(adj, deg, n, u, v):
    if u == v or not ((adj[u] >> v) & 1):
        return False
    adj[u] &= ~(1 << v)
    adj[v] &= ~(1 << u)
    deg[u] -= 1
    deg[v] -= 1
    return True

def _verify(n, adj, min_deg_req):
    # Check min degree
    if min(deg) < min_deg_req:
        return False
    # Check C4-free
    for i in range(n):
        for j in range(i + 1, n):
            if (adj[i] >> j) & 1:
                if (adj[i] & adj[j]).bit_count() >= 2:
                    return False
    return True

def _make_polarity_graph(q):
    # Returns (n, adj_list) for polarity graph of PG(2, q)
    # n = q^2 + q + 1
    # Vertices are points of PG(2, q).
    # Two points are adjacent if one lies on the polar line of the other.
    # This is a C4-free graph.
    # Construction for small q (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47...)
    # We need n <= 47.
    # q=2: n=7
    # q=3: n=13
    # q=5: n=31
    # q=7: n=57 (too big)
    # So we can use q=2, 3, 5.
    # For q=5, n=31. Min degree in polarity graph of PG(2, q) is roughly q+1?
    # Actually, each point has q+1 neighbors (the points on its polar line, excluding itself? No, self-adjacency depends on polarity).
    # In a standard polarity, each point is adjacent to q+1 points (including itself if absolute).
    # We want a simple graph, so no self-loops.
    # The degree is q or q+1.
    # For q=5, degree is 5 or 6.
    # If n=31, min_deg_req = 31 - 39 = -8. So any graph works.
    # But we want to *grow* it or use it as a seed for n=46.
    
    # This construction is complex to implement correctly for arbitrary q.
    # Let's use a simpler seed: a random graph or a known structure.
    # Actually, for n=46, min_deg_req = 7.
    # A random graph with p=0.15 has expected degree 46*0.15 = 6.9.
    # It might be C4-free? Unlikely.
    # C4-free graphs are sparse. Max edges in C4-free graph on n vertices is O(n^{3/2}).
    # For n=46, max edges ~ 46^{1.5} * 0.5 ~ 155.
    # Average degree ~ 2 * 155 / 46 ~ 6.7.
    # So we are near the extremal number.
    
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_n = 39
    best_edges = []
    
    # We aim for n=47, then 46
    for target_n in [47, 46]:
        if time.time() > deadline:
            break
            
        min_deg_req = target_n - 39
        
        # If min_deg_req is 0 or negative, it's trivial
        if min_deg_req <= 0:
            return {'n': target_n, 'edges': []}
            
        found = False
        max_restarts = 3000 if target_n == 46 else 10000
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            random.seed(restart)
            
            adj = [0] * target_n
            degrees = [0] * target_n
            
            # Strategy: Greedy construction with randomization
            # Start with an empty graph and add edges to satisfy min degree.
            
            stuck_count = 0
            max_iterations = 5000
            
            for _ in range(max_iterations):
                min_deg = min(degrees)
                if min_deg >= min_deg_req:
                    found = True
                    break
                
                # Find vertex with minimum degree
                candidates = [i for i in range(target_n) if degrees[i] == min_deg]
                v = random.choice(candidates)
                
                # Find valid neighbors for v
                valid_us = []
                for u in range(target_n):
                    if u == v:
                        continue
                    if (adj[v] >> u) & 1:
                        continue
                    # Check C4 condition
                    if (adj[v] & adj[u]).bit_count() < 2:
                        valid_us.append(u)
                        
                if not valid_us:
                    stuck_count += 1
                    if stuck_count > 5:
                        break
                    # Try a random removal to unstick?
                    # Or just break and restart.
                    break
                else:
                    # Pick a random valid u
                    u = random.choice(valid_us)
                    adj[v] |= (1 << u)
                    adj[u] |= (1 << v)
                    degrees[v] += 1
                    degrees[u] += 1
                    stuck_count = 0
            
            if found:
                # Verify
                if _verify(target_n, adj, min_deg_req):
                    edges = []
                    for i in range(target_n):
                        for j in range(i + 1, target_n):
                            if (adj[i] >> j) & 1:
                                edges.append([i, j])
                    best_n = target_n
                    best_edges = edges
                    return {'n': best_n, 'edges': best_edges}
        
        # If we didn't find it for this target_n, continue to next
        
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
