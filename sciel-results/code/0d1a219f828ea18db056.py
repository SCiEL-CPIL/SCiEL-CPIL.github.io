import numpy as np

def construct():
    # Parameters
    q = 7
    # Generate points of PG(2, q)
    points = []
    for x in range(q):
        for y in range(q):
            points.append((x, y, 1))
    for x in range(q):
        points.append((x, 1, 0))
    points.append((1, 0, 0))
    
    n_total = len(points) # 57
    point_index = {p: i for i, p in enumerate(points)}
    
    adj = [set() for _ in range(n_total)]
    degrees = [0] * n_total
    
    for i in range(n_total):
        u = points[i]
        for j in range(i + 1, n_total):
            v = points[j]
            dot = (u[0]*v[0] + u[1]*v[1] + u[2]*v[2]) % q
            if dot == 0:
                adj[i].add(j)
                adj[j].add(i)
                degrees[i] += 1
                degrees[j] += 1
                
    # Identify absolute points (degree q=7) and non-absolute (degree q+1=8)
    A = [i for i in range(n_total) if degrees[i] == 7]
    B = [i for i in range(n_total) if degrees[i] == 8]
    
    # N(A) is the set of vertices in B adjacent to any vertex in A
    N_A = set()
    for a in A:
        for b in adj[a]:
            if b in B:
                N_A.add(b)
                
    S = [b for b in B if b not in N_A]
    
    def try_find_subset(n_target):
        k = n_total - n_target
        min_deg_req = n_target - 39
        
        # Constraints:
        # For u in V \ R: deg(u) - |N(u) \cap R| >= min_deg_req
        # => |N(u) \cap R| <= deg(u) - min_deg_req
        
        limits = {}
        for u in range(n_total):
            limits[u] = degrees[u] - min_deg_req
            
        # If limit is negative, impossible (should not happen if min_deg_req <= min(deg))
        # For n=47, req=8. A has deg 7, limit -1. So R cannot touch A at all?
        # Actually if limit is -1, it means we need |N(u) \cap R| <= -1, which is impossible.
        # So for n=47, we cannot remove any neighbor of A.
        # And for u in A, limit = 7 - 8 = -1. So we cannot remove any neighbor of A.
        # This implies R must be disjoint from N(A) AND R must be disjoint from N(A) (same).
        # But also, for u in A, we need |N(u) \cap R| <= -1. This is impossible.
        # Wait. If min_deg_req = 8, and deg(u)=7, then 7 - |N(u) \cap R| >= 8 is impossible.
        # So n=47 is IMPOSSIBLE with this base graph?
        # Yes, because min degree of base is 7. We can only decrease it.
        # So min degree of induced subgraph <= 7.
        # So n=47 (req 8) is impossible.
        
        if min_deg_req > max(degrees):
            return None
            
        # Greedy selection of R
        # We want to pick k vertices from V such that for all u in V \ R, |N(u) \cap R| <= limits[u]
        
        # Candidates: We should prefer vertices with low "impact"
        # Impact of v is the number of neighbors u for which adding v to R would violate limits[u]
        # Initially, limits are as above.
        
        # Sort candidates by degree or some heuristic
        # To maximize success, pick vertices that are "least constrained" or "least impactful"
        
        R = set()
        current_limits = limits.copy()
        
        # We need to remove k vertices.
        # Available vertices: V
        # We cannot remove a vertex v if it causes some u to exceed limit.
        
        # Heuristic: 
        # 1. For n=46, req=7. 
        #    A (deg 7): limit 0. So R cannot contain any neighbor of A.
        #    B (deg 8): limit 1. So each u in B \ R can have at most 1 neighbor in R.
        
        # Let's try to construct R greedily.
        # If n=46, R must be subset of S (as derived before).
        # Let's verify if S has enough vertices.
        if n_target == 46:
            candidates = S
        else:
            candidates = list(range(n_total))
            
        if len(candidates) < k:
            return None
            
        # Shuffle candidates to break symmetry? Or use deterministic order?
        # Deterministic is better for reproducibility.
        # Sort by degree in the subgraph induced by candidates?
        
        # We will iterate and pick vertices.
        # To ensure we don't get stuck, we can use a simple backtracking if greedy fails,
        # but greedy with good ordering might work.
        
        # Let's try a greedy approach:
        # Pick v from candidates such that it is "safe".
        # v is safe if for all u in N(v), current_limits[u] > 0 (if we add v, limit[u] decreases by 1).
        # Actually, we need current_limits[u] >= 1 for all u in N(v) \ R.
        # And also, v itself is removed, so we don't care about v's limit.
        # But we must ensure that we don't remove a vertex that is required to satisfy others?
        # No, removing vertices only decreases degrees of remaining ones.
        
        # Algorithm:
        # 1. Initialize R = {}
        # 2. While len(R) < k:
        #    Find v in candidates \ R such that for all u in N(v) \ R, current_limits[u] > 0.
        #    If no such v, fail (or backtrack).
        #    Pick v (e.g., the one with smallest degree in remaining graph, or arbitrary).
        #    Add v to R.
        #    For all u in N(v), current_limits[u] -= 1.
        #    (Note: if u is in R, we don't care about its limit, but we already decremented it? 
        #     No, limits are for u in V \ R. If u is removed, its constraint is gone.
        #     So we should only decrement limits for u not in R.)
        
        R = set()
        current_limits = limits.copy()
        candidates_set = set(candidates)
        
        for _ in range(k):
            found = False
            # Iterate through candidates not in R
            # To be deterministic, iterate in sorted order
            for v in sorted(candidates_set - R):
                # Check if v is safe
                safe = True
                for u in adj[v]:
                    if u in R:
                        continue
                    if u not in candidates_set:
                        # u is not a candidate for removal, so it stays in V \ R
                        # Its limit must be >= 1
                        if current_limits[u] < 1:
                            safe = False
                            break
                    else:
                        # u is a candidate, might be removed later, but for now it is in V \ R
                        # So its limit must be >= 1
                        if current_limits[u] < 1:
                            safe = False
                            break
                if safe:
                    R.add(v)
                    # Update limits for neighbors of v
                    for u in adj[v]:
                        if u not in R: # u is still in V \ R
                            current_limits[u] -= 1
                    found = True
                    break
            
            if not found:
                return None
                
        # Verify final graph
        V_remaining = [i for i in range(n_total) if i not in R]
        for u in V_remaining:
            deg = len([v for v in adj[u] if v not in R])
            if deg < min_deg_req:
                return None
                
        return V_remaining, R

    # Try n=46, 45, 44, 43
    # n=47 is impossible as min degree 7 < 8
    for n_target in [46, 45, 44, 43]:
        res = try_find_subset(n_target)
        if res is not None:
            V_remaining, R = res
            n = len(V_remaining)
            
            old_to_new = {old: new for new, old in enumerate(V_remaining)}
            
            edges = []
            for u in V_remaining:
                for v in adj[u]:
                    if v in old_to_new:
                        if old_to_new[u] < old_to_new[v]:
                            edges.append([old_to_new[u], old_to_new[v]])
            
            return {'n': n, 'edges': edges}
            
    return {'n': 39, 'edges': []}
