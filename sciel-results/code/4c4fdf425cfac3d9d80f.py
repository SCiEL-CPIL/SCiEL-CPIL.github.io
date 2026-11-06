import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    def time_left():
        return deadline - time.time()
    
    # Fallback: Return a trivial valid graph if we fail or run out of time
    # n=39, min degree 0. Empty graph works.
    def get_fallback():
        return {'n': 39, 'edges': []}

    def generate_polarity_graph(q):
        # Generate projective plane points over F_q
        # Points are non-zero vectors in F_q^3, modulo scaling
        # We pick a representative for each 1D subspace
        points = []
        for x in range(q):
            for y in range(q):
                for z in range(q):
                    if x == 0 and y == 0 and z == 0:
                        continue
                    # Normalize: make first non-zero component 1
                    vec = np.array([x, y, z])
                    idx = np.argmax(vec != 0)
                    vec = vec // vec[idx]
                    vec = tuple(vec)
                    if vec not in points:
                        points.append(vec)
        
        n_pts = len(points)
        # Polarity matrix M = Identity
        # Adjacency: u ~ v iff u . v == 0
        adj = np.zeros((n_pts, n_pts), dtype=np.int8)
        abs_points = []
        
        for i in range(n_pts):
            u = points[i]
            # Check if absolute: u . u == 0
            dot_u = (u[0]*u[0] + u[1]*u[1] + u[2]*u[2]) % q
            if dot_u == 0:
                abs_points.append(i)
            
            for j in range(n_pts):
                if i == j:
                    continue
                v = points[j]
                dot = (u[0]*v[0] + u[1]*v[1] + u[2]*v[2]) % q
                if dot == 0:
                    adj[i, j] = 1
                    adj[j, i] = 1
        
        return adj, abs_points

    def try_prune(adj, abs_points, n_target, target_deg):
        n_total = len(adj)
        num_remove = n_total - n_target
        if num_remove < 0:
            return None
        
        # We must remove all absolute points if their degree < target_deg
        # Degree of abs points is q (for q=7, deg=7)
        # If target_deg > q, we MUST remove them.
        # For n=47, target=8, q=7. Must remove all abs.
        # For n=46, target=7, q=7. Can keep them if they lose 0 neighbors.
        
        # Strategy:
        # 1. Start with R = set of abs points (if deg < target) OR empty (if we want to try keeping them)
        # 2. Actually, let's just try removing all abs points first, as it simplifies constraints on remaining vertices.
        
        R = set(abs_points)
        if len(R) > num_remove:
            # We can't remove all abs points. We must keep some.
            # This path is complex. For q=7, |A|=8.
            # If n_target=47, num_remove=10. 8 <= 10. OK.
            # If n_target=46, num_remove=11. 8 <= 11. OK.
            # If n_target=45, num_remove=12. 8 <= 12. OK.
            pass
            
        # We need to select `num_remove - len(R)` more vertices from non-abs points
        non_abs = [i for i in range(n_total) if i not in R]
        extra_needed = num_remove - len(R)
        
        if extra_needed < 0:
            return None
        
        # Precompute neighbors of R for all non-abs vertices
        # d_R(v) = number of neighbors of v in R
        d_R = np.zeros(n_total, dtype=np.int8)
        for v in non_abs:
            d_R[v] = np.sum(adj[v, list(R)])
        
        # Constraint for remaining vertices v (not in R, not in extra):
        # deg(v) - d_R(v) - d_extra(v) >= target_deg
        # deg(v) is 8 for non-abs.
        # 8 - d_R(v) - d_extra(v) >= target_deg
        # d_extra(v) <= 8 - target_deg - d_R(v)
        
        max_extra_deg = 8 - target_deg
        
        # If d_R(v) > max_extra_deg, v MUST be removed (added to extra)
        forced_remove = []
        optional = []
        for v in non_abs:
            if d_R[v] > max_extra_deg:
                forced_remove.append(v)
            else:
                optional.append(v)
        
        if len(forced_remove) > extra_needed:
            return None # Can't satisfy constraint
        
        # We must include forced_remove in extra
        extra_set = set(forced_remove)
        remaining_slots = extra_needed - len(forced_remove)
        
        # We need to pick `remaining_slots` vertices from `optional`
        # Such that for all v in optional \ extra_set:
        # d_extra(v) <= max_extra_deg - d_R[v]
        
        # This is a small search problem.
        # Use backtracking or greedy.
        # Given small numbers (optional ~ 40, slots ~ 3), backtracking is fast.
        
        def search_extra(optional, slots, current_extra, d_R, max_extra_deg, adj, target_deg, n_total):
            if slots == 0:
                # Verify constraints for all not in current_extra
                for v in range(n_total):
                    if v in R or v in current_extra:
                        continue
                    # v is remaining
                    # d_extra(v) is count of neighbors in current_extra
                    d_ex = np.sum(adj[v, list(current_extra)])
                    if d_ex > max_extra_deg - d_R[v]:
                        return False
                return True
            
            # Heuristic: pick vertex that satisfies most constraints?
            # Just iterate
            for i in range(len(optional)):
                v = optional[i]
                # Try adding v
                new_extra = current_extra | {v}
                if search_extra(optional[:i] + optional[i+1:], slots - 1, new_extra, d_R, max_extra_deg, adj, target_deg, n_total):
                    return True
            return False
        
        # Optimization: Prune optional list
        # If a vertex v has d_R[v] == max_extra_deg, it can have 0 neighbors in extra.
        # If d_R[v] < max_extra_deg, it can have some.
        
        # To make search faster, sort optional by d_R (higher first, more constrained)
        optional.sort(key=lambda x: -d_R[x])
        
        found = search_extra(optional, remaining_slots, set(), d_R, max_extra_deg, adj, target_deg, n_total)
        
        if found:
            # Reconstruct final R
            # Wait, the search function returns bool. We need the set.
            # Let's modify search to return the set or use a global variable.
            pass
            
        # Simpler: Since numbers are small, just brute force combinations if optional is small enough?
        # optional is ~40. Combinations(40, 3) = 9880. Very fast.
        # Combinations(40, 4) = 91k. Fast.
        
        from itertools import combinations
        
        best_extra = None
        # We need to pick `remaining_slots` from `optional`
        # But we also have `forced_remove`
        
        # Let's just iterate all combinations of `remaining_slots` from `optional`
        # Note: `optional` might be large if `forced_remove` is small.
        # But `remaining_slots` is small (0, 1, 2, 3).
        
        for combo in combinations(optional, remaining_slots):
            extra_set = set(forced_remove) | set(combo)
            valid = True
            for v in range(n_total):
                if v in R or v in extra_set:
                    continue
                d_ex = np.sum(adj[v, list(extra_set)])
                if d_ex > max_extra_deg - d_R[v]:
                    valid = False
                    break
            if valid:
                best_extra = extra_set
                break
        
        if best_extra is None:
            return None
            
        final_R = R | best_extra
        final_V = [i for i in range(n_total) if i not in final_R]
        
        # Verify
        min_deg = np.inf
        for v in final_V:
            deg = np.sum(adj[v, final_V])
            if deg < min_deg:
                min_deg = deg
        
        if min_deg >= target_deg:
            # Construct edges
            edges = []
            for i in range(len(final_V)):
                ui = final_V[i]
                for j in range(i+1, len(final_V)):
                    vj = final_V[j]
                    if adj[ui, vj]:
                        edges.append([ui, vj])
            
            # Remap vertices to 0..n_target-1
            # The task requires vertices 0..n-1
            # The current vertices are labels from 0..56.
            # We need to return edges with indices 0..n_target-1
            # So we map final_V[k] -> k
            mapping = {v: i for i, v in enumerate(final_V)}
            new_edges = []
            for u, v in edges:
                new_edges.append([mapping[u], mapping[v]])
            
            return {'n': n_target, 'edges': new_edges}
        
        return None

    # Main Logic
    # Try n=47, 46, 45 with q=7 polarity
    adj, abs_pts = generate_polarity_graph(7)
    
    for n_target in [47, 46, 45]:
        if time_left() < 5:
            break
        target_deg = n_target - 39
        res = try_prune(adj, abs_pts, n_target, target_deg)
        if res is not None:
            return res
            
    # Fallback to q=5 polarity (n=31) if q=7 fails?
    # Or just return fallback
    return get_fallback()
