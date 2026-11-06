import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145

    def get_polarity_graph(q):
        points = []
        for x in range(q):
            for y in range(q):
                for z in range(q):
                    if x == 0 and y == 0 and z == 0:
                        continue
                    if x != 0:
                        inv_x = pow(x, -1, q)
                        points.append((1, (y * inv_x) % q, (z * inv_x) % q))
                    elif y != 0:
                        inv_y = pow(y, -1, q)
                        points.append((0, 1, (z * inv_y) % q))
                    else:
                        inv_z = pow(z, -1, q)
                        points.append((0, 0, 1))
        
        unique_points = list(set(points))
        unique_points.sort()
        n_pts = len(unique_points)
        
        # Use Python int bitmasks for adjacency
        adj = [0] * n_pts
        for i in range(n_pts):
            for j in range(i + 1, n_pts):
                u = unique_points[i]
                v = unique_points[j]
                dot = (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) % q
                if dot == 0:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        degs = [bin(adj[i]).count('1') for i in range(n_pts)]
        return adj, n_pts, degs

    def try_polarity_subset(n_target, q=7):
        adj_full, n_full, degs_full = get_polarity_graph(q)
        if n_full < n_target:
            return None
        
        target_deg = n_target - 39
        num_remove = n_full - n_target
        
        # Backtracking search for R
        # Constraint: for v in S, popcount(adj[v] & R_mask) <= degs_full[v] - target_deg
        # For n=46, target=7, deg=8, limit=1.
        # So popcount(adj[v] & R_mask) <= 1 for all v in S.
        
        limits = [degs_full[i] - target_deg for i in range(n_full)]
        
        # If limits are small, we can use backtracking.
        # Order of removal: try to remove vertices that are "safe".
        
        def backtrack(R_mask, removed_count, S_mask):
            if removed_count == num_remove:
                # Verify
                for v in range(n_full):
                    if S_mask & (1 << v):
                        if bin(adj_full[v] & R_mask).count('1') > limits[v]:
                            return None
                # Build result
                S = [i for i in range(n_full) if S_mask & (1 << i)]
                # Map old to new
                old_to_new = {old: new for new, old in enumerate(S)}
                edges = []
                for i, old_i in enumerate(S):
                    for j, old_j in enumerate(S):
                        if i < j:
                            if adj_full[old_i] & (1 << old_j):
                                edges.append([i, j])
                return {'n': n_target, 'edges': edges}
            
            if time.time() > deadline:
                return None
            
            # Heuristic: pick a vertex to remove from the remaining candidates
            # Candidates: vertices in S (not in R)
            # We need to remove exactly num_remove - removed_count more.
            
            # To speed up, we can check if it's possible to satisfy the limits.
            # For each v in S, let avail_v = limits[v] - popcount(adj[v] & R_mask).
            # If any avail_v < 0, fail.
            # Also, we need to remove 'k' more vertices.
            # Each removed vertex u consumes 1 from avail_v for all v in N(u) \cap S.
            
            # Simple heuristic: try removing a vertex u that minimizes the sum of avail_v for its neighbors?
            # Or just try all candidates in random order.
            
            candidates = [i for i in range(n_full) if S_mask & (1 << i)]
            random.shuffle(candidates)
            
            for u in candidates:
                # Check if removing u is valid (doesn't immediately violate limits for neighbors)
                # For each neighbor v of u in S:
                # new_avail_v = avail_v - 1
                # If new_avail_v < 0, fail.
                
                valid = True
                neighbors_in_S = adj_full[u] & S_mask
                # Iterate neighbors
                n_mask = neighbors_in_S
                while n_mask:
                    lsb = n_mask & -n_mask
                    v = lsb.bit_length() - 1
                    # Count current removals in N(v)
                    # This is expensive if done naively.
                    # We can maintain a count array?
                    # But recursion depth is small (11).
                    # Let's just check.
                    if bin(adj_full[v] & R_mask).count('1') >= limits[v]:
                        valid = False
                        break
                    n_mask -= lsb
                
                if valid:
                    res = backtrack(R_mask | (1 << u), removed_count + 1, S_mask & ~(1 << u))
                    if res:
                        return res
            
            return None

        # Initial S_mask is all ones
        S_mask = (1 << n_full) - 1
        return backtrack(0, 0, S_mask)

    def try_random_greedy(n_target):
        target_deg = n_target - 39
        adj = [0] * n_target
        degs = [0] * n_target
        
        all_edges = [(i, j) for i in range(n_target) for j in range(i + 1, n_target)]
        
        for attempt in range(500):
            if time.time() > deadline:
                return None
            
            random.shuffle(all_edges)
            adj = [0] * n_target
            degs = [0] * n_target
            
            for u, v in all_edges:
                # Check C4: common neighbors of u and v must be <= 1
                common = adj[u] & adj[v]
                if common & (common - 1):
                    continue
                
                # Add edge
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                degs[u] += 1
                degs[v] += 1
            
            if min(degs) >= target_deg:
                edges = []
                for i in range(n_target):
                    for j in range(i + 1, n_target):
                        if adj[i] & (1 << j):
                            edges.append([i, j])
                return {'n': n_target, 'edges': edges}
        
        return None

    # Try n=46, 45, 44...
    for n in range(46, 39, -1):
        if time.time() > deadline:
            break
        
        # 1. Try Polarity Subset (stronger for high n)
        if n <= 57:
            res = try_polarity_subset(n)
            if res:
                return res
        
        # 2. Try Random Greedy
        res = try_random_greedy(n)
        if res:
            return res

    return {'n': 39, 'edges': []}
