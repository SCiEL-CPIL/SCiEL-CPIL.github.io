import time
import random
import numpy as np

def construct():
    n = 16
    # Initialize with K_{8,8} + (0,1)
    # This has 64 + 1 = 65 edges.
    adj = np.zeros((n, n), dtype=np.int8)
    for u in range(8):
        for v in range(8, 16):
            adj[u, v] = 1
            adj[v, u] = 1
    adj[0, 1] = 1
    adj[1, 0] = 1
    
    def build_masks(adj):
        masks = [0] * n
        for u in range(n):
            m = 0
            for i in range(n):
                if adj[u, i]:
                    m |= (1 << i)
            masks[u] = m
        return masks

    def get_score_and_sq_sum(adj, masks, deg=None):
        """
        Returns (max_triangle_sum, sum_of_squared_triangle_sums)
        """
        if deg is None:
            deg = adj.sum(axis=1)
        max_sum = 0
        sq_sum = 0
        # Iterate over edges
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v]:
                    common_mask = masks[u] & masks[v]
                    while common_mask:
                        lsb = common_mask & -common_mask
                        w = lsb.bit_length() - 1
                        s = deg[u] + deg[v] + deg[w]
                        if s > max_sum:
                            max_sum = s
                        sq_sum += s * s
                        common_mask &= common_mask - 1
        return max_sum, sq_sum

    def get_worst_triangles(adj, masks, deg, k=5):
        """
        Returns a list of the k worst triangles (highest degree sum).
        Each entry is (sum, u, v, w).
        """
        triangles = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v]:
                    common_mask = masks[u] & masks[v]
                    while common_mask:
                        lsb = common_mask & -common_mask
                        w = lsb.bit_length() - 1
                        s = deg[u] + deg[v] + deg[w]
                        triangles.append((s, u, v, w))
                        common_mask &= common_mask - 1
        # Sort descending by sum
        triangles.sort(key=lambda x: -x[0])
        return triangles[:k]

    start_time = time.time()
    time_limit = 140  # Safety margin for 150s limit
    
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Initial setup
    current_adj = adj.copy()
    current_masks = build_masks(current_adj)
    current_deg = current_adj.sum(axis=1)
    current_max, current_sq = get_score_and_sq_sum(current_adj, current_masks, current_deg)
    
    best_adj = current_adj.copy()
    best_masks = current_masks.copy()
    best_deg = current_deg.copy()
    best_max = current_max
    best_sq = current_sq

    # Simulated Annealing Parameters
    # We minimize a combined objective: primarily sum of squares, secondarily max
    # Objective = sq_sum * weight_sq + max_sum * weight_max
    # Using sq_sum as the primary driver helps with uniformity.
    
    T_start = 100.0
    T_end = 1.0
    total_iters = 300000 # Will be cut by time limit
    
    # Weights for the objective function
    # We want to push down the max, but the sum of squares is a smoother landscape.
    # Let's use a hybrid: accept if max improves, or if sq improves significantly.
    # Or simply minimize: L = sq_sum + lambda * max_sum
    # Let's try minimizing sq_sum primarily, as it's a QUBO-like objective.
    
    for it in range(total_iters):
        if time.time() - start_time > time_limit:
            break

        frac = it / total_iters
        T = T_start * (T_end / T_start) ** frac

        # 1. Select edge to remove
        # Bias towards edges in the worst triangles
        worst_tris = get_worst_triangles(current_adj, current_masks, current_deg, k=3)
        
        if worst_tris and random.random() < 0.7:
            # Pick a random edge from one of the worst triangles
            tri = random.choice(worst_tris)
            _, u, v, w = tri
            # Pick one of the 3 edges in this triangle
            edge_pair = random.choice([(u, v), (u, w), (v, w)])
            ru, rv = edge_pair
            # Ensure it's actually an edge (it should be)
            if not current_adj[ru, rv]:
                ru, rv = random.choice(all_edges)
                while not current_adj[ru, rv]:
                    ru, rv = random.choice(all_edges)
        else:
            # Pick random existing edge
            ru, rv = random.choice(all_edges)
            while not current_adj[ru, rv]:
                ru, rv = random.choice(all_edges)

        # 2. Select non-edge to add
        # Bias towards non-edges that connect low-degree vertices
        # or non-edges that are in the "worst" triangles (to break them)
        
        # Simple strategy: pick a random non-edge
        # Better: pick a non-edge (x,y) such that deg[x]+deg[y] is small
        # Let's just pick a random non-edge for now, but we can optimize this.
        # To speed up, we can precompute non-edges? No, adjacency changes.
        
        # Random non-edge selection
        while True:
            rx, ry = random.choice(all_edges)
            if not current_adj[rx, ry]:
                break
        
        # 3. Apply move
        current_adj[ru, rv] = 0
        current_adj[rv, ru] = 0
        current_adj[rx, ry] = 1
        current_adj[ry, rx] = 1

        current_masks[ru] &= ~(1 << rv)
        current_masks[rv] &= ~(1 << ru)
        current_masks[rx] |= (1 << ry)
        current_masks[ry] |= (1 << rx)

        current_deg[ru] -= 1
        current_deg[rv] -= 1
        current_deg[rx] += 1
        current_deg[ry] += 1

        new_max, new_sq = get_score_and_sq_sum(current_adj, current_masks, current_deg)
        
        # Objective: Minimize primarily the max, but use sq_sum to guide acceptance of neutral moves
        # If new_max < current_max, accept.
        # If new_max == current_max, accept if new_sq < current_sq (improves uniformity)
        # If new_max > current_max, accept with probability exp(-(new_max - current_max)/T)
        
        delta_max = new_max - current_max
        
        if delta_max < 0:
            accept = True
        elif delta_max == 0:
            # If max is same, prefer lower sq_sum (more uniform, might lead to lower max later)
            if new_sq < current_sq:
                accept = True
            else:
                # Accept with some probability to escape flat regions
                delta_sq = new_sq - current_sq
                accept = random.random() < np.exp(-delta_sq / (T * 10)) # T scaled for sq
        else:
            # Max increased
            accept = random.random() < np.exp(-delta_max / T)

        if accept:
            current_max = new_max
            current_sq = new_sq
            if new_max < best_max:
                best_max = new_max
                best_adj = current_adj.copy()
                best_masks = current_masks.copy()
                best_deg = current_deg.copy()
                # Optional: print progress for debugging (commented out for speed)
                # print(f"Iter {it}: New Best Max {best_max}")
        else:
            # Revert
            current_adj[ru, rv] = 1
            current_adj[rv, ru] = 1
            current_adj[rx, ry] = 0
            current_adj[ry, rx] = 0

            current_masks[ru] |= (1 << rv)
            current_masks[rv] |= (1 << ru)
            current_masks[rx] &= ~(1 << ry)
            current_masks[ry] &= ~(1 << rx)

            current_deg[ru] += 1
            current_deg[rv] += 1
            current_deg[rx] -= 1
            current_deg[ry] -= 1

    # Final polish: Deterministic local search minimizing max_sum strictly
    polish_adj = best_adj.copy()
    polish_masks = build_masks(polish_adj)
    polish_deg = polish_adj.sum(axis=1)
    polish_max, _ = get_score_and_sq_sum(polish_adj, polish_masks, polish_deg)
    
    improved = True
    polish_deadline = start_time + time_limit - 5  # Leave 5s margin
    
    # 2-opt swap polish
    while improved and time.time() < polish_deadline:
        improved = False
        for u, v in all_edges:
            if time.time() > polish_deadline:
                break
            if not polish_adj[u, v]:
                continue
            for x, y in all_edges:
                if x == u or x == v or y == u or y == v:
                    continue
                if polish_adj[x, y]:
                    continue
                
                # Try swap: remove (u,v), add (x,y)
                polish_adj[u, v] = 0
                polish_adj[v, u] = 0
                polish_adj[x, y] = 1
                polish_adj[y, x] = 1

                polish_masks[u] &= ~(1 << v)
                polish_masks[v] &= ~(1 << u)
                polish_masks[x] |= (1 << y)
                polish_masks[y] |= (1 << x)

                polish_deg[u] -= 1
                polish_deg[v] -= 1
                polish_deg[x] += 1
                polish_deg[y] += 1

                new_polish_max, _ = get_score_and_sq_sum(polish_adj, polish_masks, polish_deg)

                if new_polish_max < polish_max:
                    polish_max = new_polish_max
                    improved = True
                else:
                    # Revert
                    polish_adj[u, v] = 1
                    polish_adj[v, u] = 1
                    polish_adj[x, y] = 0
                    polish_adj[y, x] = 0

                    polish_masks[u] |= (1 << v)
                    polish_masks[v] |= (1 << u)
                    polish_masks[x] &= ~(1 << y)
                    polish_masks[y] &= ~(1 << x)

                    polish_deg[u] += 1
                    polish_deg[v] += 1
                    polish_deg[x] -= 1
                    polish_deg[y] -= 1

    # Use the better of best_adj and polish_adj
    if polish_max < best_max:
        best_adj = polish_adj
        best_max = polish_max

    # Convert best_adj to edges list
    best_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u, v]:
                best_edges.append([u, v])
                
    return {'n': 16, 'edges': best_edges}
