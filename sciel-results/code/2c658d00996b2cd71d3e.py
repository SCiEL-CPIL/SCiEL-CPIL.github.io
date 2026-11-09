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

    def get_score_fast(adj, masks):
        deg = adj.sum(axis=1)
        max_sum = 0
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
                        common_mask &= common_mask - 1
        return max_sum

    def find_worst_triangle_edges(adj, masks, deg):
        worst_sum = 0
        worst_edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v]:
                    common_mask = masks[u] & masks[v]
                    while common_mask:
                        lsb = common_mask & -common_mask
                        w = lsb.bit_length() - 1
                        s = deg[u] + deg[v] + deg[w]
                        if s > worst_sum:
                            worst_sum = s
                            worst_edges = [(u, v), (u, w), (v, w)]
                        common_mask &= common_mask - 1
        return worst_edges, worst_sum

    start_time = time.time()
    time_limit = 145  # Safety margin for 150s limit

    all_possible_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    best_adj = adj.copy()
    best_masks = build_masks(best_adj)
    best_deg = best_adj.sum(axis=1)
    best_score = get_score_fast(best_adj, best_masks)

    # Simulated Annealing / Local Search
    current_adj = adj.copy()
    current_masks = build_masks(current_adj)
    current_deg = current_adj.sum(axis=1)
    current_score = get_score_fast(current_adj, current_masks)

    T_start = 5.0
    T_end = 0.1
    total_iters = 200000 # Will be cut by time limit

    for it in range(total_iters):
        if time.time() - start_time > time_limit:
            break

        frac = it / total_iters
        T = T_start * (T_end / T_start) ** frac

        # 1. Select edge to remove
        # With high probability, remove an edge from the worst triangle
        worst_edges, _ = find_worst_triangle_edges(current_adj, current_masks, current_deg)
        
        if worst_edges and random.random() < 0.8:
            u, v = random.choice(worst_edges)
        else:
            # Pick random edge
            while True:
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u < v and current_adj[u, v]:
                    break
        
        # 2. Select non-edge to add
        # Prefer non-edges that don't create new high-sum triangles or connect low-degree nodes
        # Simple heuristic: pick random non-edge
        while True:
            x = random.randint(0, n-1)
            y = random.randint(0, n-1)
            if x < y and not current_adj[x, y]:
                break

        # 3. Apply move
        current_adj[u, v] = 0
        current_adj[v, u] = 0
        current_adj[x, y] = 1
        current_adj[y, x] = 1

        current_masks[u] &= ~(1 << v)
        current_masks[v] &= ~(1 << u)
        current_masks[x] |= (1 << y)
        current_masks[y] |= (1 << x)

        current_deg[u] -= 1
        current_deg[v] -= 1
        current_deg[x] += 1
        current_deg[y] += 1

        new_score = get_score_fast(current_adj, current_masks)

        # 4. Accept/Reject
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_adj = current_adj.copy()
                best_masks = current_masks.copy()
                best_deg = current_deg.copy()
        else:
            delta = new_score - current_score
            if random.random() < np.exp(-delta / T):
                current_score = new_score
            else:
                # Revert
                current_adj[u, v] = 1
                current_adj[v, u] = 1
                current_adj[x, y] = 0
                current_adj[y, x] = 0

                current_masks[u] |= (1 << v)
                current_masks[v] |= (1 << u)
                current_masks[x] &= ~(1 << y)
                current_masks[y] &= ~(1 << x)

                current_deg[u] += 1
                current_deg[v] += 1
                current_deg[x] -= 1
                current_deg[y] -= 1

    # Convert best_adj to edges list
    best_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u, v]:
                best_edges.append([u, v])
                
    return {'n': 16, 'edges': best_edges}
