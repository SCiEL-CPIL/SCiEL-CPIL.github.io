import numpy as np
import time
import random

def construct():
    n = 20
    deadline = time.time() + 115
    best_edges = 0
    best_adj = None
    
    random.seed(42)
    np.random.seed(42)
    
    # Precompute witnesses as NumPy arrays
    # For each edge (u,v) with u<v, store arrays of x and y such that v-x-y-u is a 4-cycle
    witness_u = {}
    witness_v = {}
    for u in range(n):
        for v in range(u+1, n):
            xs = []
            ys = []
            for x in range(n):
                if x == u or x == v:
                    continue
                for y in range(n):
                    if y == u or y == v or y == x:
                        continue
                    xs.append(x)
                    ys.append(y)
            witness_u[(u,v)] = np.array(xs, dtype=np.int32)
            witness_v[(u,v)] = np.array(ys, dtype=np.int32)
            witness_u[(v,u)] = witness_u[(u,v)]
            witness_v[(v,u)] = witness_v[(u,v)]
    
    def get_forbidden_and_blocked(adj, u, v):
        """Vectorized: Return (forbidden_set, blocked_flag) for adding edge (u,v)."""
        if u > v:
            u, v = v, u
        xs = witness_u[(u,v)]
        ys = witness_v[(u,v)]
        
        # Get colors for all witness paths
        c_vx = adj[v, xs]  # shape (len(xs),)
        c_xy = adj[xs, ys]
        c_yu = adj[ys, u]
        
        # Mask where all three edges exist
        mask = (c_vx != -1) & (c_xy != -1) & (c_yu != -1)
        
        if not np.any(mask):
            return set(), False
        
        c_vx_m = c_vx[mask]
        c_xy_m = c_xy[mask]
        c_yu_m = c_yu[mask]
        
        # Check if any path has a conflict (two edges with same color)
        if np.any(c_vx_m == c_xy_m) or np.any(c_vx_m == c_yu_m) or np.any(c_xy_m == c_yu_m):
            return set(), True
        
        # Collect forbidden colors
        forbidden = set(np.unique(np.concatenate([c_vx_m, c_xy_m, c_yu_m])).tolist())
        return forbidden, False
    
    def find_valid_color(adj, u, v, deterministic=False):
        """Find a valid color for edge (u,v)."""
        forbidden, blocked = get_forbidden_and_blocked(adj, u, v)
        if blocked:
            return -1
        available = [c for c in range(n) if c not in forbidden]
        if not available:
            return -1
        if deterministic:
            color_counts = np.zeros(n, dtype=int)
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i, j] != -1:
                        color_counts[adj[i, j]] += 1
            best_c = min(available, key=lambda c: color_counts[c])
            return best_c
        return random.choice(available)
    
    def can_recolor_edge_fast(adj, u, v, new_c):
        """Vectorized check if recoloring edge (u,v) to new_c keeps all 4-cycles valid."""
        if u > v:
            u, v = v, u
        old_c = adj[u, v]
        if old_c == new_c:
            return True
        
        xs = witness_u[(u,v)]
        ys = witness_v[(u,v)]
        
        c_vx = adj[v, xs]
        c_xy = adj[xs, ys]
        c_yu = adj[ys, u]
        
        mask = (c_vx != -1) & (c_xy != -1) & (c_yu != -1)
        
        if np.any(mask):
            if np.any(c_vx[mask] == new_c) or np.any(c_xy[mask] == new_c) or np.any(c_yu[mask] == new_c):
                return False
        
        return True
    
    def local_search(adj, max_rounds=100):
        """Iteratively try to add missing edges via recoloring. Returns edge count."""
        edges_missing = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i, j] == -1]
        random.shuffle(edges_missing)
        
        for rnd in range(max_rounds):
            if time.time() > deadline:
                break
            progress = False
            random.shuffle(edges_missing)
            for u, v in edges_missing:
                if adj[u, v] != -1:
                    continue
                c = find_valid_color(adj, u, v)
                if c != -1:
                    adj[u, v] = c
                    adj[v, u] = c
                    edges_missing.remove((u, v))
                    progress = True
                    continue
                # Try recoloring a neighbor edge
                added = False
                for neighbor in [u, v]:
                    if added:
                        break
                    for x in range(n):
                        if x == neighbor:
                            continue
                        if adj[neighbor, x] == -1:
                            continue
                        old_c = adj[neighbor, x]
                        candidates = list(range(n))
                        random.shuffle(candidates)
                        for new_c in candidates:
                            if new_c == old_c:
                                continue
                            if can_recolor_edge_fast(adj, neighbor, x, new_c):
                                adj[neighbor, x] = new_c
                                adj[x, neighbor] = new_c
                                c2 = find_valid_color(adj, u, v)
                                if c2 != -1:
                                    adj[u, v] = c2
                                    adj[v, u] = c2
                                    edges_missing.remove((u, v))
                                    added = True
                                    progress = True
                                    break
                                else:
                                    adj[neighbor, x] = old_c
                                    adj[x, neighbor] = old_c
                        if added:
                            break
            if not progress:
                break
        
        return int(np.sum(adj != -1)) // 2
    
    for restart in range(2000):
        if time.time() > deadline:
            break
        
        adj = -np.ones((n, n), dtype=np.int32)
        np.fill_diagonal(adj, -1)
        
        # Build initial spanning tree with random colors
        vertices = list(range(n))
        random.shuffle(vertices)
        for i in range(1, n):
            u = vertices[i]
            v = vertices[random.randint(0, i-1)]
            c = random.randint(0, n-1)
            adj[u, v] = c
            adj[v, u] = c
        
        # Greedy add edges in random order
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(edges)
        
        for u, v in edges:
            if time.time() > deadline:
                break
            if adj[u, v] != -1:
                continue
            c = find_valid_color(adj, u, v)
            if c != -1:
                adj[u, v] = c
                adj[v, u] = c
        
        # Local search to add more edges
        cnt = local_search(adj)
        
        if cnt > best_edges:
            best_edges = cnt
            best_adj = adj.copy()
        
        if best_edges > 170:
            break
    
    res = []
    if best_adj is None:
        for v in range(1, n):
            res.append([0, v, 0])
        return res
    
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i, j] != -1:
                res.append([i, j, int(best_adj[i, j])])
    return res
