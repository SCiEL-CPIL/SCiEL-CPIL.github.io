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
    
    # Precompute witnesses as numpy arrays for vectorized operations
    # For edge (u,v), we need all (x,y) with x,y != u,v, x != y
    # such that v-x-y-u forms a 4-cycle (all 4 edges exist)
    witness = {}
    for u in range(n):
        for v in range(u+1, n):
            wl = []
            for x in range(n):
                if x == u or x == v:
                    continue
                for y in range(n):
                    if y == u or y == v or y == x:
                        continue
                    wl.append((x, y))
            witness[(u,v)] = np.array(wl, dtype=np.int32)
            witness[(v,u)] = witness[(u,v)]

    def get_forbidden_and_blocked(adj, u, v):
        """Vectorized: return (forbidden_set, blocked_flag) for adding edge (u,v)."""
        if u > v:
            u, v = v, u
        wl = witness[(u,v)]
        if len(wl) == 0:
            return set(), False
        x_arr = wl[:, 0]
        y_arr = wl[:, 1]
        c_vx = adj[v, x_arr]
        c_xy = adj[x_arr, y_arr]
        c_yu = adj[y_arr, u]
        
        # Find rows where all three edges exist
        mask = (c_vx != -1) & (c_xy != -1) & (c_yu != -1)
        if not np.any(mask):
            return set(), False
        
        c_vx_m = c_vx[mask]
        c_xy_m = c_xy[mask]
        c_yu_m = c_yu[mask]
        
        # Check for immediate blocking: any two colors equal in same cycle
        if np.any(c_vx_m == c_xy_m) or np.any(c_vx_m == c_yu_m) or np.any(c_xy_m == c_yu_m):
            return set(), True
        
        forbidden = set(np.unique(np.concatenate([c_vx_m, c_xy_m, c_yu_m])).tolist())
        return forbidden, False

    def find_valid_color(adj, u, v, color_counts, prefer_rare=True):
        """Find a valid color for edge (u,v) using maintained color counts."""
        forbidden, blocked = get_forbidden_and_blocked(adj, u, v)
        if blocked:
            return -1
        available = [c for c in range(n) if c not in forbidden]
        if not available:
            return -1
        if prefer_rare:
            best_c = min(available, key=lambda c: color_counts[c])
            return best_c
        return random.choice(available)

    def can_recolor_edge_fast(adj, u, v, new_c):
        """Check if recoloring edge (u,v) to new_c keeps all 4-cycles valid."""
        if u > v:
            u, v = v, u
        old_c = adj[u, v]
        if old_c == new_c:
            return True
        
        adj[u, v] = new_c
        adj[v, u] = new_c
        
        wl = witness[(u,v)]
        if len(wl) == 0:
            adj[u, v] = old_c
            adj[v, u] = old_c
            return True
        
        x_arr = wl[:, 0]
        y_arr = wl[:, 1]
        c_vx = adj[v, x_arr]
        c_xy = adj[x_arr, y_arr]
        c_yu = adj[y_arr, u]
        
        mask = (c_vx != -1) & (c_xy != -1) & (c_yu != -1)
        if np.any(mask):
            c_vx_m = c_vx[mask]
            c_xy_m = c_xy[mask]
            c_yu_m = c_yu[mask]
            if np.any(new_c == c_vx_m) or np.any(new_c == c_xy_m) or np.any(new_c == c_yu_m):
                adj[u, v] = old_c
                adj[v, u] = old_c
                return False
        
        adj[u, v] = old_c
        adj[v, u] = old_c
        return True

    def try_add_with_recolor(adj, u, v, color_counts, max_recolor_attempts=3):
        """Try to add edge (u,v) by recoloring blocking edges. Returns True if added."""
        if adj[u, v] != -1:
            return True
        
        c = find_valid_color(adj, u, v, color_counts)
        if c != -1:
            adj[u, v] = c
            adj[v, u] = c
            color_counts[c] += 1
            return True
        
        # Identify blocking cycles and try to recolor edges within them
        if u > v:
            u, v = v, u
        wl = witness[(u,v)]
        if len(wl) == 0:
            return False
        
        x_arr = wl[:, 0]
        y_arr = wl[:, 1]
        c_vx_all = adj[v, x_arr]
        c_xy_all = adj[x_arr, y_arr]
        c_yu_all = adj[y_arr, u]
        
        mask = (c_vx_all != -1) & (c_xy_all != -1) & (c_yu_all != -1)
        if not np.any(mask):
            return False
        
        x_m = x_arr[mask]
        y_m = y_arr[mask]
        c_vx_m = c_vx_all[mask]
        c_xy_m = c_xy_all[mask]
        c_yu_m = c_yu_all[mask]
        
        # Find blocking cycles (where any two colors are equal)
        block_mask = (c_vx_m == c_xy_m) | (c_vx_m == c_yu_m) | (c_xy_m == c_yu_m)
        
        if not np.any(block_mask):
            return False
        
        block_idx = np.where(block_mask)[0]
        random.shuffle(block_idx.tolist())
        block_idx = np.array(block_idx[:max_recolor_attempts * 2])
        
        for idx in block_idx:
            x = int(x_m[idx])
            y = int(y_m[idx])
            cv = int(c_vx_m[idx])
            cx = int(c_xy_m[idx])
            cy = int(c_yu_m[idx])
            
            # Try recoloring (v, x)
            for new_c in range(n):
                if new_c == cv:
                    continue
                if can_recolor_edge_fast(adj, v, x, new_c):
                    adj[v, x] = new_c
                    adj[x, v] = new_c
                    color_counts[cv] -= 1
                    color_counts[new_c] += 1
                    c2 = find_valid_color(adj, u, v, color_counts)
                    if c2 != -1:
                        adj[u, v] = c2
                        adj[v, u] = c2
                        color_counts[c2] += 1
                        return True
                    else:
                        adj[v, x] = cv
                        adj[x, v] = cv
                        color_counts[cv] += 1
                        color_counts[new_c] -= 1
            
            # Try recoloring (x, y)
            for new_c in range(n):
                if new_c == cx:
                    continue
                if can_recolor_edge_fast(adj, x, y, new_c):
                    adj[x, y] = new_c
                    adj[y, x] = new_c
                    color_counts[cx] -= 1
                    color_counts[new_c] += 1
                    c2 = find_valid_color(adj, u, v, color_counts)
                    if c2 != -1:
                        adj[u, v] = c2
                        adj[v, u] = c2
                        color_counts[c2] += 1
                        return True
                    else:
                        adj[x, y] = cx
                        adj[y, x] = cx
                        color_counts[cx] += 1
                        color_counts[new_c] -= 1
            
            # Try recoloring (y, u)
            for new_c in range(n):
                if new_c == cy:
                    continue
                if can_recolor_edge_fast(adj, y, u, new_c):
                    adj[y, u] = new_c
                    adj[u, y] = new_c
                    color_counts[cy] -= 1
                    color_counts[new_c] += 1
                    c2 = find_valid_color(adj, u, v, color_counts)
                    if c2 != -1:
                        adj[u, v] = c2
                        adj[v, u] = c2
                        color_counts[c2] += 1
                        return True
                    else:
                        adj[y, u] = cy
                        adj[u, y] = cy
                        color_counts[cy] += 1
                        color_counts[new_c] -= 1
        
        return False

    def local_search(adj, color_counts, max_rounds=80):
        """Iteratively try to add missing edges via recoloring. Returns edge count."""
        edges_missing = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i, j] == -1]
        random.shuffle(edges_missing)
        
        for rnd in range(max_rounds):
            if time.time() > deadline:
                break
            progress = False
            random.shuffle(edges_missing)
            for u, v in edges_missing[:]:
                if adj[u, v] != -1:
                    edges_missing.remove((u, v))
                    continue
                if try_add_with_recolor(adj, u, v, color_counts):
                    edges_missing.remove((u, v))
                    progress = True
            if not progress:
                break
        
        return np.sum(adj != -1) // 2

    for restart in range(3000):
        if time.time() > deadline:
            break
        
        adj = -np.ones((n, n), dtype=int)
        np.fill_diagonal(adj, -1)
        color_counts = np.zeros(n, dtype=int)
        
        # Build initial spanning tree with random colors
        vertices = list(range(n))
        random.shuffle(vertices)
        for i in range(1, n):
            u = vertices[i]
            v = vertices[random.randint(0, i-1)]
            c = random.randint(0, n-1)
            adj[u, v] = c
            adj[v, u] = c
            color_counts[c] += 1
        
        # Greedy add edges in random order
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(edges)
        
        for u, v in edges:
            if time.time() > deadline:
                break
            if adj[u, v] != -1:
                continue
            c = find_valid_color(adj, u, v, color_counts)
            if c != -1:
                adj[u, v] = c
                adj[v, u] = c
                color_counts[c] += 1
        
        # Local search to add more edges
        cnt = local_search(adj, color_counts)
        
        if cnt > best_edges:
            best_edges = cnt
            best_adj = adj.copy()
        
        if best_edges > 175:
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
