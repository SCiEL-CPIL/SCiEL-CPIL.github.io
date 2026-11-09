import numpy as np
import time
import random

def construct():
    n = 20
    deadline = time.time() + 110  # 10s margin before 120s limit
    best_edges = 0
    best_adj = None
    
    random.seed(42)
    np.random.seed(42)
    
    # Precompute witness indices for vectorization
    witness_data = {}
    for u in range(n):
        for v in range(u+1, n):
            vs = []
            xs = []
            ys = []
            us = []
            for x in range(n):
                if x == u or x == v:
                    continue
                for y in range(n):
                    if y == u or y == v or y == x:
                        continue
                    vs.append(v)
                    xs.append(x)
                    ys.append(y)
                    us.append(u)
            
            witness_data[(u, v)] = (np.array(vs), np.array(xs), np.array(ys), np.array(us))
            witness_data[(v, u)] = witness_data[(u, v)]

    def can_add_edge_fast(adj, u, v, c):
        if u > v:
            u, v = v, u
        vs, xs, ys, us = witness_data[(u, v)]
        if len(vs) == 0:
            return True
            
        c_vx = adj[vs, xs]
        c_xy = adj[xs, ys]
        c_yu = adj[ys, us]
        
        # Check existence
        valid_mask = (c_vx != -1) & (c_xy != -1) & (c_yu != -1)
        if not np.any(valid_mask):
            return True
            
        # Check rainbow condition
        conflict = (
            (c_vx == c) | (c_xy == c) | (c_yu == c) |
            (c_vx == c_xy) | (c_vx == c_yu) | (c_xy == c_yu)
        )
        
        # If any valid witness has a conflict, it's invalid
        if np.any(valid_mask & conflict):
            return False
            
        return True

    def can_recolor_edge_fast(adj, u, v, new_c):
        if u > v:
            u, v = v, u
        old_c = adj[u, v]
        if old_c == new_c:
            return True
        
        # Temporarily set
        adj[u, v] = new_c
        adj[v, u] = new_c
        
        valid = can_add_edge_fast(adj, u, v, new_c)
        
        # Revert
        adj[u, v] = old_c
        adj[v, u] = old_c
        
        return valid

    # Try multiple restarts
    for restart in range(500):
        if time.time() > deadline:
            break
        
        adj = -np.ones((n, n), dtype=int)
        np.fill_diagonal(adj, -1)
        
        # Start with a random tree
        vertices = list(range(n))
        random.shuffle(vertices)
        for i in range(1, n):
            u = vertices[i]
            v = vertices[random.randint(0, i-1)]
            c = random.randint(0, n-1)
            adj[u, v] = c
            adj[v, u] = c
        
        # Try to add more edges
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(edges)
        
        for u, v in edges:
            if time.time() > deadline:
                break
            if adj[u, v] != -1:
                continue
            
            added = False
            colors = list(range(n))
            random.shuffle(colors)
            
            for c in colors:
                if can_add_edge_fast(adj, u, v, c):
                    adj[u, v] = c
                    adj[v, u] = c
                    added = True
                    break
            
            if not added:
                # Try to recolor an edge to make room
                # Try recoloring edges incident to u or v
                for neighbor in [u, v]:
                    for x in range(n):
                        if x == neighbor:
                            continue
                        if adj[neighbor, x] == -1:
                            continue
                        old_c = adj[neighbor, x]
                        for new_c in range(n):
                            if new_c == old_c:
                                continue
                            if can_recolor_edge_fast(adj, neighbor, x, new_c):
                                adj[neighbor, x] = new_c
                                adj[x, neighbor] = new_c
                                
                                # Try to add (u,v) again
                                added_after = False
                                for c2 in range(n):
                                    if can_add_edge_fast(adj, u, v, c2):
                                        adj[u, v] = c2
                                        adj[v, u] = c2
                                        added_after = True
                                        break
                                
                                if added_after:
                                    added = True
                                    break
                                else:
                                    adj[neighbor, x] = old_c
                                    adj[x, neighbor] = old_c
                        
                        if added:
                            break
                    if added:
                        break
        
        # Local Search Phase: Try to add edges in random order multiple times
        for _ in range(5):
            if time.time() > deadline: break
            missing = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i, j] == -1]
            random.shuffle(missing)
            for u, v in missing:
                if time.time() > deadline: break
                if adj[u, v] != -1: continue
                added = False
                colors = list(range(n))
                random.shuffle(colors)
                for c in colors:
                    if can_add_edge_fast(adj, u, v, c):
                        adj[u, v] = c
                        adj[v, u] = c
                        added = True
                        break
        
        cnt = np.sum(adj != -1) // 2
        if cnt > best_edges:
            best_edges = cnt
            best_adj = adj.copy()
        
        if best_edges > 160:
            break
    
    # Convert to result format
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
