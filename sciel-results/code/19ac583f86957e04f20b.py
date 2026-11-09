import numpy as np
import time
import random

def count_edges(adj):
    return np.sum(adj != -1) // 2

def construct():
    n = 20
    deadline = time.time() + 115
    random.seed(42)
    np.random.seed(42)
    
    best_count = 0
    best_adj = None
    
    # Precompute all pairs
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def get_forbidden(u, v, adj):
        """
        Returns a set of colors that are forbidden for edge (u, v)
        if it were to be added to the graph represented by adj.
        If the set is all colors, it means (u, v) cannot be added.
        """
        forbidden = set()
        # Check all potential 4-cycles u-v-x-y-u
        # Edges (v,x), (x,y), (y,u) must exist.
        for x in range(n):
            if x == u or x == v:
                continue
            if adj[v, x] == -1:
                continue
            for y in range(n):
                if y == u or y == v or y == x:
                    continue
                if adj[x, y] == -1:
                    continue
                if adj[y, u] == -1:
                    continue
                
                c1 = adj[v, x]
                c2 = adj[x, y]
                c3 = adj[y, u]
                
                # The existing 3 edges must have distinct colors
                if c1 == c2 or c1 == c3 or c2 == c3:
                    # This configuration makes it impossible to add (u,v) with ANY color
                    # because the other 3 colors are not distinct, so the set of 4
                    # will have size < 4 regardless of the 4th color.
                    return set(range(n))
                
                forbidden.add(c1)
                forbidden.add(c2)
                forbidden.add(c3)
                
        return forbidden

    def is_recolor_valid(u, v, new_c, adj):
        """
        Check if recoloring existing edge (u, v) to new_c is valid.
        """
        # We need to ensure that for all 4-cycles containing (u, v),
        # the 4 colors are distinct.
        # Cycle u-v-x-y-u.
        # Colors: new_c, adj[v,x], adj[x,y], adj[y,u].
        # We know adj[v,x], adj[x,y], adj[y,u] are distinct (from previous validity).
        # So we just need new_c to be distinct from them.
        for x in range(n):
            if x == u or x == v:
                continue
            if adj[v, x] == -1:
                continue
            for y in range(n):
                if y == u or y == v or y == x:
                    continue
                if adj[x, y] == -1:
                    continue
                if adj[y, u] == -1:
                    continue
                
                c1 = adj[v, x]
                c2 = adj[x, y]
                c3 = adj[y, u]
                
                if new_c == c1 or new_c == c2 or new_c == c3:
                    return False
        return True

    for restart in range(50):
        if time.time() > deadline:
            break
            
        adj = -np.ones((n, n), dtype=int)
        np.fill_diagonal(adj, -1)
        
        # Greedy Phase
        edges = all_edges[:]
        random.shuffle(edges)
        
        for u, v in edges:
            if time.time() > deadline:
                break
            if adj[u, v] != -1:
                continue
            
            forb = get_forbidden(u, v, adj)
            if len(forb) < n:
                # Pick a random available color
                available = [c for c in range(n) if c not in forb]
                c = random.choice(available)
                adj[u, v] = c
                adj[v, u] = c
        
        # Local Search Phase
        for iteration in range(1000):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try to add missing edges
            missing = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i,j] == -1]
            random.shuffle(missing)
            
            for u, v in missing:
                if time.time() > deadline:
                    break
                if adj[u, v] != -1:
                    continue
                
                forb = get_forbidden(u, v, adj)
                if len(forb) < n:
                    available = [c for c in range(n) if c not in forb]
                    c = random.choice(available)
                    adj[u, v] = c
                    adj[v, u] = c
                    improved = True
                    # Don't break, try to add more in this iteration
                    # But we need to be careful with time.
                    # Breaking after one add is safer for time control,
                    # but adding multiple is better for density.
                    # Let's break to re-shuffle and prioritize others?
                    # No, just continue checking others.
            
            if not improved:
                # Try recoloring
                existing = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i,j] != -1]
                random.shuffle(existing)
                
                for u, v in existing[:10]:
                    if time.time() > deadline:
                        break
                    
                    old_c = adj[u, v]
                    colors = list(range(n))
                    random.shuffle(colors)
                    
                    for new_c in colors:
                        if new_c == old_c:
                            continue
                        
                        if is_recolor_valid(u, v, new_c, adj):
                            adj[u, v] = new_c
                            adj[v, u] = new_c
                            improved = True
                            break
                    
                    if improved:
                        break
            
            if not improved:
                # Stuck in this restart
                break
        
        cnt = count_edges(adj)
        if cnt > best_count:
            best_count = cnt
            best_adj = adj.copy()
            
        if best_count > 150:
            break

    # Convert to result
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
