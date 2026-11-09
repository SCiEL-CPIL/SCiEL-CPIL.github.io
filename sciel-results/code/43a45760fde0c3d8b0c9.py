import numpy as np
import time
import random

def can_add_edge(adj, u, v, c):
    """
    Check if adding edge (u, v) with color c to graph adj is valid.
    Returns True if valid, False otherwise.
    """
    n = adj.shape[0]
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
            
            # Check cycle u-v-x-y-u
            c_vx = adj[v, x]
            c_xy = adj[x, y]
            c_yu = adj[y, u]
            
            if len({c, c_vx, c_xy, c_yu}) < 4:
                return False
    return True

def can_recolor_edge(adj, u, v, new_c):
    """
    Check if changing color of edge (u, v) to new_c is valid.
    Assumes edge (u, v) exists.
    """
    n = adj.shape[0]
    old_c = adj[u, v]
    if old_c == new_c:
        return True
    
    # Temporarily change
    adj[u, v] = new_c
    adj[v, u] = new_c
    
    valid = True
    # Check all 4-cycles containing (u, v)
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
            
            # Cycle u-v-x-y-u
            c_vx = adj[v, x]
            c_xy = adj[x, y]
            c_yu = adj[y, u]
            
            if len({new_c, c_vx, c_xy, c_yu}) < 4:
                valid = False
                break
        if not valid:
            break
    
    # Revert
    adj[u, v] = old_c
    adj[v, u] = old_c
    
    return valid

def construct():
    n = 20
    best_edges = 0
    best_adj = None
    
    deadline = time.time() + 110  # 10s margin
    
    random.seed(42)
    
    # Try multiple restarts with different initializations
    for restart in range(200):
        if time.time() > deadline:
            break
        
        # Initialize with a random sparse graph
        # Start with a random matching to ensure some edges
        adj = -np.ones((n, n), dtype=int)
        np.fill_diagonal(adj, -1)
        
        # Add a random tree (ensures connectivity and no cycles initially)
        # Or just random edges
        # Let's start with a random set of edges that is valid
        # A tree is always valid (no 4-cycles)
        # Generate a random tree
        vertices = list(range(n))
        random.shuffle(vertices)
        for i in range(1, n):
            u = vertices[i]
            v = vertices[random.randint(0, i-1)]
            c = random.randint(0, n-1)
            adj[u, v] = c
            adj[v, u] = c
        
        # Now try to add more edges
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(edges)
        
        for u, v in edges:
            if time.time() > deadline:
                break
            if adj[u, v] != -1:
                continue
            
            added = False
            # Try all colors
            colors = list(range(n))
            random.shuffle(colors)
            
            for c in colors:
                if can_add_edge(adj, u, v, c):
                    adj[u, v] = c
                    adj[v, u] = c
                    added = True
                    break
            
            if not added:
                # Try to recolor an existing edge to make room
                # Find a conflicting cycle
                # We can just try to recolor each edge in the neighborhood
                # This is expensive, so we do it rarely or with a limit
                
                # Try to recolor edges (u, x) or (v, x) for some x
                # Actually, the conflict comes from a cycle u-v-x-y-u
                # So we need to change one of (v,x), (x,y), (y,u)
                
                # Let's try to change color of (v, x) for a random x connected to v
                # If that fails, try (y, u) etc.
                
                # To keep it simple and fast, we might skip recoloring in the first pass
                # and rely on restarts.
                # But recoloring can significantly improve density.
                
                # Let's try a limited number of recoloring attempts
                recolor_success = False
                # Identify potential conflicting edges
                # We don't know which x,y caused the conflict without re-running the check
                # Let's just try to recolor a random edge incident to u or v
                
                # Actually, a smarter way:
                # Find the first conflicting cycle
                conflict_edges = []
                for x in range(n):
                    if x == u or x == v: continue
                    if adj[v, x] == -1: continue
                    for y in range(n):
                        if y == u or y == v or y == x: continue
                        if adj[x, y] == -1: continue
                        if adj[y, u] == -1: continue
                        
                        # This is a potential cycle
                        # Check if it's actually conflicting for ANY color c
                        # We already tried all c and failed.
                        # So for this cycle, for every c, {c, c_vx, c_xy, c_yu} has size < 4
                        # This means {c_vx, c_xy, c_yu} must contain at least 2 colors that block all c?
                        # No, it means that for every c, c is in {c_vx, c_xy, c_yu} OR one of the others matches c.
                        
                        # Let's just try to recolor one of the existing edges in this cycle
                        # Try recoloring (v, x)
                        # Try recoloring (x, y)
                        # Try recoloring (y, u)
                        
                        # We'll pick one at random
                        pass
                
                # Simple heuristic: try to recolor a random edge incident to u or v
                # This is weak.
                
                # Better: Try to recolor edges (v, x) for all x connected to v
                # If any recoloring allows adding (u, v), do it.
                
                # This is O(deg(v) * n^2) which is okay.
                
                for x in range(n):
                    if x == u or x == v: continue
                    if adj[v, x] == -1: continue
                    
                    old_c_vx = adj[v, x]
                    for new_c in range(n):
                        if new_c == old_c_vx: continue
                        
                        # Check if recoloring (v, x) to new_c is valid
                        if can_recolor_edge(adj, v, x, new_c):
                            # Recolor
                            adj[v, x] = new_c
                            adj[x, v] = new_c
                            
                            # Now try to add (u, v) again
                            added_after_recolor = False
                            for c2 in range(n):
                                if can_add_edge(adj, u, v, c2):
                                    adj[u, v] = c2
                                    adj[v, u] = c2
                                    added_after_recolor = True
                                    break
                            
                            if added_after_recolor:
                                added = True
                                recolor_success = True
                                break
                            else:
                                # Revert recolor
                                adj[v, x] = old_c_vx
                                adj[x, v] = old_c_vx
                            
                    if recolor_success:
                        break
                
                if not added:
                    # Try similar for u
                    for x in range(n):
                        if x == u or x == v: continue
                        if adj[u, x] == -1: continue
                        
                        old_c_ux = adj[u, x]
                        for new_c in range(n):
                            if new_c == old_c_ux: continue
                            
                            if can_recolor_edge(adj, u, x, new_c):
                                adj[u, x] = new_c
                                adj[x, u] = new_c
                                
                                added_after_recolor = False
                                for c2 in range(n):
                                    if can_add_edge(adj, u, v, c2):
                                        adj[u, v] = c2
                                        adj[v, u] = c2
                                        added_after_recolor = True
                                        break
                                
                                if added_after_recolor:
                                    added = True
                                    break
                                else:
                                    adj[u, x] = old_c_ux
                                    adj[x, u] = old_c_ux
                            
                            if added:
                                break
                        
                        if added:
                            break

        # Count edges
        cnt = np.sum(adj != -1) // 2
        if cnt > best_edges:
            best_edges = cnt
            best_adj = adj.copy()
            # Update deadline if we found a very good solution? No, keep fixed.
        
        # Early stop if we reach a high density
        if best_edges > 150:
            break

    # Convert best_adj to list of lists
    res = []
    if best_adj is None:
        # Fallback to star
        for v in range(1, n):
            res.append([0, v, 0])
        return res

    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i, j] != -1:
                res.append([i, j, int(best_adj[i, j])])
    return res
