import numpy as np
import time
import random

def count_edges(adj):
    return np.sum(adj != -1) // 2

def construct():
    n = 20
    deadline = time.time() + 145
    random.seed(42)
    np.random.seed(42)
    
    # Precompute 4-cycle witnesses for each edge (u,v)
    # witness[(u,v)] = list of (x, y) such that u-v-x-y-u is a 4-cycle
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
            witness[(u, v)] = wl
            witness[(v, u)] = wl
    
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def get_forbidden(u, v, adj):
        """
        Returns set of colors forbidden for edge (u,v).
        If the 3 other edges in any 4-cycle have duplicate colors, return all colors (impossible).
        """
        forbidden = set()
        for (x, y) in witness[(u, v)]:
            if adj[v, x] == -1:
                continue
            if adj[x, y] == -1:
                continue
            if adj[y, u] == -1:
                continue
            
            c1 = adj[v, x]
            c2 = adj[x, y]
            c3 = adj[y, u]
            
            if c1 == c2 or c1 == c3 or c2 == c3:
                # This 4-cycle is already broken by duplicate colors in the other 3 edges?
                # Wait, the problem states "Every 4-cycle ... must receive four distinct colours".
                # If c1==c2, then the cycle u-v-x-y-u has colors {c_uv, c1, c1, c3}.
                # This is invalid regardless of c_uv. 
                # However, our construction maintains validity. 
                # If we are in a valid state, this case should not happen for existing edges.
                # But if we are checking a *new* edge, and the existing edges form a non-rainbow cycle,
                # then the current state is invalid. 
                # The algorithm assumes we start valid and maintain validity.
                # So this check is a safety net. If it triggers, we can't add any color.
                return set(range(n))
            
            forbidden.add(c1)
            forbidden.add(c2)
            forbidden.add(c3)
        
        return forbidden

    def is_recolor_valid(u, v, new_c, adj):
        """Check if recoloring edge (u,v) to new_c keeps all 4-cycles rainbow."""
        for (x, y) in witness[(u, v)]:
            if adj[v, x] == -1:
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

    def find_blocking_edge(u, v, adj):
        """
        For a missing edge (u,v), find a specific edge (a,b) and color conflict
        that prevents adding (u,v).
        Returns (a, b) of an edge involved in a conflicting 4-cycle, or None.
        """
        forb = get_forbidden(u, v, adj)
        if len(forb) < n:
            return None # It can be added
            
        # Find a specific witness that causes a conflict
        # We collect all potential blocker edges from the first few witnesses
        # and pick one randomly to avoid bias.
        potential_blockers = []
        for (x, y) in witness[(u, v)]:
            if adj[v, x] == -1 or adj[x, y] == -1 or adj[y, u] == -1:
                continue
            
            c1 = adj[v, x]
            c2 = adj[x, y]
            c3 = adj[y, u]
            
            # If any of these colors are duplicates, the state is invalid (shouldn't happen)
            if c1 == c2 or c1 == c3 or c2 == c3:
                return (v, x)
            
            # Add the 3 edges in the path to potential blockers
            potential_blockers.append((v, x))
            potential_blockers.append((x, y))
            potential_blockers.append((y, u))
            
            # Limit the list size to avoid excessive memory/time if many witnesses
            if len(potential_blockers) > 20:
                break
        
        if potential_blockers:
            return random.choice(potential_blockers)
            
        return None

    best_count = 0
    best_adj = None
    
    for restart in range(100):
        if time.time() > deadline:
            break
        
        # Initialize with random spanning tree
        adj = -np.ones((n, n), dtype=int)
        np.fill_diagonal(adj, -1)
        
        vertices = list(range(n))
        random.shuffle(vertices)
        for i in range(1, n):
            u = vertices[i]
            v = vertices[random.randint(0, i-1)]
            c = random.randint(0, n-1)
            adj[u, v] = c
            adj[v, u] = c
        
        # Greedy edge addition with forbidden-color check
        edges = all_edges[:]
        random.shuffle(edges)
        
        for u, v in edges:
            if time.time() > deadline:
                break
            if adj[u, v] != -1:
                continue
            
            forb = get_forbidden(u, v, adj)
            if len(forb) < n:
                available = [c for c in range(n) if c not in forb]
                # Heuristic: Choose the color that is least frequent among neighbors?
                # Simple: Random choice
                c = random.choice(available)
                adj[u, v] = c
                adj[v, u] = c
        
        # Local search
        for iteration in range(2000):
            if time.time() > deadline:
                break
            
            improved = False
            
            # 1. Try adding missing edges
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
            
            if improved:
                continue
            
            # 2. Targeted Recoloring
            # Pick a missing edge that is blocked
            if missing:
                u, v = random.choice(missing)
                blocker = find_blocking_edge(u, v, adj)
                if blocker:
                    a, b = blocker
                    old_c = adj[a, b]
                    colors = list(range(n))
                    random.shuffle(colors)
                    for new_c in colors:
                        if new_c == old_c:
                            continue
                        if is_recolor_valid(a, b, new_c, adj):
                            # Perform recolor
                            adj[a, b] = new_c
                            adj[b, a] = new_c
                            improved = True
                            # Immediately try to add the edge that was blocked
                            if adj[u, v] == -1:
                                forb = get_forbidden(u, v, adj)
                                if len(forb) < n:
                                    available = [c for c in range(n) if c not in forb]
                                    if available:
                                        c_add = random.choice(available)
                                        adj[u, v] = c_add
                                        adj[v, u] = c_add
                            break
            
            # 3. Random Recoloring (Escape local optima)
            if not improved:
                existing = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i,j] != -1]
                random.shuffle(existing)
                # Try recoloring more edges to find a way out
                for u, v in existing[:20]:
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
        
        if best_count > 160:
            break
    
    # Final polish
    if best_adj is not None and time.time() < deadline - 5:
        adj = best_adj.copy()
        for polish_iter in range(1000):
            if time.time() > deadline - 2:
                break
            
            improved = False
            missing = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i,j] == -1]
            random.shuffle(missing)
            
            # Try adding
            for u, v in missing:
                if time.time() > deadline - 2:
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
            
            if not improved:
                # Targeted recoloring on best solution
                if missing:
                    u, v = random.choice(missing)
                    blocker = find_blocking_edge(u, v, adj)
                    if blocker:
                        a, b = blocker
                        old_c = adj[a, b]
                        colors = list(range(n))
                        random.shuffle(colors)
                        for new_c in colors:
                            if new_c == old_c:
                                continue
                            if is_recolor_valid(a, b, new_c, adj):
                                adj[a, b] = new_c
                                adj[b, a] = new_c
                                improved = True
                                # Try adding again
                                if adj[u, v] == -1:
                                    forb = get_forbidden(u, v, adj)
                                    if len(forb) < n:
                                        available = [c for c in range(n) if c not in forb]
                                        if available:
                                            c_add = random.choice(available)
                                            adj[u, v] = c_add
                                            adj[v, u] = c_add
                                break
            
            if not improved:
                break
        
        cnt = count_edges(adj)
        if cnt > best_count:
            best_count = cnt
            best_adj = adj
    
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
