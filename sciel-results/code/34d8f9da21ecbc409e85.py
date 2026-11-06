import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_n = 39
    best_edges = []
    
    def check_c4(adj, u, v, n):
        au = adj[u]
        av = adj[v]
        w_mask = au
        while w_mask:
            lsb = w_mask & -w_mask
            w = lsb.bit_length() - 1
            if adj[w] & av:
                return True
            w_mask ^= lsb
        return False

    def try_find_graph(n, time_limit):
        min_deg_req = n - 39
        if min_deg_req <= 0:
            return True, []

        attempts = 0
        max_attempts = 5000
        
        while time.time() < time_limit and attempts < max_attempts:
            attempts += 1
            adj = [0] * n
            
            # Phase 1: Greedy construction
            edges_list = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(edges_list)
            
            for u, v in edges_list:
                if not check_c4(adj, u, v, n):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            
            # Phase 2: Local Search
            # We will try to improve the minimum degree.
            # We maintain a list of degrees.
            degrees = [bin(adj[i]).count('1') for i in range(n)]
            
            # We run a loop until we satisfy min_deg or we get stuck
            # To avoid infinite loops, we limit the number of steps in this phase
            max_ls_steps = 2000
            for step in range(max_ls_steps):
                if time.time() > time_limit:
                    break
                
                min_deg = min(degrees)
                if min_deg >= min_deg_req:
                    # Success
                    edges = []
                    for i in range(n):
                        for j in range(i+1, n):
                            if adj[i] & (1 << j):
                                edges.append([i, j])
                    return True, edges
                    
                # Pick a vertex with the lowest degree
                u = degrees.index(min_deg)
                
                # Try to add an edge to u
                added = False
                non_neigh = [v for v in range(n) if v != u and not (adj[u] & (1 << v))]
                random.shuffle(non_neigh)
                
                for v in non_neigh:
                    if not check_c4(adj, u, v, n):
                        # Add edge
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        degrees[u] += 1
                        degrees[v] += 1
                        added = True
                        if degrees[u] >= min_deg_req:
                            break
                
                if not added:
                    # Try a swap: remove (x, y), add (u, x) and (u, y)?
                    # Or remove (x, y), add (u, z)?
                    # Let's try to find an edge (x, y) such that removing it allows adding (u, x) or (u, y) or both.
                    # Actually, if we can't add any edge to u, it means for all v not in N(u), check_c4(u, v) is True.
                    # This means for all v not in N(u), there exists w in N(u) such that w is connected to v.
                    # This is a strong condition.
                    # Let's try to remove a random edge (x, y) where x, y != u.
                    # Then check if we can add (u, x) or (u, y).
                    # We need to be careful not to drop min degree of x or y below min_deg_req.
                    
                    # Pick a random edge (x, y)
                    all_edges = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i] & (1 << j)]
                    if not all_edges:
                        break
                    x, y = random.choice(all_edges)
                    
                    # Check if removing (x, y) helps
                    # We can remove it, then try to add (u, x) and (u, y)
                    # But we must ensure x and y don't drop below min_deg_req
                    if degrees[x] <= min_deg_req or degrees[y] <= min_deg_req:
                        continue
                        
                    # Remove (x, y)
                    adj[x] ^= (1 << y)
                    adj[y] ^= (1 << x)
                    degrees[x] -= 1
                    degrees[y] -= 1
                    
                    # Try to add (u, x)
                    added_x = False
                    if not (adj[u] & (1 << x)):
                        if not check_c4(adj, u, x, n):
                            adj[u] |= (1 << x)
                            adj[x] |= (1 << u)
                            degrees[u] += 1
                            degrees[x] += 1
                            added_x = True
                            
                    # Try to add (u, y)
                    added_y = False
                    if not (adj[u] & (1 << y)):
                        if not check_c4(adj, u, y, n):
                            adj[u] |= (1 << y)
                            adj[y] |= (1 << u)
                            degrees[u] += 1
                            degrees[y] += 1
                            added_y = True
                            
                    if not (added_x or added_y):
                        # Restore (x, y)
                        adj[x] |= (1 << y)
                        adj[y] |= (1 << x)
                        degrees[x] += 1
                        degrees[y] += 1
                        # If we couldn't improve, break local search and restart
                        break
                    else:
                        # We improved u's degree, continue local search
                        pass

        return False, None

    # Try n=47, 46, 45...
    for target_n in [47, 46, 45, 44, 43, 42, 41, 40]:
        if time.time() > deadline - 5:
            break
            
        t_limit = deadline - 5
        success, edges = try_find_graph(target_n, t_limit)
        if success:
            best_n = target_n
            best_edges = edges
            if target_n == 47:
                break
            break

    if best_n > 39:
        return {'n': best_n, 'edges': best_edges}
    else:
        return {'n': 39, 'edges': []}

if __name__ == "__main__":
    res = construct()
    print(res['n'])
