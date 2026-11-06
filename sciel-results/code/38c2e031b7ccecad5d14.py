import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s deadline to be safe
    
    # Try n from 17 down to 9
    for n in range(17, 8, -1):
        if time.time() > deadline - 5:
            break
        result = try_solve(n, deadline)
        if result is not None:
            return result
            
    # Fallback to n=9
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}

def try_solve(n, deadline):
    # Precompute edge indices
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)
    edge_to_idx = {}
    for idx, (u, v) in enumerate(edges):
        edge_to_idx[(u, v)] = idx
        edge_to_idx[(v, u)] = idx
        
    # Initialize random coloring
    colors = [random.randint(0, 3) for _ in range(num_edges)]
    
    # Adjacency matrices for each color
    adj = [np.zeros((n, n), dtype=bool) for _ in range(4)]
    for idx, (u, v) in enumerate(edges):
        c = colors[idx]
        adj[c][u, v] = True
        adj[c][v, u] = True
        
    def count_c6_using_edge(c, u, v):
        A = adj[c]
        # Get neighbors of u and v excluding u and v
        # We need w1 in N(v), w4 in N(u)
        # Path v -> w1 -> w2 -> w3 -> w4 -> u
        # w1, w2, w3, w4 must be distinct and not u, v
        
        # Create list of valid vertices
        valid_vertices = [w for w in range(n) if w != u and w != v]
        
        # Get neighbors in valid set
        N_v = [w for w in valid_vertices if A[v, w]]
        N_u = [w for w in valid_vertices if A[u, w]]
        
        count = 0
        # Iterate w1 in N_v, w4 in N_u
        for w1 in N_v:
            for w4 in N_u:
                if w1 == w4:
                    continue
                # Iterate w2, w3 in valid_vertices
                # w1 -> w2 -> w3 -> w4
                # w2 != w1, w2 != w4, w3 != w1, w3 != w4, w2 != w3
                for w2 in valid_vertices:
                    if w2 == w1 or w2 == w4:
                        continue
                    if not A[w1, w2]:
                        continue
                    for w3 in valid_vertices:
                        if w3 == w1 or w3 == w4 or w3 == w2:
                            continue
                        if not A[w2, w3]:
                            continue
                        if A[w3, w4]:
                            count += 1
        return count

    def total_cost():
        cost = 0
        for c in range(4):
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[c][u, v]:
                        cost += count_c6_using_edge(c, u, v)
        # Each C6 is counted twice (once for each direction? No, count_c6_using_edge counts paths v->...->u.
        # A cycle u-v-...-u is counted once when we pick edge (u,v) and path v->...->u.
        # But we iterate all edges.
        # A specific C6 has 6 edges.
        # For each edge, we count the C6 if the rest of the path exists.
        # So each C6 is counted 6 times?
        # Let's check.
        # C6: u0-u1-u2-u3-u4-u5-u0.
        # Edge (u0, u1). Path u1->u2->u3->u4->u5->u0. Counted.
        # Edge (u1, u2). Path u2->u3->u4->u5->u0->u1. Counted.
        # ...
        # So each C6 is counted 6 times.
        # We want cost = number of C6s.
        # So return cost // 6.
        return cost // 6

    # Local Search
    best_cost = total_cost()
    best_colors = colors.copy()
    
    if best_cost == 0:
        return {'n': n, 'colors': best_colors}
        
    current_cost = best_cost
    no_improve_count = 0
    max_no_improve = 1000
    
    while time.time() < deadline - 10:
        if no_improve_count > max_no_improve:
            break
            
        # Pick random edge
        idx = random.randint(0, num_edges - 1)
        u, v = edges[idx]
        c_old = colors[idx]
        
        best_new_color = -1
        best_delta = 0
        
        for c_new in range(4):
            if c_new == c_old:
                continue
            
            # Calculate delta
            # Remove (u,v) from c_old
            delta_old = count_c6_using_edge(c_old, u, v)
            
            # Add (u,v) to c_new
            # Temporarily add? No, count_c6_using_edge uses current adj.
            # We need to count C6s in G_c_new that use (u,v) if it were added.
            # This is equivalent to counting paths in G_c_new (without u,v) from N_v to N_u.
            # So we can just call count_c6_using_edge(c_new, u, v) on the CURRENT graph.
            # Because the paths do not use (u,v), so adding it doesn't affect the path existence.
            delta_new = count_c6_using_edge(c_new, u, v)
            
            delta = delta_new - delta_old
            if delta < best_delta:
                best_delta = delta
                best_new_color = c_new
                
        if best_new_color != -1 and best_delta < 0:
            # Make the move
            adj[c_old][u, v] = False
            adj[c_old][v, u] = False
            adj[best_new_color][u, v] = True
            adj[best_new_color][v, u] = True
            colors[idx] = best_new_color
            current_cost += best_delta // 6 # Approximation?
            # Actually, cost change is (delta_new - delta_old) / 6?
            # No, each C6 is counted 6 times.
            # So change in count of C6s is (delta_new - delta_old) / 6.
            # But delta_new and delta_old are counts of paths.
            # So change in total_cost is (delta_new - delta_old) / 6.
            # We should update current_cost accurately.
            # But calculating total_cost is expensive.
            # Let's just recalculate every 100 steps?
            # Or trust the delta.
            current_cost += best_delta // 6
            no_improve_count = 0
        else:
            no_improve_count += 1
            # Random restart?
            if no_improve_count == max_no_improve:
                # Restart with random coloring
                colors = [random.randint(0, 3) for _ in range(num_edges)]
                for c in range(4):
                    adj[c] = np.zeros((n, n), dtype=bool)
                for idx, (u, v) in enumerate(edges):
                    c = colors[idx]
                    adj[c][u, v] = True
                    adj[c][v, u] = True
                current_cost = total_cost()
                if current_cost == 0:
                    return {'n': n, 'colors': colors}
                best_cost = min(best_cost, current_cost)
                if current_cost < best_cost:
                    best_colors = colors.copy()
                    best_cost = current_cost
                no_improve_count = 0
                max_no_improve = int(max_no_improve * 1.5)
                
        if current_cost == 0:
            return {'n': n, 'colors': colors}
            
        if current_cost < best_cost:
            best_cost = current_cost
            best_colors = colors.copy()
            
    if best_cost == 0:
        return {'n': n, 'colors': best_colors}
        
    return None
