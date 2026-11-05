import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    n = 31
    m = 91
    
    # 1. Construct PG(2,5) points
    points = []
    seen = set()
    for a in range(5):
        for b in range(5):
            for c in range(5):
                if a == 0 and b == 0 and c == 0:
                    continue
                vec = [a, b, c]
                # Normalize
                for i in range(3):
                    if vec[i] != 0:
                        inv = pow(vec[i], 3, 5)
                        vec = [(v * inv) % 5 for v in vec]
                        break
                tvec = tuple(vec)
                if tvec not in seen:
                    seen.add(tvec)
                    points.append(tvec)
    
    assert len(points) == 31, f"Expected 31 points, got {len(points)}"
    
    # 2. Construct polarity graph
    adj = [0] * n
    edges = []
    
    for i in range(n):
        for j in range(i + 1, n):
            p1 = points[i]
            p2 = points[j]
            dot = (p1[0]*p2[0] + p1[1]*p2[1] + p1[2]*p2[2]) % 5
            if dot == 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                edges.append((i, j))
                
    # Helper to count C4s using bitmasks
    def count_c4s(adj_list):
        total = 0
        for u in range(n):
            for v in range(u + 1, n):
                common = adj_list[u] & adj_list[v]
                k = bin(common).count('1')
                if k >= 2:
                    total += k * (k - 1) // 2
        return total // 2

    # Helper to calculate delta C4 when adding edge (x, y)
    def delta_add(x, y):
        # x and y are not currently connected
        delta = 0
        # Pairs (x, v) and (y, v) for v != x, y
        # k_xv = popcount(adj[x] & adj[v])
        # k_yv = popcount(adj[y] & adj[v])
        
        # We need to know if y is in N(v) for the pair (x,v) update?
        # k'_xv = k_xv + (1 if y in N(v) else 0)
        # k'_yv = k_yv + (1 if x in N(v) else 0)
        
        for v in range(n):
            if v == x or v == y:
                continue
            
            # For pair (x, v)
            k_xv = bin(adj[x] & adj[v]).count('1')
            # y is in N(v) if adj[v] has bit y
            if adj[v] & (1 << y):
                k_xv_new = k_xv + 1
            else:
                k_xv_new = k_xv
            
            # For pair (y, v)
            k_yv = bin(adj[y] & adj[v]).count('1')
            # x is in N(v) if adj[v] has bit x
            if adj[v] & (1 << x):
                k_yv_new = k_yv + 1
            else:
                k_yv_new = k_yv
                
            delta += (k_xv_new * (k_xv_new - 1) // 2) - (k_xv * (k_xv - 1) // 2)
            delta += (k_yv_new * (k_yv_new - 1) // 2) - (k_yv * (k_yv - 1) // 2)
            
        # Pair (x, y) itself doesn't change k_xy as derived earlier?
        # Let's double check.
        # k_xy = popcount(adj[x] & adj[y])
        # k'_xy = popcount((adj[x] | (1<<y)) & (adj[y] | (1<<x)))
        # = popcount((adj[x] & adj[y]) | (adj[x] & (1<<x)) | ((1<<y) & adj[y]) | ((1<<y) & (1<<x)))
        # adj[x] & (1<<x) is 0.
        # (1<<y) & adj[y] is 0.
        # (1<<y) & (1<<x) is 0.
        # So k'_xy = k_xy.
        # So no change for pair (x,y).
        
        return delta

    # Helper to calculate delta C4 when removing edge (x, y)
    def delta_remove(x, y):
        # x and y are currently connected
        # We revert the addition
        # This is symmetric to delta_add but with signs reversed?
        # Actually, we can just call delta_add on the graph WITHOUT the edge?
        # No, we are in the state WITH the edge.
        # Let's just calculate the change from current state to state without edge.
        # It's easier to just implement a generic delta that takes the current adjacency and the change.
        
        # Let's just compute the new C4 count for the affected pairs?
        # Or just use the same logic:
        # Removing (x,y) means x is removed from N(y) and y from N(x).
        # k_xv_new = k_xv - (1 if y in N(v) else 0)  -- Wait, y is in N(x) currently.
        # In the current graph, y IS in N(x).
        # So for pair (x,v), k_xv includes y if y in N(v).
        # After removal, y is NOT in N(x).
        # So k_xv_new = k_xv - (1 if y in N(v) else 0).
        
        delta = 0
        for v in range(n):
            if v == x or v == y:
                continue
            
            k_xv = bin(adj[x] & adj[v]).count('1')
            if adj[v] & (1 << y):
                k_xv_new = k_xv - 1
            else:
                k_xv_new = k_xv
            
            k_yv = bin(adj[y] & adj[v]).count('1')
            if adj[v] & (1 << x):
                k_yv_new = k_yv - 1
            else:
                k_yv_new = k_yv
                
            delta += (k_xv_new * (k_xv_new - 1) // 2) - (k_xv * (k_xv - 1) // 2)
            delta += (k_yv_new * (k_yv_new - 1) // 2) - (k_yv * (k_yv - 1) // 2)
            
        return delta

    # 3. Add the 91st edge
    if len(edges) < 91:
        best_delta = float('inf')
        best_edge = None
        
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] & (1 << j)):
                    # Calculate C4 increment
                    d = delta_add(i, j)
                    if d < best_delta:
                        best_delta = d
                        best_edge = (i, j)
                        
        if best_edge:
            i, j = best_edge
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
            edges.append((i, j))
            
    current_c4s = count_c4s(adj)
    
    # 4. Local Search
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        best_swap = None
        best_delta_total = 0 # We want to minimize C4s, so we want negative delta
        best_delta_val = 0
        
        edges_list = list(edges)
        non_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] & (1 << j)):
                    non_edges.append((i, j))
                    
        for e1 in edges_list:
            u1, v1 = e1
            # Remove e1
            adj[u1] &= ~(1 << v1)
            adj[v1] &= ~(1 << u1)
            
            # Find best non-edge to add
            local_best_delta = 0
            local_best_edge = None
            local_best_total = current_c4s # Just for reference
            
            for e2 in non_edges:
                u2, v2 = e2
                # Calculate delta for adding e2 to the graph where e1 is removed
                # Note: delta_add assumes the edge is NOT present.
                # Since we removed e1, and e2 is a non-edge (and e2 != e1), e2 is not present.
                # However, the adjacency lists have changed (e1 removed).
                # Our delta_add function uses the current `adj`.
                # So we can just call it.
                
                d = delta_add(u2, v2)
                
                if d < local_best_delta:
                    local_best_delta = d
                    local_best_edge = e2
                    
            # Revert remove
            adj[u1] |= (1 << v1)
            adj[v1] |= (1 << u1)
            
            if local_best_edge and local_best_delta < best_delta_total:
                best_delta_total = local_best_delta
                best_swap = (e1, local_best_edge)
                
        if best_swap:
            e1, e2 = best_swap
            u1, v1 = e1
            u2, v2 = e2
            
            # Perform swap
            adj[u1] &= ~(1 << v1)
            adj[v1] &= ~(1 << u1)
            adj[u2] |= (1 << v2)
            adj[v2] |= (1 << u2)
            
            # Update edges list
            edges.remove(e1)
            edges.append(e2)
            
            current_c4s += best_delta_total
            improved = True
            
    # Return edges
    result = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i] & (1 << j):
                result.append([i, j])
                
    return result
