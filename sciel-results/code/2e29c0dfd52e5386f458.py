import time
import numpy as np

def count_c4_adj(adj, n):
    """Count C4s using adjacency sets."""
    count = 0
    for i in range(n):
        for j in adj[i]:
            if i < j:
                common = adj[i] & adj[j]
                count += len(common) * (len(common) - 1) // 2
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    # 1. Generate points of PG(2,5)
    points = []
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    inv_x = pow(x, 4, 5)
                    pts = (1, (y * inv_x) % 5, (z * inv_x) % 5)
                elif y != 0:
                    inv_y = pow(y, 4, 5)
                    pts = (0, 1, (z * inv_y) % 5)
                else:
                    pts = (0, 0, 1)
                if pts not in points:
                    points.append(pts)
    
    n = len(points)
    
    # 2. Build polarity graph (C4-free with 90 edges)
    adj = [set() for _ in range(n)]
    edges = []
    
    for i in range(n):
        for j in range(i + 1, n):
            u = points[i]
            v = points[j]
            b_val = (u[0]*v[0] + u[1]*v[1] + u[2]*v[2]) % 5
            if b_val == 0:
                adj[i].add(j)
                adj[j].add(i)
                edges.append([i, j])
    
    if len(edges) != 90:
        pairs = [[i, j] for i in range(31) for j in range(i + 1, 31)]
        return pairs[:91]
    
    # 3. Find best edge to add (minimizes new C4s)
    min_count = float('inf')
    best_edge = None
    
    for i in range(n):
        if time.time() > deadline:
            break
        for j in range(i + 1, n):
            if j in adj[i]:
                continue
            common = adj[i] & adj[j]
            delta = len(common) * (len(common) - 1) // 2
            if delta < min_count:
                min_count = delta
                best_edge = [i, j]
    
    if best_edge is None:
        pairs = [[i, j] for i in range(31) for j in range(i + 1, 31)]
        return pairs[:91]
    
    # Add best edge
    edges.append(best_edge)
    adj[best_edge[0]].add(best_edge[1])
    adj[best_edge[1]].add(best_edge[0])
    
    # 4. Compute initial C4 count
    current_c4 = count_c4_adj(adj, n)
    best_c4 = current_c4
    best_edges = [list(e) for e in edges]
    
    # 5. Local search: try edge swaps with incremental C4 accounting
    for iteration in range(100):
        if time.time() > deadline - 5:
            break
        
        improved = False
        
        for rem_idx in range(len(edges)):
            if time.time() > deadline - 5:
                break
            
            u, v = edges[rem_idx]
            
            # C4s destroyed by removing (u,v)
            common_uv = adj[u] & adj[v]
            destroyed = len(common_uv) * (len(common_uv) - 1) // 2
            
            # Temporarily remove (u,v)
            adj[u].remove(v)
            adj[v].remove(u)
            
            # Try adding each non-edge, compute C4s created
            best_net = 0  # only accept if net < 0
            best_add = None
            
            for i in range(n):
                if time.time() > deadline - 5:
                    break
                for j in range(i + 1, n):
                    if j in adj[i]:
                        continue
                    if i == u and j == v:
                        continue
                    
                    common_ij = adj[i] & adj[j]
                    created = len(common_ij) * (len(common_ij) - 1) // 2
                    net = created - destroyed
                    
                    if net < best_net:
                        best_net = net
                        best_add = [i, j]
            
            # Restore (u,v)
            adj[u].add(v)
            adj[v].add(u)
            
            if best_add is not None and best_net < 0:
                # Apply the swap
                a, b = best_add
                adj[u].remove(v)
                adj[v].remove(u)
                adj[a].add(b)
                adj[b].add(a)
                edges[rem_idx] = [a, b]
                
                current_c4 += best_net
                if current_c4 < best_c4:
                    best_c4 = current_c4
                    best_edges = [list(e) for e in edges]
                improved = True
        
        if not improved:
            break
    
    return best_edges
