"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 145.0
    n = 31
    target_edges = 91
    
    # 1. Construct the polarity graph of PG(2,5)
    q = 5
    points = []
    for x in range(q):
        for y in range(q):
            points.append((x, y, 1))
    for x in range(q):
        points.append((x, 1, 0))
    points.append((1, 0, 0))
    
    adj = np.zeros((n, n), dtype=np.int8)
    edges = []
    
    for i in range(n):
        for j in range(i + 1, n):
            p1, p2 = points[i], points[j]
            # Orthogonality condition in PG(2,5)
            # x1*x2 + y1*y2 - z1*z2 = 0 mod 5
            if (p1[0]*p2[0] + p1[1]*p2[1] - p1[2]*p2[2]) % q == 0:
                adj[i, j] = 1
                adj[j, i] = 1
                edges.append((i, j))
                
    current_edges = len(edges)
    
    # 2. Greedily add edges to reach 90 edges, keeping C4 count at 0
    # We need to add edges that do not create any C4s.
    # This means the number of length-3 paths between the endpoints must be 0.
    
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_set = set(edges)
    
    while current_edges < 90:
        if time.time() - start_time > DEADLINE - 10:
            break
        best_pair = None
        best_cost = float('inf')
        
        # We only need to check non-edges
        # To speed up, we can iterate through all pairs and check if not in edge_set
        for u, v in all_pairs:
            if (u, v) in edge_set:
                continue
            
            # Count length-3 paths: edges between N(u) and N(v)
            # M[x,y] = 1 if x in N(u) and y in N(v)
            # Count = sum(M * adj)
            # Optimization: 
            # neighbors_u = np.where(adj[u] == 1)[0]
            # neighbors_v = np.where(adj[v] == 1)[0]
            # count = 0
            # for x in neighbors_u:
            #   for y in neighbors_v:
            #     if adj[x,y] == 1: count += 1
            
            # Vectorized:
            # mask_u = adj[u]
            # mask_v = adj[v]
            # count = np.sum(mask_u[:, None] * mask_v[None, :] * adj)
            
            # This is O(n^2) per pair. With ~500 non-edges, 500 * 31^2 ~ 500k ops. Fast.
            count = np.sum(adj[u][:, None] * adj[v][None, :] * adj)
            
            if count < best_cost:
                best_cost = count
                best_pair = (u, v)
                
        if best_pair is None:
            break
            
        # If best_cost > 0, we cannot add it without creating C4s.
        # But we need to reach 90 edges.
        # If ex(31, C4) = 90, there should be a sequence of additions with cost 0.
        # If we get stuck, we might have chosen a bad path.
        # However, the polarity graph is a good start.
        
        if best_cost > 0:
            # If we can't add a cost-0 edge, we might have to accept C4s?
            # But the goal is to minimize total C4s.
            # If we are forced to add C4s before reaching 90, we should pick the minimum.
            # But the problem states ex=90, so a C4-free graph with 90 edges exists.
            # We might have hit a local minimum in the greedy choice.
            # For now, just add the best one found.
            pass
            
        u, v = best_pair
        adj[u, v] = 1
        adj[v, u] = 1
        edges.append(best_pair)
        edge_set.add(best_pair)
        current_edges += 1
        
    # 3. Now we have 90 edges. Add the 91st edge with minimum cost.
    if current_edges == 90:
        best_pair = None
        best_cost = float('inf')
        
        for u, v in all_pairs:
            if (u, v) in edge_set:
                continue
            
            count = np.sum(adj[u][:, None] * adj[v][None, :] * adj)
            
            if count < best_cost:
                best_cost = count
                best_pair = (u, v)
                
        if best_pair:
            u, v = best_pair
            adj[u, v] = 1
            adj[v, u] = 1
            edges.append(best_pair)
            edge_set.add(best_pair)
            current_edges += 1

    return [[u, v] for u, v in edges]
# EVOLVE-BLOCK-END
