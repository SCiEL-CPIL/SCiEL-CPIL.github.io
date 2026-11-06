import numpy as np
import time
import random

def count_c4(adj):
    n = len(adj)
    total = 0
    for u in range(n):
        for v in range(u+1, n):
            c = np.sum(adj[u] & adj[v])
            if c >= 2:
                total += c * (c - 1) // 2
    return total // 2

def construct():
    n = 31
    target_edges = 91
    deadline = time.time() + 110
    
    # Phase 1: Construct the polarity graph of PG(2, 5) which is C4-free with 90 edges
    q = 5
    points = []
    seen = set()
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize: find first non-zero coordinate, make it 1
                if x != 0:
                    inv_x = pow(x, q-2, q)
                    rep = (1, (y * inv_x) % q, (z * inv_x) % q)
                elif y != 0:
                    inv_y = pow(y, q-2, q)
                    rep = (0, 1, (z * inv_y) % q)
                else:
                    rep = (0, 0, 1)
                
                if rep not in seen:
                    seen.add(rep)
                    points.append(rep)
    
    # Map point to index
    point_to_idx = {p: i for i, p in enumerate(points)}
    
    adj = np.zeros((n, n), dtype=bool)
    edges = set()
    
    for i in range(n):
        for j in range(i + 1, n):
            p1 = points[i]
            p2 = points[j]
            # Check orthogonality: x1*x2 + y1*y2 + z1*z2 == 0 mod q
            val = (p1[0]*p2[0] + p1[1]*p2[1] + p1[2]*p2[2]) % q
            if val == 0:
                adj[i, j] = adj[j, i] = True
                edges.add((i, j))
    
    # Verify edge count is 90
    assert len(edges) == 90, f"Expected 90 edges, got {len(edges)}"
    
    # Phase 2: Find the best edge to add to reach 91 edges
    # We want to minimize the number of new C4s created.
    # Adding edge (u, v) creates C4s for each pair of common neighbors (w1, w2) of u and v
    # such that w1 and w2 are connected.
    # Actually, the number of new C4s is exactly the number of paths of length 3 between u and v
    # in the existing graph.
    
    min_new_c4s = float('inf')
    best_edge_to_add = None
    
    # Precompute neighbors for efficiency
    neighbors = [np.where(adj[i])[0] for i in range(n)]
    
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j]:
                continue
            
            # Count paths of length 3 between i and j
            # A path i-w1-w2-j exists if w1 in N(i), w2 in N(j), and w1-w2 is an edge.
            # We need to count distinct pairs (w1, w2).
            # Since the graph is C4-free, there is at most one such path for each pair?
            # No, we just count the number of edges between N(i) \ {j} and N(j) \ {i}.
            # Note: i and j are not connected, so N(i) and N(j) are disjoint? No, they can share neighbors.
            # If w1 == w2, it's a path of length 2, not 3. So we need w1 != w2.
            
            new_c4s = 0
            for w1 in neighbors[i]:
                # w1 cannot be j since adj[i,j] is False
                for w2 in neighbors[j]:
                    # w2 cannot be i since adj[i,j] is False
                    if w1 != w2 and adj[w1, w2]:
                        new_c4s += 1
            
            if new_c4s < min_new_c4s:
                min_new_c4s = new_c4s
                best_edge_to_add = (i, j)
                
                # If we find an edge that creates only 1 C4, it's optimal (minimum possible > 0)
                if min_new_c4s == 1:
                    break
        if min_new_c4s == 1:
            break
    
    if best_edge_to_add:
        u, v = best_edge_to_add
        adj[u, v] = adj[v, u] = True
        edges.add((u, v))
    
    # Phase 3: Local Search to further minimize C4 count
    # Swap edges to reduce C4 count
    
    best_c4 = count_c4(adj)
    best_adj = adj.copy()
    best_edges = set(edges)
    
    temperature = 1.0
    cooling = 0.99
    
    while time.time() < deadline:
        edge_list = list(best_edges)
        if not edge_list:
            break
        u, v = random.choice(edge_list)
        
        non_edges = [(i, j) for i in range(n) for j in range(i+1, n) 
                     if (i,j) not in best_edges]
        if not non_edges:
            break
        x, y = random.choice(non_edges)
        
        # Remove (u,v) and add (x,y)
        best_adj[u, v] = best_adj[v, u] = False
        best_edges.discard((u,v))
        best_adj[x, y] = best_adj[y, x] = True
        best_edges.add((x,y))
        
        c4_new = count_c4(best_adj)
        
        if c4_new < best_c4 or (c4_new - best_c4) < random.expovariate(1.0/max(temperature, 0.01)):
            best_c4 = c4_new
        else:
            # Revert
            best_adj[u, v] = best_adj[v, u] = True
            best_edges.add((u,v))
            best_adj[x, y] = best_adj[y, x] = False
            best_edges.discard((x,y))
        
        temperature *= cooling
    
    result = [[u, v] for u, v in sorted(best_edges)]
    return result

# EVOLVE-BLOCK-END
