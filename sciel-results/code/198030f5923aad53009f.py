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
    
    # Precompute all non-edge pairs for efficiency
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_set = set(edges)
    
    def count_c4s_for_edge(u, v, adj_matrix):
        """Count the number of C4s that would be created by adding edge (u,v).
        This equals the number of length-3 paths between u and v, which is
        the number of edges between N(u) and N(v)."""
        # A C4 containing edge (u,v) has form u-a-b-v-u where a,b are distinct
        # and a != b, a not in {u,v}, b not in {u,v}
        # Number of such C4s = sum over all pairs (a,b) where a in N(u), b in N(v),
        # adj[a,b]=1, a != b, a != v, b != u
        # Since graph is simple and u,v not adjacent yet, N(u) and N(v) don't contain u or v
        # We just need: count of edges (a,b) with a in N(u), b in N(v), a != b
        # But since a in N(u) means adj[u,a]=1 and a!=u, similarly for b
        # And a,b are from different neighbor sets (unless they share a common neighbor)
        # Actually: number of C4s = number of pairs (a,b) with a in N(u), b in N(v),
        # adj[a,b] = 1. Since the graph is undirected and simple, each such pair
        # gives exactly one C4: u-a-b-v-u.
        # We need to be careful: if a == b, that would be a triangle, not a C4.
        # But a in N(u) and b in N(v) with a=b means a is a common neighbor,
        # which would form a triangle u-a-v-u, not a C4. So we must exclude a==b.
        
        nu = adj_matrix[u]
        nv = adj_matrix[v]
        
        # Count edges between N(u) and N(v), excluding the diagonal (a==b)
        # M[a,b] = 1 if a in N(u) and b in N(v)
        # count = sum(M * adj) - sum(diagonal of M * adj_diag) but adj has 0 diagonal
        # Actually, adj[a,a] = 0 always, so sum(M * adj) already excludes a==b
        # because when a==b, adj[a,a]=0
        count = int(np.sum(nu[:, None] * nv[None, :] * adj_matrix))
        return count
    
    # 2. Greedily add edges to reach 90 edges if needed
    while current_edges < 90:
        if time.time() - start_time > DEADLINE - 5:
            break
        best_pair = None
        best_cost = float('inf')
        
        for u, v in all_pairs:
            if (u, v) in edge_set:
                continue
            
            count = count_c4s_for_edge(u, v, adj)
            
            if count < best_cost:
                best_cost = count
                best_pair = (u, v)
                
        if best_pair is None:
            break
            
        u, v = best_pair
        adj[u, v] = 1
        adj[v, u] = 1
        edges.append(best_pair)
        edge_set.add(best_pair)
        current_edges += 1
        
    # 3. Add the 91st edge with minimum cost
    if current_edges < target_edges:
        best_pair = None
        best_cost = float('inf')
        
        for u, v in all_pairs:
            if (u, v) in edge_set:
                continue
            
            count = count_c4s_for_edge(u, v, adj)
            
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
    
    # If we still don't have enough edges (shouldn't happen), add arbitrary edges
    while current_edges < target_edges:
        for u, v in all_pairs:
            if (u, v) not in edge_set:
                adj[u, v] = 1
                adj[v, u] = 1
                edges.append((u, v))
                edge_set.add((u, v))
                current_edges += 1
                if current_edges >= target_edges:
                    break
    
    return [[u, v] for u, v in edges]
# EVOLVE-BLOCK-END
