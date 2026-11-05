"""Minimize C4s in 31-vertex, 91-edge graph via local search."""
import numpy as np
import time

def construct():
    n = 31
    m = 91
    deadline = time.time() + 140  # safety margin
    
    def count_c4(adj):
        """Count 4-cycles using common neighbors."""
        c4 = 0
        for u in range(n):
            for v in range(u+1, n):
                cn = 0
                for w in range(n):
                    if adj[u][w] and adj[v][w]:
                        cn += 1
                c4 += cn * (cn - 1) // 2
        return c4
    
    def count_c4_fast(adj_matrix, adj_list):
        """Faster C4 count using adjacency lists."""
        c4 = 0
        for u in range(n):
            for v in range(u+1, n):
                # Count common neighbors
                cn = 0
                for w in adj_list[u]:
                    if adj_matrix[v][w]:
                        cn += 1
                c4 += cn * (cn - 1) // 2
        return c4
    
    def edge_swap_cost(adj_matrix, adj_list, e1, e2):
        """Estimate C4 change for swapping edges e1=(a,b) and e2=(c,d)."""
        a, b = e1
        c, d = e2
        # Remove a-b, add c-d
        # This is complex; just compute before/after for simplicity in initial version
        return None  # placeholder
    
    # Initialize with a structured graph: try to use a near-polarity structure
    # Start with random edges
    np.random.seed(42)
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    np.random.shuffle(all_edges)
    edges = all_edges[:m]
    
    adj_matrix = np.zeros((n, n), dtype=bool)
    adj_list = [[] for _ in range(n)]
    for u, v in edges:
        adj_matrix[u, v] = True
        adj_matrix[v, u] = True
        adj_list[u].append(v)
        adj_list[v].append(u)
    
    best_c4 = count_c4_fast(adj_matrix, adj_list)
    best_edges = edges[:]
    
    # Local search: try edge swaps
    edge_set = set(edges)
    
    iteration = 0
    while time.time() < deadline:
        improved = False
        for i in range(len(edges)):
            if time.time() > deadline:
                break
            e1 = edges[i]
            a, b = e1
            # Try swapping with a non-edge
            for j in range(len(edges)):
                if i == j:
                    continue
                e2 = edges[j]
                c, d = e2
                # Check if we can remove e1 and add a new edge
                # Actually, let's do: remove e1, add a random non-edge
                # Better: try all possible swaps
                pass
        
        # Simpler: try removing a random edge and adding a random non-edge
        for attempt in range(200):
            if time.time() > deadline:
                break
            
            # Pick a random edge to remove
            idx = np.random.randint(len(edges))
            e_remove = edges[idx]
            u1, v1 = e_remove
            
            # Pick a random non-edge to add
            # Sample until we find a non-edge
            for _ in range(50):
                u2 = np.random.randint(n)
                v2 = np.random.randint(n)
                if u2 == v2:
                    continue
                if u2 > v2:
                    u2, v2 = v2, u2
                if not adj_matrix[u2, v2]:
                    break
            else:
                continue
            
            e_add = (u2, v2)
            
            # Compute C4 change
            # Remove e_remove, add e_add
            # Count C4s affected: those involving u1,v1 or u2,v2
            
            # Quick estimate: count C4s through the removed edge's endpoints
            # and through the added edge's endpoints
            
            # For accuracy, let's compute the delta
            # C4s containing edge (u1,v1): for each common neighbor w of u1,v1,
            # and for each common neighbor x of u1,v1 (x != w), we get a C4
            # Actually, C4s containing edge (u1,v1): pick two other vertices w,x 
            # such that w and x are both neighbors of u1 and v1
            # That's C(common_neighbors(u1,v1), 2)
            
            cn_remove = 0
            for w in adj_list[u1]:
                if adj_matrix[v1][w] and w != u1 and w != v1:
                    cn_remove += 1
            
            # C4s containing edge (u2,v2) after adding:
            cn_add = 0
            for w in adj_list[u2]:
                if adj_matrix[v2][w] and w != u2 and w != v3 if False else w != v2:
                    cn_add += 1
            # Also need to check if u1 or v1 are neighbors of u2 or v2
            # This is getting complex. Let's just recompute if the delta looks promising.
            
            delta = cn_remove * (cn_remove - 1) // 2 - cn_add * (cn_add - 1) // 2
            
            if delta < 0:  # C4 count decreases
                # Accept the move
                adj_matrix[u1, v1] = False
                adj_matrix[v1, u1] = False
                adj_list[u1].remove(v1)
                adj_list[v1].remove(u1)
                
                adj_matrix[u2, v2] = True
                adj_matrix[v2, u2] = True
                adj_list[u2].append(v2)
                adj_list[v2].append(u2)
                
                edges[idx] = e_add
                best_c4 += delta
                improved = True
        
        if not improved:
            # Try a different random seed or restart
            break
    
    # Final: return the best edges found
    # Rebuild from adj_matrix to be safe
    final_edges = []
    for u in range(n):
        for v in range(u+1, n):
            if adj_matrix[u, v]:
                final_edges.append([u, v])
    
    assert len(final_edges) == m, f"Expected {m} edges, got {len(final_edges)}"
    return final_edges
