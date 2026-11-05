"""Minimize C4s in 31-vertex, 91-edge graph via local search."""
import numpy as np
import time

def construct():
    n = 31
    m = 91
    deadline = time.time() + 145  # safety margin before 150s kill
    
    def count_c4(adj_matrix):
        """Count 4-cycles: sum over all pairs of C(common_neighbors, 2)."""
        c4 = 0
        for u in range(n):
            for v in range(u+1, n):
                cn = 0
                for w in range(n):
                    if adj_matrix[u][w] and adj_matrix[v][w]:
                        cn += 1
                c4 += cn * (cn - 1) // 2
        return c4
    
    def compute_delta(adj_matrix, remove_edge, add_edge):
        """Compute exact C4 change when removing remove_edge and adding add_edge."""
        a, b = remove_edge
        c, d = add_edge
        
        # Pairs whose common-neighbor count may change:
        # Any pair (u,v) where u or v is in {a,b,c,d}
        affected = set()
        for x in [a, b, c, d]:
            for y in range(n):
                if x != y:
                    affected.add((min(x,y), max(x,y)))
        
        # Also need to check if a==c, a==d, b==c, b==d (shared vertices)
        # The set above handles all cases
        
        old_c4 = 0
        new_c4 = 0
        
        # Create modified adjacency
        # Remove (a,b), add (c,d)
        adj_mod = adj_matrix.copy()
        adj_mod[a][b] = False
        adj_mod[b][a] = False
        adj_mod[c][d] = True
        adj_mod[d][c] = True
        
        for (u, v) in affected:
            # Old common neighbors
            cn_old = 0
            for w in range(n):
                if adj_matrix[u][w] and adj_matrix[v][w]:
                    cn_old += 1
            old_c4 += cn_old * (cn_old - 1) // 2
            
            # New common neighbors
            cn_new = 0
            for w in range(n):
                if adj_mod[u][w] and adj_mod[v][w]:
                    cn_new += 1
            new_c4 += cn_new * (cn_new - 1) // 2
        
        return new_c4 - old_c4
    
    # Initialize: random graph with 91 edges
    np.random.seed(42)
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    np.random.shuffle(all_edges)
    edges = all_edges[:m]
    
    adj_matrix = np.zeros((n, n), dtype=bool)
    edge_set = set()
    for u, v in edges:
        adj_matrix[u, v] = True
        adj_matrix[v, u] = True
        edge_set.add((u, v))
    
    best_c4 = count_c4(adj_matrix)
    best_adj = adj_matrix.copy()
    best_edges = edges[:]
    
    # Local search: try all possible edge swaps
    non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if (i,j) not in edge_set]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try all swaps: remove one edge, add one non-edge
        for i, e_remove in enumerate(edges):
            if time.time() > deadline:
                break
            for e_add in non_edges:
                # Skip if adding an edge that's already there
                if e_add in edge_set:
                    continue
                # Skip if same edge
                if e_remove == e_add:
                    continue
                # Check if the swap is valid (no multi-edges, etc.)
                a, b = e_remove
                c, d = e_add
                # Ensure we're not creating a self-loop
                if c == d:
                    continue
                
                # Compute delta
                delta = compute_delta(adj_matrix, e_remove, e_add)
                
                if delta < 0:
                    # Apply the swap
                    adj_matrix[a][b] = False
                    adj_matrix[b][a] = False
                    adj_matrix[c][d] = True
                    adj_matrix[d][c] = True
                    edge_set.remove(e_remove)
                    edge_set.add(e_add)
                    edges[i] = e_add
                    best_c4 += delta
                    improved = True
                    # Update non_edges list
                    non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if (i,j) not in edge_set]
                    break  # Restart inner loop with updated non_edges
        
        if best_c4 < best_c4_prev if 'best_c4_prev' in dir() else True:
            best_adj = adj_matrix.copy()
            best_edges = edges[:]
        best_c4_prev = best_c4
    
    # Rebuild final edges from best_adj
    final_edges = []
    for u in range(n):
        for v in range(u+1, n):
            if best_adj[u, v]:
                final_edges.append([u, v])
    
    assert len(final_edges) == m, f"Expected {m} edges, got {len(final_edges)}"
    return final_edges
