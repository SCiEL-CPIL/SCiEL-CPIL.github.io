"""
Strategy:
1. Construct a C4-free graph with 90 edges using a greedy approach.
2. Add one edge to reach 91 edges.
3. Choose the edge that minimizes the number of new C4s created.
   The number of new C4s created by adding edge (u,v) is the number of
   length-3 paths between u and v in the base graph.
4. Optionally, try local swaps (2-flips) to see if we can reduce C4s further.
"""
import time
import random
import numpy as np

def construct():
    n = 31
    m = 91
    deadline = time.time() + 145  # Safety margin
    
    rng = random.Random(42)
    
    def count_c4_fast(adj_mat):
        A = adj_mat.astype(np.int32)
        common = A @ A
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        non_edge_mask = mask & (adj_mat == 0)
        cv = common[non_edge_mask]
        total = np.sum(cv * (cv - 1) // 2)
        return total // 2
    
    def find_c4_free_graph():
        # Try to find a C4-free graph with 90 edges.
        # Start with a random graph and use local search to reduce C4s.
        # If we reach 0 C4s with 90 edges, we're done.
        
        best_adj = None
        best_c4 = float('inf')
        
        for attempt in range(100):
            if time.time() > deadline - 20: break
            
            # Start with a random 90-edge graph
            all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
            edges = rng.sample(all_pairs, 90)
            adj = np.zeros((n, n), dtype=np.int8)
            for u, v in edges:
                adj[u, v] = 1
                adj[v, u] = 1
                
            c4s = count_c4_fast(adj)
            
            if c4s < best_c4:
                best_c4 = c4s
                best_adj = adj.copy()
                
            if c4s == 0:
                return adj
            
            # Local search: try to reduce C4s by swapping edges
            improved = True
            while improved and time.time() < deadline - 10:
                improved = False
                c4s = count_c4_fast(adj)
                if c4s == 0:
                    break
                
                # Find a C4 and try to break it
                A = adj.astype(np.int32)
                common = A @ A
                mask = np.triu(np.ones((n, n), dtype=bool), k=1)
                non_edge_mask = mask & (adj == 0)
                cv = common[non_edge_mask]
                indices = np.where(cv >= 2)
                
                if len(indices[0]) == 0:
                    break
                
                # Pick a random C4-defining pair
                idx = rng.randint(0, len(indices[0]) - 1)
                u, v = indices[0][idx], indices[1][idx]
                
                # Common neighbors of u and v
                cn = np.where((adj[u] == 1) & (adj[v] == 1))[0]
                
                # Try to remove an edge incident to u or v and add a random non-edge
                best_swap_c4 = c4s
                best_swap = None
                
                candidates_to_remove = [(u, cn[0]), (u, cn[1]), (v, cn[0]), (v, cn[1])]
                
                for ru, rv in candidates_to_remove:
                    if ru > rv: ru, rv = rv, ru
                    if adj[ru, rv] == 0: continue
                    
                    # Try adding a random non-edge
                    for _ in range(5):
                        a = rng.randint(0, n-1)
                        b = rng.randint(0, n-1)
                        if a == b: continue
                        if a > b: a, b = b, a
                        if adj[a, b] == 1: continue
                        if (a, b) == (ru, rv): continue
                        
                        # Perform swap
                        adj[ru, rv] = 0
                        adj[rv, ru] = 0
                        adj[a, b] = 1
                        adj[b, a] = 1
                        
                        new_c4s = count_c4_fast(adj)
                        
                        if new_c4s < best_swap_c4:
                            best_swap_c4 = new_c4s
                            best_swap = (ru, rv, a, b)
                            
                        # Revert
                        adj[ru, rv] = 1
                        adj[rv, ru] = 1
                        adj[a, b] = 0
                        adj[b, a] = 0
                        
                if best_swap:
                    ru, rv, a, b = best_swap
                    adj[ru, rv] = 0
                    adj[rv, ru] = 0
                    adj[a, b] = 1
                    adj[b, a] = 1
                    improved = True
                    c4s = count_c4_fast(adj)
                    if c4s < best_c4:
                        best_c4 = c4s
                        best_adj = adj.copy()
                    if c4s == 0:
                        return adj
                        
        return best_adj
    
    base_adj = find_c4_free_graph()
    
    if base_adj is None:
        # Fallback: use a random 90-edge graph
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        edges = rng.sample(all_pairs, 90)
        base_adj = np.zeros((n, n), dtype=np.int8)
        for u, v in edges:
            base_adj[u, v] = 1
            base_adj[v, u] = 1
    
    # Now, base_adj has 90 edges (hopefully 0 C4s).
    # We need to add 1 edge to get 91 edges.
    # We want to minimize C4s.
    
    best_total_c4 = float('inf')
    best_final_adj = None
    
    # Calculate length-3 paths for all non-edges
    A = base_adj.astype(np.int32)
    A2 = A @ A
    A3 = A @ A2
    
    # Iterate over non-edges
    non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if base_adj[i, j] == 0]
    
    for u, v in non_edges:
        paths = A3[u, v]
        if paths < best_total_c4:
            best_total_c4 = paths
            best_final_adj = base_adj.copy()
            best_final_adj[u, v] = 1
            best_final_adj[v, u] = 1
            
    if best_final_adj is None:
        best_final_adj = base_adj.copy()
        for i in range(n):
            for j in range(i+1, n):
                if best_final_adj[i, j] == 0:
                    best_final_adj[i, j] = 1
                    best_final_adj[j, i] = 1
                    break
            else:
                continue
            break

    # Optional: try local swaps to see if we can reduce C4s further
    # This is a heuristic improvement step.
    current_c4 = count_c4_fast(best_final_adj)
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # Try a few random swaps
        for _ in range(100):
            if time.time() > deadline - 5: break
            # Pick a random edge to remove
            u1, v1 = rng.choice([(i, j) for i in range(n) for j in range(i+1, n) if best_final_adj[i, j] == 1])
            # Pick a random non-edge to add
            u2, v2 = rng.choice([(i, j) for i in range(n) for j in range(i+1, n) if best_final_adj[i, j] == 0])
            
            # Perform swap
            best_final_adj[u1, v1] = 0
            best_final_adj[v1, u1] = 0
            best_final_adj[u2, v2] = 1
            best_final_adj[v2, u2] = 1
            
            new_c4 = count_c4_fast(best_final_adj)
            
            if new_c4 < current_c4:
                current_c4 = new_c4
                improved = True
            else:
                # Revert
                best_final_adj[u1, v1] = 1
                best_final_adj[v1, u1] = 1
                best_final_adj[u2, v2] = 0
                best_final_adj[v2, u2] = 0
    
    # Extract edges
    final_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_final_adj[i, j] == 1:
                final_edges.append([i, j])
                
    return final_edges

if __name__ == "__main__":
    edges = construct()
    print(f"Edges: {len(edges)}")
    adj = np.zeros((31, 31), dtype=np.int8)
    for u, v in edges:
        adj[u, v] = 1
        adj[v, u] = 1
    # Recount
    A = adj.astype(np.int32)
    common = A @ A
    mask = np.triu(np.ones((31, 31), dtype=bool), k=1)
    non_edge_mask = mask & (adj == 0)
    cv = common[non_edge_mask]
    total = np.sum(cv * (cv - 1) // 2)
    print(f"C4: {total // 2}")
