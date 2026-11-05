"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import random

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a graph with 40 vertices and 401 edges minimizing the largest book.
    
    Approach:
    1. Initialize with a random graph having 401 edges.
    2. Local search: Identify edges with maximum book size.
    3. Attempt to swap a high-book edge with a non-edge that reduces the global maximum book size.
    4. Use random restarts/perturbations if stuck.
    """
    n = 40
    target_edges = 401
    max_iter = 20000
    random.seed(42)
    
    # Helper to compute max book size and the set of edges achieving it
    def get_max_book(adj):
        A2 = adj @ adj
        max_b = 0
        bad_edges = []
        for u in range(n):
            for v in range(u+1, n):
                if adj[u, v]:
                    b = A2[u, v]
                    if b > max_b:
                        max_b = b
                        bad_edges = [(u, v)]
                    elif b == max_b:
                        bad_edges.append((u, v))
        return max_b, bad_edges

    # Initialize random graph
    adj = np.zeros((n, n), dtype=np.int32)
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(all_pairs)
    for i in range(target_edges):
        u, v = all_pairs[i]
        adj[u, v] = 1
        adj[v, u] = 1
        
    current_max, bad_edges = get_max_book(adj)
    best_adj = adj.copy()
    best_max = current_max
    
    # Local Search
    for iter_num in range(max_iter):
        if current_max <= 6: # Theoretical lower bound is ~6.67, so 7 is likely min. 6 is impossible.
            break
            
        # Pick a random bad edge to remove
        u, v = random.choice(bad_edges)
        
        # Find all non-edges
        non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not adj[i, j]]
        
        if not non_edges:
            continue
            
        # Try to find a swap that improves the max book
        best_swap = None
        best_new_max = current_max
        
        # We can sample non-edges if there are too many, but for N=40, 
        # number of non-edges = 780 - 401 = 379. This is small enough to check all?
        # Checking all 379 swaps * 40x40 matmul might be slow in Python loops.
        # Let's optimize: 
        # Instead of full matmul for each candidate, we can estimate.
        # But N=40 is tiny. Let's try a smarter heuristic first:
        # Look for non-edges (w,x) with low A2[w][x] initially.
        
        A2 = adj @ adj
        
        # Sort non-edges by their current A2 value (potential book size if added)
        # This is a heuristic: adding an edge with low common neighbors is "safer"
        non_edges_sorted = sorted(non_edges, key=lambda e: A2[e[0]][e[1]])
        
        found_improvement = False
        # Check top 50 candidates to save time, or all if fast enough
        # 379 candidates * (40*40*2 mults) ~ 1.2M ops per candidate? No, matmul is C-optimized.
        # Let's just check the first 100 promising ones.
        for w, x in non_edges_sorted[:100]:
            # Simulate swap
            # Remove (u,v)
            adj[u, v] = 0
            adj[v, u] = 0
            # Add (w,x)
            adj[w, x] = 1
            adj[x, w] = 1
            
            new_max, _ = get_max_book(adj)
            
            # Restore
            adj[w, x] = 0
            adj[x, w] = 0
            adj[u, v] = 1
            adj[v, u] = 1
            
            if new_max < best_new_max:
                best_new_max = new_max
                best_swap = (w, x)
                if new_max < 7: # If we get very low, we can stop early
                    found_improvement = True
                    break
            elif new_max == best_new_max and best_swap is None:
                # Tie-breaking: accept if it changes the structure (optional)
                pass

        if best_swap:
            w, x = best_swap
            # Apply swap
            adj[u, v] = 0
            adj[v, u] = 0
            adj[w, x] = 1
            adj[x, w] = 1
            current_max = best_new_max
            if current_max < best_max:
                best_max = current_max
                best_adj = adj.copy()
            # Recalculate bad edges for next iteration
            _, bad_edges = get_max_book(adj)
        else:
            # Stuck? Perturb
            if iter_num % 50 == 49:
                # Random swap
                edge_list = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i, j]]
                if edge_list:
                    u2, v2 = random.choice(edge_list)
                    adj[u2, v2] = 0
                    adj[v2, u2] = 0
                    non_edges_now = [(i, j) for i in range(n) for j in range(i+1, n) if not adj[i, j]]
                    if non_edges_now:
                        w2, x2 = random.choice(non_edges_now)
                        adj[w2, x2] = 1
                        adj[x2, w2] = 1
                    current_max, bad_edges = get_max_book(adj)
                    if current_max < best_max:
                        best_max = current_max
                        best_adj = adj.copy()

    # Use best found
    adj = best_adj
    
    # Extract edges
    result = []
    for u in range(n):
        for v in range(u+1, n):
            if adj[u, v]:
                result.append([u, v])
                
    # Verify count
    if len(result) != target_edges:
         # Fallback
         result = [[i, j] for i in range(20) for j in range(20, 40)] + [[0, 1]]
         
    return result
# EVOLVE-BLOCK-END
