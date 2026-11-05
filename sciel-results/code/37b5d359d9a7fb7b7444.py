import numpy as np
import random

def construct():
    n = 16
    target_edges = 65
    # Initialize with K_{8,8} + 1 edge
    # Vertices 0-7 are Part A, 8-15 are Part B
    adj = np.zeros((n, n), dtype=int)
    for u in range(8):
        for v in range(8, 16):
            adj[u, v] = 1
            adj[v, u] = 1
    # Add one edge within Part A
    adj[0, 1] = 1
    adj[1, 0] = 1
    
    # Calculate current score
    def calc_score(adj_matrix):
        degs = np.sum(adj_matrix, axis=1)
        max_sum = 0
        # Iterate over all triangles
        # For efficiency, iterate over edges (u,v) and check common neighbors
        for u in range(n):
            for v in range(u+1, n):
                if adj_matrix[u, v]:
                    # Common neighbors
                    common = np.where(adj_matrix[u] & adj_matrix[v] > 0)[0]
                    for w in common:
                        if w > v: # Avoid duplicates, though w can be anything
                            s = degs[u] + degs[v] + degs[w]
                            if s > max_sum:
                                max_sum = s
        return max_sum

    current_score = calc_score(adj)
    current_edges = int(np.sum(adj) // 2)
    
    # Simulated Annealing
    best_adj = adj.copy()
    best_score = current_score
    
    # We want to minimize the max triangle degree sum.
    # Penalty for wrong edge count is huge.
    T = 10.0
    cooling = 0.999
    iterations = 50000
    rng = np.random.default_rng(42)
    
    for i in range(iterations):
        # Pick a random move: Add or Remove an edge
        # To maintain edge count, we can do swap moves or just accept edge count penalties
        # Let's do: pick a random non-edge to add, or random edge to remove.
        # If we add, we must remove another to keep count? 
        # It's easier to allow edge count fluctuation with a heavy penalty in the score.
        
        # Pick two distinct vertices u, v
        u = rng.integers(0, n)
        v = rng.integers(0, n)
        while u == v:
            v = rng.integers(0, n)
            
        # Flip edge (u,v)
        adj[u, v] ^= 1
        adj[v, u] ^= 1
        
        new_edges = int(np.sum(adj) // 2)
        
        # Calculate new score
        # To make this fast, we can update degrees and check affected triangles.
        # But for simplicity in this rewrite, and given n=16 is small, 
        # recalculating is fast enough if optimized.
        
        # Optimization: Only check triangles involving u or v
        degs = np.sum(adj, axis=1)
        
        # We need max triangle sum.
        # The change in max sum can only come from triangles involving u or v.
        # However, removing a triangle might lower the max if it WAS the max.
        # So we need to know the second max? 
        # Given n=16, total triangles is small. Let's just recalculate max sum for triangles involving u or v,
        # and keep a global max if no triangle involving u/v is the new max.
        # Actually, simplest robust way for n=16: Recalculate all. 
        # 120 edges, ~100 triangles. 50k iters * 100 ops = 5M ops. Fast.
        
        new_score = calc_score(adj)
        
        # Penalty for edge count
        edge_penalty = 100 * abs(new_edges - target_edges)
        
        total_new = new_score + edge_penalty
        total_old = current_score + 100 * abs(current_edges - target_edges)
        
        delta = total_new - total_old
        
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_score = new_score
            current_edges = new_edges
            if new_score < best_score and new_edges == target_edges:
                best_score = new_score
                best_adj = adj.copy()
        else:
            # Revert
            adj[u, v] ^= 1
            adj[v, u] ^= 1
            
        T *= cooling
        
        if best_score <= 23:
            # We found a very good solution, can break early or continue to polish
            # Let's continue to see if we can get 22
            pass

    # Ensure the best solution has exactly 65 edges
    # If best_adj doesn't have 65 edges, we might need to fix it.
    # The penalty should have kept it close.
    
    # Extract edges
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if best_adj[u, v]:
                edges.append([u, v])
                
    # If for some reason the edge count is wrong, adjust
    if len(edges) != target_edges:
        # Fallback to the seed if SA failed to converge to correct count
        # But the penalty is strong, so this is unlikely.
        # Let's just return what we have, assuming the penalty worked.
        pass
        
    return {'n': n, 'edges': edges}
