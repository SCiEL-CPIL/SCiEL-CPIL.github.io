"""
Optimized construction for the triangle degree-sum minimization task.

Strategy:
1. Start from the Turán graph T_2(16) = K_{8,8} (64 edges) plus one intra-part edge.
2. Use local search (edge swaps) to minimize the maximum triangle degree sum.
   - An edge swap replaces one edge (u,v) with (x,y) where (u,v) is not in the graph
     and (x,y) is in the graph, maintaining the edge count of 65.
   - We focus on swaps that might reduce the maximum triangle degree sum.
3. Also try different initial intra-part edges.

The key insight: in K_{8,8}, all vertices have degree 8. Adding one edge (say (0,1) in part A)
gives deg(0)=9, deg(1)=9, others=8. The only triangles are those containing the edge (0,1)
and a common neighbor in part B. Each such triangle has vertices {0,1,v} for v in B, with
degree sum 9+9+8=26. So the score is 26.

To improve, we can try to balance degrees better. If we add an edge within part A, both endpoints
get degree 9. The triangles formed are (0,1,v) for v in B, each with sum 9+9+8=26.

What if we use a different structure? Consider a near-regular graph with 65 edges.
Average degree = 2*65/16 = 8.125. So most vertices have degree 8, some have degree 9.

If we can arrange that the triangles have vertices with lower degree sums, we can improve.
For example, if a triangle has degrees (8,8,8), sum=24. If (8,8,9), sum=25. If (8,9,9), sum=26.

So the ideal is to have triangles with all three vertices of degree 8, giving sum 24.

Let's try a different approach: start with a random graph with 65 edges and use simulated annealing
or local search to minimize the maximum triangle degree sum.
"""

import time
import random
import numpy as np

def _compute_score(adj, n):
    """Compute the maximum triangle degree sum."""
    max_sum = 0
    deg = [0] * n
    for i in range(n):
        for j in range(n):
            deg[i] += adj[i][j]
    
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                for k in range(j+1, n):
                    if adj[i][k] and adj[j][k]:
                        s = deg[i] + deg[j] + deg[k]
                        if s > max_sum:
                            max_sum = s
    return max_sum

def _compute_score_fast(adj, n):
    """Faster score computation using numpy."""
    deg = adj.sum(axis=1)
    max_sum = 0
    # Find triangles
    for i in range(n):
        for j in range(i+1, n):
            if adj[i, j]:
                common = np.where(adj[i, j+1:] & adj[j, j+1:])[0]
                for k_idx in common:
                    k = k_idx + j + 1
                    s = deg[i] + deg[j] + deg[k]
                    if s > max_sum:
                        max_sum = s
    return max_sum

def _edge_swap(adj, n, u, v, x, y):
    """Remove edge (u,v) and add edge (x,y). Return new adj and whether valid."""
    if not adj[u, v] or adj[x, y]:
        return None
    new_adj = adj.copy()
    new_adj[u, v] = 0
    new_adj[v, u] = 0
    new_adj[x, y] = 1
    new_adj[y, x] = 1
    return new_adj

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 16
    target_edges = 65
    
    best_adj = None
    best_score = float('inf')
    
    # Try multiple initial configurations
    random.seed(42)
    
    # Initial: K_{8,8} + one edge in part A
    # Part A: 0-7, Part B: 8-15
    adj = np.zeros((n, n), dtype=np.uint8)
    for i in range(8):
        for j in range(8, 16):
            adj[i, j] = 1
            adj[j, i] = 1
    # Add one edge in part A
    adj[0, 1] = 1
    adj[1, 0] = 1
    
    score = _compute_score_fast(adj, n)
    best_adj = adj.copy()
    best_score = score
    
    # Local search: try edge swaps
    # Strategy: try to swap an intra-part edge with an inter-part edge,
    # or swap between different intra-part edges
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # List of current edges
    def get_edges(adj):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i, j]:
                    edges.append((i, j))
        return edges
    
    # Local search loop
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_edges = get_edges(adj)
        current_score = _compute_score_fast(adj, n)
        
        # Try swaps: remove one edge, add another
        for rem_i, rem_j in current_edges:
            if time.time() >= deadline:
                break
            for add_i, add_j in all_edges:
                if (add_i, add_j) == (rem_i, rem_j):
                    continue
                if adj[add_i, add_j]:
                    continue
                
                new_adj = _edge_swap(adj, n, rem_i, rem_j, add_i, add_j)
                if new_adj is None:
                    continue
                
                new_score = _compute_score_fast(new_adj, n)
                if new_score < current_score:
                    adj = new_adj
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_adj = adj.copy()
                    improved = True
                    break
            if improved:
                break
    
    # Also try a few different starting points
    for trial in range(5):
        if time.time() >= deadline:
            break
        
        adj = np.zeros((n, n), dtype=np.uint8)
        for i in range(8):
            for j in range(8, 16):
                adj[i, j] = 1
                adj[j, i] = 1
        
        # Try different intra-part edges
        intra_a = [(i, j) for i in range(8) for j in range(i+1, 8)]
        intra_b = [(8+i, 8+j) for i in range(8) for j in range(i+1, 8)]
        
        random.shuffle(intra_a)
        random.shuffle(intra_b)
        
        # Try a few random intra-part edges
        for _ in range(3):
            if time.time() >= deadline:
                break
            
            if random.random() < 0.5:
                u, v = intra_a[random.randint(0, len(intra_a)-1)]
            else:
                u, v = intra_b[random.randint(0, len(intra_b)-1)]
            
            adj[u, v] = 1
            adj[v, u] = 1
            
            score = _compute_score_fast(adj, n)
            if score < best_score:
                best_score = score
                best_adj = adj.copy()
            
            # Local search from this starting point
            improved = True
            while improved and time.time() < deadline:
                improved = False
                current_edges = get_edges(adj)
                current_score = _compute_score_fast(adj, n)
                
                # Try a subset of swaps for speed
                for rem_i, rem_j in random.sample(current_edges, min(10, len(current_edges))):
                    if time.time() >= deadline:
                        break
                    for add_i, add_j in random.sample(all_edges, min(20, len(all_edges))):
                        if (add_i, add_j) == (rem_i, rem_j):
                            continue
                        if adj[add_i, add_j]:
                            continue
                        
                        new_adj = _edge_swap(adj, n, rem_i, rem_j, add_i, add_j)
                        if new_adj is None:
                            continue
                        
                        new_score = _compute_score_fast(new_adj, n)
                        if new_score < current_score:
                            adj = new_adj
                            current_score = new_score
                            if current_score < best_score:
                                best_score = current_score
                                best_adj = adj.copy()
                            improved = True
                            break
                    if improved:
                        break
    
    # Convert best_adj to edge list
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i, j]:
                edges.append([i, j])
    
    return {'n': n, 'edges': edges}
