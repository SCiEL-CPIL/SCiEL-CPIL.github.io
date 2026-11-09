import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    n = 16
    target_edges = 65
    
    def make_adj(edges):
        adj = np.zeros((n, n), dtype=np.uint8)
        for u, v in edges:
            adj[u, v] = 1
            adj[v, u] = 1
        return adj
    
    def compute_degrees(adj):
        return adj.sum(axis=1)
    
    def score_graph(adj, degrees):
        """Compute the score: max degree sum over all triangles."""
        max_sum = 0
        # Iterate over all triples i < j < k
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j]:
                    for k in range(j + 1, n):
                        if adj[i, k] and adj[j, k]:
                            s = degrees[i] + degrees[j] + degrees[k]
                            if s > max_sum:
                                max_sum = s
        return max_sum

    def initial_bipartite():
        edges = []
        for u in range(8):
            for v in range(8, 16):
                edges.append([u, v])
        # Add one edge to make 65 edges and create triangles
        edges.append([0, 1])
        return edges

    def random_graph():
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_pairs)
        return [[u, v] for u, v in all_pairs[:target_edges]]

    best_score = float('inf')
    best_adj = None
    
    # We will run several restarts
    # Start with the known good bipartite structure, then random ones
    initial_configs = [initial_bipartite(), random_graph(), random_graph(), random_graph()]
    
    for init_idx, init_edges in enumerate(initial_configs):
        if time.time() > deadline:
            break
            
        adj = make_adj(init_edges)
        degrees = compute_degrees(adj)
        current_score = score_graph(adj, degrees)
        
        if current_score < best_score:
            best_score = current_score
            best_adj = adj.copy()
            
        # Local Search: Hill Climbing with all single-edge swaps
        # To speed up, we can precompute the list of present and absent edges
        # and iterate through them.
        
        improved = True
        while improved:
            improved = False
            if time.time() > deadline:
                break
                
            present_edges = []
            absent_edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[i, j]:
                        present_edges.append((i, j))
                    else:
                        absent_edges.append((i, j))
                        
            # Shuffle to avoid bias, or just iterate
            # For best improvement, we should check all.
            # 3500 checks * ~10us per check = 35ms per sweep. Very fast.
            
            best_swap_score = current_score
            best_swap_remove = None
            best_swap_add = None
            
            for rem in present_edges:
                for add in absent_edges:
                    # Check if swap is valid (it always is for simple graph swaps)
                    # Create temporary change
                    u, v = rem
                    a, b = add
                    
                    # We need to calculate new score efficiently.
                    # Modifying adj in-place and reverting is faster than copying.
                    # But score_graph needs degrees.
                    
                    # Let's just copy for simplicity first, if it's fast enough.
                    # 3500 copies of 16x16 array is cheap.
                    
                    new_adj = adj.copy()
                    new_adj[u, v] = 0
                    new_adj[v, u] = 0
                    new_adj[a, b] = 1
                    new_adj[b, a] = 1
                    
                    new_degrees = degrees.copy()
                    new_degrees[u] -= 1
                    new_degrees[v] -= 1
                    new_degrees[a] += 1
                    new_degrees[b] += 1
                    
                    new_score = score_graph(new_adj, new_degrees)
                    
                    if new_score < best_swap_score:
                        best_swap_score = new_score
                        best_swap_remove = rem
                        best_swap_add = add
                        
            if best_swap_remove is not None and best_swap_score < current_score:
                u, v = best_swap_remove
                a, b = best_swap_add
                adj[u, v] = 0
                adj[v, u] = 0
                adj[a, b] = 1
                adj[b, a] = 1
                
                degrees = compute_degrees(adj)
                current_score = best_swap_score
                improved = True
                
                if current_score < best_score:
                    best_score = current_score
                    best_adj = adj.copy()
                    
        # Add some random perturbation to escape local minima if stuck
        # Try a few random restarts from the current best if we have time
        # Or just rely on the initial multiple restarts.
        
    # Convert best_adj back to edge list
    best_edges = [[i, j] for i in range(n) for j in range(i + 1, n) if best_adj[i, j]]
    
    return {'n': n, 'edges': best_edges}
