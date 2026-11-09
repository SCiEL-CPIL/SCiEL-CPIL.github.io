import numpy as np
import random
import time

N = 31
E = 91
DEADLINE_SECONDS = 115.0

def count_c4(adj):
    """
    Count C4s in graph represented by adjacency matrix.
    """
    A2 = adj @ adj
    # A2[i,j] = number of common neighbors
    # Number of C4s = 1/2 * sum_{i<j} C(A2[i,j], 2)
    # We only need to sum over i < j
    # np.triu_indices
    i, j = np.triu_indices(N, k=1)
    common_counts = A2[i, j]
    # Calculate C(k, 2) = k * (k-1) / 2
    terms = common_counts * (common_counts - 1) // 2
    total = np.sum(terms)
    return total // 2

def construct():
    start_time = time.time()
    best_edges = None
    best_score = float('inf') # We want to minimize C4 count
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    # Try multiple random restarts
    # Since N is small, we can do many iterations per restart
    max_restarts = 20
    
    for r in range(max_restarts):
        if time.time() - start_time > DEADLINE_SECONDS - 10:
            break
            
        # Initialize random graph
        edges = random.sample(all_edges, E)
        edge_set = set(edges)
        adj = np.zeros((N, N), dtype=int)
        for u, v in edges:
            adj[u, v] = 1
            adj[v, u] = 1
            
        current_c4 = count_c4(adj)
        
        # Local Search (Hill Climbing)
        # We will perform a fixed number of iterations or until time limit
        # Estimated time per iteration: ~1ms (conservative)
        # Let's run as many as we can
        iterations = 0
        while time.time() - start_time < DEADLINE_SECONDS - 5:
            # Pick a random edge to remove
            u1, v1 = random.choice(edges)
            
            # Pick a random non-edge to add
            # To avoid checking if it's in edge_set repeatedly, 
            # we can pick from a list of non-edges or just pick and check.
            # With N=31, 465 total edges, 91 in graph.
            # Probability of picking a non-edge is (465-91)/465 ~ 0.8.
            # So just picking random and checking is fine.
            u2, v2 = random.choice(all_edges)
            if (u2, v2) in edge_set or (v2, u2) in edge_set:
                continue
                
            # Perform swap
            # Remove (u1, v1)
            adj[u1, v1] = 0
            adj[v1, u1] = 0
            edge_set.remove((u1, v1))
            
            # Add (u2, v2)
            adj[u2, v2] = 1
            adj[v2, u2] = 1
            edge_set.add((u2, v2))
            
            # Update edges list (replace u1,v1 with u2,v2)
            # Instead of modifying list, just use edge_set for validity, 
            # but we need a list for random choice.
            # We can just regenerate the list or use a separate structure.
            # For speed, let's just keep `edges` as a list and swap elements.
            # Find index of (u1, v1)
            idx = edges.index((u1, v1))
            edges[idx] = (u2, v2)
            
            new_c4 = count_c4(adj)
            
            if new_c4 < current_c4:
                current_c4 = new_c4
                if current_c4 < best_score:
                    best_score = current_c4
                    best_edges = edges[:]
                # If we hit 1, we can stop this restart? 
                # Theoretical min is 1.
                if current_c4 == 1:
                    break
            else:
                # Revert
                adj[u2, v2] = 0
                adj[v2, u2] = 0
                edge_set.remove((u2, v2))
                
                adj[u1, v1] = 1
                adj[v1, u1] = 1
                edge_set.add((u1, v1))
                
                edges[idx] = (u1, v1)
                
            iterations += 1
            # Optional: Simulated Annealing acceptance
            # If new_c4 > current_c4, accept with probability exp(-delta/T)
            # But for now, simple hill climbing with restarts is robust.
            
    # Ensure best_edges is valid
    if best_edges is None:
        # Fallback
        best_edges = random.sample(all_edges, E)
        
    return best_edges
