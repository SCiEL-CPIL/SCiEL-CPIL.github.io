"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    n = 16
    target_edges = 65
    
    # Precompute all possible triangles
    all_triangles = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                all_triangles.append((i, j, k))
                
    def compute_score(adj, deg):
        """Compute the maximum triangle degree sum given adjacency matrix and degrees."""
        max_sum = 0
        for u, v, w in all_triangles:
            if adj[u][v] and adj[v][w] and adj[u][w]:
                s = deg[u] + deg[v] + deg[w]
                if s > max_sum:
                    max_sum = s
        return max_sum
    
    def is_valid_adj(adj):
        """Check if adjacency matrix represents a simple graph with exactly target_edges edges."""
        # Since we start valid and only swap edges, this is mostly for initial check
        edge_count = 0
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j]:
                    edge_count += 1
                    if edge_count > target_edges:
                        return False
        return edge_count == target_edges

    # Start with the baseline construction: K_{8,8} + one edge (0,1)
    # Vertices 0-7 and 8-15
    adj = [[False] * n for _ in range(n)]
    deg = [0] * n
    
    for u in range(8):
        for v in range(8, 16):
            adj[u][v] = True
            adj[v][u] = True
            deg[u] += 1
            deg[v] += 1
            
    # Add one edge inside part 0
    adj[0][1] = True
    adj[1][0] = True
    deg[0] += 1
    deg[1] += 1
    
    # Verify edge count
    edge_count = sum(1 for i in range(n) for j in range(i+1, n) if adj[i][j])
    if edge_count != target_edges:
        # Fallback if construction logic fails
        return {'n': n, 'edges': [[0, 1]]} # This won't happen, but safe
        
    current_score = compute_score(adj, deg)
    best_score = current_score
    best_adj = [row[:] for row in adj]
    best_deg = deg[:]
    
    # Local Search
    improved = True
    iters = 0
    
    while improved and time.time() < deadline - 5.0:
        improved = False
        iters += 1
        
        # Collect current edges and non-edges
        edges = []
        non_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j]:
                    edges.append((i, j))
                else:
                    non_edges.append((i, j))
                    
        # Try all single edge swaps
        for e1 in edges:
            u1, v1 = e1
            for e2 in non_edges:
                u2, v2 = e2
                
                # Simulate swap: remove (u1, v1), add (u2, v2)
                # Update degrees
                deg[u1] -= 1
                deg[v1] -= 1
                deg[u2] += 1
                deg[v2] += 1
                
                # Update adjacency
                adj[u1][v1] = False
                adj[v1][u1] = False
                adj[u2][v2] = True
                adj[v2][u2] = True
                
                new_score = compute_score(adj, deg)
                
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    # Keep the change
                    
                    if current_score < best_score:
                        best_score = current_score
                        best_adj = [row[:] for row in adj]
                        best_deg = deg[:]
                    
                    # Restart search from new state to avoid local minima loops?
                    # Or continue? Restarting is safer for "first improvement" strategy.
                    # Let's break out to restart the edge/non-edge collection
                    # and continue the while loop.
                    break
                else:
                    # Roll back
                    adj[u1][v1] = True
                    adj[v1][u1] = True
                    adj[u2][v2] = False
                    adj[v2][u2] = False
                    
                    deg[u1] += 1
                    deg[v1] += 1
                    deg[u2] -= 1
                    deg[v2] -= 1
            
            if improved:
                break
                
    # Convert best_adj back to edge list
    best_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj[i][j]:
                best_edges.append([i, j])
                
    return {'n': n, 'edges': best_edges}
# EVOLVE-BLOCK-END
