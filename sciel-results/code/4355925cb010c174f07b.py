"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # 10s safety margin before 150s kill
    
    def is_k33_free(adj_matrix):
        """Check if graph has no K3,3 subgraph."""
        n = len(adj_matrix)
        # For every pair of vertices (u, v), count common neighbors
        # If any 3 vertices share 3+ common neighbors, we have K3,3
        # More precisely: K3,3 exists iff there are 3 vertices with 3 common neighbors
        # OR equivalently, we can check: for every subset of 3 vertices, 
        # the number of common neighbors must be < 3
        
        # Efficient check: for each pair (u,v), compute common neighbors
        # Then for each triple of common neighbors... that's expensive
        # Alternative: K3,3-free iff for every 3 vertices, they have at most 2 common neighbors
        # Let's use a different approach: check all pairs of vertices
        # For each pair (u,v), let C(u,v) = common neighbors
        # K3,3 exists iff there exist 3 vertices a,b,c such that |N(a) ∩ N(b) ∩ N(c)| >= 3
        # This is equivalent to: there exist 3 vertices with 3 common neighbors
        
        # More practical: for each triple of vertices, check common neighbors
        # But that's O(n^3 * n) which for n=20 is 8000*20 = 160k, acceptable
        
        # Actually, a faster check: 
        # K3,3-free means no two sets of 3 vertices have all 9 edges between them
        # Let's just check all triples of vertices for common neighbors >= 3
        
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    # Count common neighbors of i, j, k
                    common = 0
                    for m in range(n):
                        if m != i and m != j and m != k:
                            if adj_matrix[i][m] and adj_matrix[j][m] and adj_matrix[k][m]:
                                common += 1
                                if common >= 3:
                                    return False
        return True
    
    def count_edges(adj_matrix):
        return sum(adj_matrix[i][j] for i in range(len(adj_matrix)) for j in range(i+1, len(adj_matrix)))
    
    # Start with a known valid graph: complete bipartite K2,18
    n = 20
    adj = np.zeros((n, n), dtype=bool)
    for a in range(2):
        for b in range(2, 20):
            adj[a][b] = True
            adj[b][a] = True
    
    best_edges = count_edges(adj)
    best_adj = adj.copy()
    
    # Try to add edges greedily
    all_possible = [(i, j) for i in range(n) for j in range(i+1, n) if not adj[i][j]]
    
    # Shuffle for randomization
    random.shuffle(all_possible)
    
    for u, v in all_possible:
        if time.time() > deadline - 5:
            break
        # Try adding edge (u, v)
        adj[u][v] = True
        adj[v][u] = True
        
        if is_k33_free(adj):
            new_count = count_edges(adj)
            if new_count > best_edges:
                best_edges = new_count
                best_adj = adj.copy()
        else:
            adj[u][v] = False
            adj[v][u] = False
    
    # Try local search: swap edges
    if time.time() < deadline - 10:
        # Create list of current edges
        current_edges = [(i, j) for i in range(n) for j in range(i+1, n) if best_adj[i][j]]
        missing_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not best_adj[i][j]]
        
        random.shuffle(current_edges)
        random.shuffle(missing_edges)
        
        for _ in range(min(500, len(current_edges) * len(missing_edges) // 10)):
            if time.time() > deadline - 5:
                break
            if not current_edges or not missing_edges:
                break
            
            e1 = random.choice(current_edges)
            e2 = random.choice(missing_edges)
            
            # Try swapping: remove e1, add e2
            u1, v1 = e1
            u2, v2 = e2
            best_adj[u1][v1] = False
            best_adj[v1][u1] = False
            best_adj[u2][v2] = True
            best_adj[v2][u2] = True
            
            if is_k33_free(best_adj):
                # Accept
                current_edges.remove(e1)
                current_edges.append(e2)
                missing_edges.remove(e2)
                missing_edges.append(e1)
            else:
                # Revert
                best_adj[u1][v1] = True
                best_adj[v1][u1] = True
                best_adj[u2][v2] = False
                best_adj[v2][u2] = False
    
    # Convert to list of edges
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i][j]:
                result.append([i, j])
    
    return result
# EVOLVE-BLOCK-END
