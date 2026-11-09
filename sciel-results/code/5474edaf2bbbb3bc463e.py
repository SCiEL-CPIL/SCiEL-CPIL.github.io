# EVOLVE-BLOCK-START
import numpy as np
import itertools
import random

def construct():
    n = 16
    # Q3 adjacency matrix (8x8)
    q3_adj = np.zeros((8, 8), dtype=int)
    edges_q3 = [(0,1), (0,2), (0,4), (1,3), (1,5), (2,3), (2,6), (3,7), (4,5), (4,6), (5,7), (6,7)]
    for u, v in edges_q3:
        q3_adj[u, v] = 1
        q3_adj[v, u] = 1
        
    # Precompute neighbors for Q3
    q3_neighbors = [[] for _ in range(8)]
    for i in range(8):
        for j in range(8):
            if q3_adj[i, j] == 1:
                q3_neighbors[i].append(j)

    def check_q3_in_subset(sub_adj):
        # Backtracking check for Q3 subgraph in 8x8 sub_adj
        mapping = [-1] * 8
        used = [False] * 8
        
        def backtrack(q3_idx):
            if q3_idx == 8:
                return True
            for subset_idx in range(8):
                if used[subset_idx]:
                    continue
                valid = True
                for prev_q3_idx in range(q3_idx):
                    if q3_adj[q3_idx, prev_q3_idx] == 1:
                        if sub_adj[subset_idx, mapping[prev_q3_idx]] == 0:
                            valid = False
                            break
                if valid:
                    mapping[q3_idx] = subset_idx
                    used[subset_idx] = True
                    if backtrack(q3_idx + 1):
                        return True
                    used[subset_idx] = False
                    mapping[q3_idx] = -1
            return False
            
        return backtrack(0)

    def contains_q3_using_edge(adj, u, v):
        # Check if adding (u,v) creates a Q3.
        # Only check 8-subsets containing u and v.
        # There are C(14, 6) = 3003 such subsets.
        other_vertices = [i for i in range(n) if i != u and i != v]
        
        for subset6 in itertools.combinations(other_vertices, 6):
            # Form the 8-vertex set
            vertices = [u, v] + list(subset6)
            
            # Extract sub_adj
            # Map u, v to indices 0, 1 in the local 8x8 matrix?
            # No, just use numpy ix_
            sub_adj = adj[np.ix_(vertices, vertices)]
            
            # Fast checks
            edge_count = np.sum(sub_adj) // 2
            if edge_count < 12:
                continue
            degrees = np.sum(sub_adj, axis=1)
            if np.min(degrees) < 3:
                continue
            
            # Check if Q3 is a subgraph
            if check_q3_in_subset(sub_adj):
                return True
                
        return False

    # Greedy Construction
    adj = np.zeros((n, n), dtype=int)
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(all_edges)
    
    for u, v in all_edges:
        adj[u, v] = 1
        adj[v, u] = 1
        
        if contains_q3_using_edge(adj, u, v):
            adj[u, v] = 0
            adj[v, u] = 0

    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i, j] == 1:
                edges.append([i, j])
                
    return edges
# EVOLVE-BLOCK-END
