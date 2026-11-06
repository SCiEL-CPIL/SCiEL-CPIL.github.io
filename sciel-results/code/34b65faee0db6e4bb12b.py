import numpy as np
import random
import time

def construct():
    n = 16
    # Initial construction: K_{6,10} + 5 disjoint edges in part B
    # Part A: 0-5, Part B: 6-15
    A = np.zeros((n, n), dtype=np.uint8)
    
    # K_{6,10}
    for u in range(6):
        for v in range(6, 16):
            A[u, v] = 1
            A[v, u] = 1
            
    # 5 disjoint edges in B
    edges_b = [(6, 7), (8, 9), (10, 11), (12, 13), (14, 15)]
    for u, v in edges_b:
        A[u, v] = 1
        A[v, u] = 1
        
    # Precompute all triplets as numpy arrays for vectorized scoring
    triplets_list = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                triplets_list.append((i, j, k))
    triplets_arr = np.array(triplets_list, dtype=np.int32)  # shape (416, 3)
    ti, tj, tk = triplets_arr[:, 0], triplets_arr[:, 1], triplets_arr[:, 2]
    
    # Precompute all possible edges as a list for fast random selection
    all_edges = []
    for i in range(n):
        for j in range(i+1, n):
            all_edges.append((i, j))
    num_edges = len(all_edges)  # 120
    
    def get_score(A_mat):
        """Vectorized score computation."""
        deg = A_mat.sum(axis=1).astype(np.int32)
        # For each triplet, check if it's a triangle and compute degree sum
        is_tri = A_mat[ti, tj] & A_mat[tj, tk] & A_mat[ti, tk]
        deg_sums = deg[ti] + deg[tj] + deg[tk]
        # Set non-triangle degree sums to 0
        deg_sums = np.where(is_tri, deg_sums, 0)
        return int(deg_sums.max())

    best_A = A.copy()
    best_score = get_score(best_A)
    current_A = A.copy()
    current_score = best_score
    
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    T = 1.0
    cooling = 0.9997
    
    # Track current edges as a set for O(1) membership
    current_edge_set = set()
    for i in range(n):
        for j in range(i+1, n):
            if current_A[i, j]:
                current_edge_set.add((i, j))
    
    # Track non-edges for fast random non-edge selection
    current_nonedge_set = set()
    for i in range(n):
        for j in range(i+1, n):
            if not current_A[i, j]:
                current_nonedge_set.add((i, j))
    
    iter_count = 0
    while time.time() < deadline:
        # Pick random edge to remove
        u, v = random.choice(all_edges)
        if (u, v) not in current_edge_set:
            continue
            
        # Pick random non-edge to add
        x, y = random.choice(all_edges)
        if (x, y) in current_edge_set:
            continue
            
        # Ensure distinct
        if (u, v) == (x, y):
            continue
            
        # Apply move
        current_A[u, v] = 0
        current_A[v, u] = 0
        current_A[x, y] = 1
        current_A[y, x] = 1
        current_edge_set.discard((u, v))
        current_edge_set.add((x, y))
        current_nonedge_set.discard((x, y))
        current_nonedge_set.add((u, v))
        
        new_score = get_score(current_A)
        
        # Acceptance
        if new_score < current_score or (new_score > current_score and random.random() < np.exp(-(new_score - current_score) / T)):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_A = current_A.copy()
        else:
            # Revert
            current_A[u, v] = 1
            current_A[v, u] = 1
            current_A[x, y] = 0
            current_A[y, x] = 0
            current_edge_set.discard((x, y))
            current_edge_set.add((u, v))
            current_nonedge_set.discard((u, v))
            current_nonedge_set.add((x, y))
            
        T *= cooling
        iter_count += 1
        
    # Reconstruct edges list from best_A
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_A[i, j]:
                edges.append([i, j])
                
    return {'n': n, 'edges': edges}
