import numpy as np
import random
import time

def construct():
    n = 16
    # Precompute all triplets
    triplets_list = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                triplets_list.append((i, j, k))
    triplets_arr = np.array(triplets_list, dtype=np.int32)
    ti, tj, tk = triplets_arr[:, 0], triplets_arr[:, 1], triplets_arr[:, 2]
    
    # All possible edges
    all_edges = []
    for i in range(n):
        for j in range(i+1, n):
            all_edges.append((i, j))
    
    def build_A(part_a, part_b, edges_in_b):
        """Build adjacency matrix from partition and edges in B."""
        A = np.zeros((n, n), dtype=np.uint8)
        for u in part_a:
            for v in part_b:
                A[u, v] = 1
                A[v, u] = 1
        for u, v in edges_in_b:
            A[u, v] = 1
            A[v, u] = 1
        return A
    
    def get_score(A_mat):
        """Vectorized score computation."""
        deg = A_mat.sum(axis=1).astype(np.int32)
        is_tri = A_mat[ti, tj] & A_mat[tj, tk] & A_mat[ti, tk]
        deg_sums = deg[ti] + deg[tj] + deg[tk]
        deg_sums = np.where(is_tri, deg_sums, 0)
        return int(deg_sums.max())
    
    # Try multiple initial constructions
    A1 = build_A([0,1,2,3,4,5], [6,7,8,9,10,11,12,13,14,15],
                 [(6,7),(8,9),(10,11),(12,13),(14,15)])
    A2 = build_A([0,1,2,3,4,5,6], [7,8,9,10,11,12,13,14,15],
                 [(7,8),(9,10)])
    A3 = build_A([0,1,2,3,4], [5,6,7,8,9,10,11,12,13,14,15],
                 [(5,6),(7,8),(9,10),(11,12),(13,14),(5,7),(6,9),(8,11),(10,13),(12,15)])
    A4 = build_A([0,1,2,3,4,5,6,7], [8,9,10,11,12,13,14,15],
                 [(8,9)])
    
    candidates = [A1, A2, A3, A4]
    scores = [get_score(A) for A in candidates]
    
    best_idx = int(np.argmin(scores))
    best_A = candidates[best_idx].copy()
    best_score = scores[best_idx]
    current_A = best_A.copy()
    current_score = best_score
    
    start_time = time.time()
    deadline = start_time + 115
    
    # Quick local search on best initial
    for _ in range(300):
        if time.time() > deadline - 60:
            break
        improved = False
        for i in range(n):
            for j in range(i+1, n):
                if current_A[i, j]:
                    for x in range(n):
                        for y in range(x+1, n):
                            if not current_A[x, y] and (x,y) != (i,j):
                                current_A[i, j] = 0
                                current_A[j, i] = 0
                                current_A[x, y] = 1
                                current_A[y, x] = 1
                                
                                new_score = get_score(current_A)
                                
                                if new_score < current_score:
                                    current_score = new_score
                                    improved = True
                                    if new_score < best_score:
                                        best_score = new_score
                                        best_A = current_A.copy()
                                else:
                                    current_A[i, j] = 1
                                    current_A[j, i] = 1
                                    current_A[x, y] = 0
                                    current_A[y, x] = 0
        if not improved:
            break
    
    # Simulated annealing
    T = 3.0
    cooling = 0.9999
    
    current_edge_set = set()
    for i in range(n):
        for j in range(i+1, n):
            if current_A[i, j]:
                current_edge_set.add((i, j))
    
    iter_count = 0
    while time.time() < deadline - 10:
        u, v = random.choice(all_edges)
        if (u, v) not in current_edge_set:
            continue
            
        x, y = random.choice(all_edges)
        if (x, y) in current_edge_set:
            continue
            
        if (u, v) == (x, y):
            continue
            
        current_A[u, v] = 0
        current_A[v, u] = 0
        current_A[x, y] = 1
        current_A[y, x] = 1
        current_edge_set.discard((u, v))
        current_edge_set.add((x, y))
        
        new_score = get_score(current_A)
        
        if new_score <= current_score or (new_score > current_score and random.random() < np.exp(-(new_score - current_score) / T)):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_A = current_A.copy()
        else:
            current_A[u, v] = 1
            current_A[v, u] = 1
            current_A[x, y] = 0
            current_A[y, x] = 0
            current_edge_set.discard((x, y))
            current_edge_set.add((u, v))
            
        T *= cooling
        iter_count += 1
    
    # Final local polish
    for _ in range(100):
        if time.time() > deadline - 2:
            break
        improved = False
        for i in range(n):
            for j in range(i+1, n):
                if best_A[i, j]:
                    for x in range(n):
                        for y in range(x+1, n):
                            if not best_A[x, y] and (x,y) != (i,j):
                                best_A[i, j] = 0
                                best_A[j, i] = 0
                                best_A[x, y] = 1
                                best_A[y, x] = 1
                                
                                new_score = get_score(best_A)
                                
                                if new_score < best_score:
                                    best_score = new_score
                                    improved = True
                                else:
                                    best_A[i, j] = 1
                                    best_A[j, i] = 1
                                    best_A[x, y] = 0
                                    best_A[y, x] = 0
        if not improved:
            break
    
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_A[i, j]:
                edges.append([i, j])
                
    return {'n': n, 'edges': edges}
