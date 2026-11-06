import time
import random
import numpy as np

def _paley(n, p):
    """Paley coloring using prime p >= n."""
    if p < n:
        return None
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            colors.append(0 if (j - i) % p in squares else 1)
    return colors

def _build_adj_np(colors, n):
    """Build adjacency matrices (boolean) from color list."""
    A0 = np.zeros((n, n), dtype=bool)
    A1 = np.zeros((n, n), dtype=bool)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if colors[idx] == 0:
                A0[i, j] = True
                A0[j, i] = True
            else:
                A1[i, j] = True
                A1[j, i] = True
            idx += 1
    return A0, A1

def _count_k222_np(A):
    """
    Count K2,2,2 subgraphs in graph defined by boolean adjacency matrix A.
    """
    n = A.shape[0]
    # M[i, j] = number of common neighbors of i and j
    M = A @ A.T
    
    total = 0
    # Iterate over non-edges (i, j)
    # A[i, j] == 0 means non-edge
    non_edges = np.argwhere(~A)
    # Exclude self-loops (i == j)
    non_edges = non_edges[non_edges[:, 0] < non_edges[:, 1]]
    
    for u, v in non_edges:
        # Common neighbors of u and v
        # mask = A[u] & A[v]
        # We need to count K2,2 in the subgraph induced by these common neighbors.
        # Let S be the set of common neighbors.
        # We need to count the number of K2,2 in A[S, S].
        # This is equivalent to summing over pairs (z, w) in S of C(common_neighbors(z, w) in S, 2) / 2?
        # Actually, number of K2,2 in H is 0.5 * sum_{z != w} C(M_H[z, w], 2)
        # where M_H is the common neighbor count within H.
        
        mask = A[u] & A[v]
        # Get indices of common neighbors
        verts = np.argwhere(mask)
        if len(verts) < 4:
            continue
        
        # Extract submatrix
        # This is slow if done with indexing every time.
        # But n=29, so it's small.
        sub_A = A[verts[:, 0], :][:, verts[:, 0]]
        sub_M = sub_A @ sub_A.T
        
        # Sum C(sub_M[z, w], 2) for all z < w
        # sub_M is symmetric.
        # We need sum over z < w of (sub_M[z, w] * (sub_M[z, w] - 1)) // 2
        # Note: sub_M[z, z] is degree, we ignore diagonal.
        
        # Create upper triangle indices
        iu, ju = np.triu_indices(len(verts), k=1)
        vals = sub_M[iu, ju]
        # C(k, 2) = k * (k - 1) // 2
        total += np.sum(vals * (vals - 1) // 2)
        
    return total

def _is_valid_np(A0, A1):
    return _count_k222_np(A0) == 0 and _count_k222_np(A1) == 0

def _simulated_annealing_np(colors, n, max_iters=100000, time_limit=60):
    start_time = time.time()
    A0, A1 = _build_adj_np(colors, n)
    current_cost = _count_k222_np(A0) + _count_k222_np(A1)
    
    if current_cost == 0:
        return colors, True
    
    edge_map = []
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_map.append((i, j))
            idx += 1
    num_edges = len(edge_map)
    
    T0 = 5.0
    T_min = 0.001
    T = T0
    best_cost = current_cost
    best_colors = colors[:]
    no_improve = 0
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        if no_improve > 2000:
            T = max(T0 * 0.3, T_min)
            no_improve = 0
        
        idx = random.randint(0, num_edges - 1)
        i, j = edge_map[idx]
        
        # Flip edge in A0 and A1
        in_color0 = A0[i, j]
        
        if in_color0:
            A0[i, j] = False
            A0[j, i] = False
            A1[i, j] = True
            A1[j, i] = True
            colors[idx] = 1
        else:
            A0[i, j] = True
            A0[j, i] = True
            A1[i, j] = False
            A1[j, i] = False
            colors[idx] = 0
        
        new_cost = _count_k222_np(A0) + _count_k222_np(A1)
        delta = new_cost - current_cost
        
        if delta <= 0:
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_colors = colors[:]
                no_improve = 0
            else:
                no_improve += 1
        else:
            if T > T_min:
                if random.random() < np.exp(-delta / T):
                    current_cost = new_cost
                    no_improve += 1
                else:
                    # Revert
                    if in_color0:
                        A0[i, j] = True
                        A0[j, i] = True
                        A1[i, j] = False
                        A1[j, i] = False
                        colors[idx] = 0
                    else:
                        A0[i, j] = False
                        A0[j, i] = False
                        A1[i, j] = True
                        A1[j, i] = True
                        colors[idx] = 1
            else:
                # Revert
                if in_color0:
                    A0[i, j] = True
                    A0[j, i] = True
                    A1[i, j] = False
                    A1[j, i] = False
                    colors[idx] = 0
                else:
                    A0[i, j] = False
                    A0[j, i] = False
                    A1[i, j] = True
                    A1[j, i] = True
                    colors[idx] = 1
        
        T *= 0.9995
        
        if current_cost == 0:
            return colors, True
    
    return best_colors, best_cost == 0

def construct():
    deadline = time.time() + 145
    target_n = 29
    p = 29
    
    colors = _paley(target_n, p)
    if colors:
        A0, A1 = _build_adj_np(colors, target_n)
        if _is_valid_np(A0, A1):
            return {'n': target_n, 'colors': colors}
    
    # SA with NumPy
    if time.time() < deadline - 40:
        colors_sa = colors[:]
        _, success = _simulated_annealing_np(colors_sa, target_n, max_iters=200000, time_limit=50)
        if success:
            return {'n': target_n, 'colors': colors_sa}
    
    # Multi-restart
    for attempt in range(30):
        if time.time() > deadline - 15:
            break
        colors_rand = [random.randint(0, 1) for _ in range(target_n * (target_n - 1) // 2)]
        if attempt % 2 == 0 and colors:
            colors_rand = colors[:]
            for _ in range(random.randint(5, 20)):
                idx = random.randint(0, len(colors_rand) - 1)
                colors_rand[idx] = 1 - colors_rand[idx]
        
        _, success = _simulated_annealing_np(colors_rand, target_n, max_iters=80000, time_limit=20)
        if success:
            return {'n': target_n, 'colors': colors_rand}
            
    # Fallback
    return {'n': 1, 'colors': [0]}
