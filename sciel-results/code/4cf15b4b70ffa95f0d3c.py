"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin for 150s limit
    
    # Convert to float matrix for efficient matrix multiplication
    adj = np.asarray(adjacency, dtype=np.float64)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([])
    
    # Precompute features
    # 1. Degree: Lower degree is generally better for greedy MIS (blocks fewer nodes)
    degrees = adj.sum(axis=1)
    
    # 2. Average Neighbor Degree: 
    # If neighbors are high degree, they are "expensive" to block. 
    # We might want to pick nodes with LOW average neighbor degree to keep high-degree nodes available?
    # Or does it mean the node is in a dense core?
    # Let's define: avg_neigh_deg[i] = sum(degrees[N(i)]) / degrees[i]
    # Handle zero division
    avg_neigh_deg = np.zeros(n)
    mask = degrees > 0
    avg_neigh_deg[mask] = (adj @ degrees)[mask] / degrees[mask]
    
    # 3. Clustering Coefficient (approximate or exact for small N)
    # For N=200, we can compute exact clustering coefficient efficiently.
    # C[i] = 2 * triangles[i] / (degrees[i] * (degrees[i] - 1))
    # Triangles count: (A^3)[i,i] / 6
    # A^3 is O(N^3) which is 8M ops, very fast.
    try:
        # Compute A^2
        A2 = adj @ adj
        # Triangles = 0.5 * trace(A^3) per node? 
        # Actually, (A^3)[i,i] counts closed walks of length 3. 
        # Each triangle contributes 6 to the sum of diagonal elements of A^3? No.
        # For a triangle i-j-k-i, the walk i->j->k->i is one. 
        # A^3[i,i] = sum_j sum_k A[i,j]A[j,k]A[k,i].
        # This counts the number of triangles containing i, multiplied by 2 (directions)?
        # Let's just use the standard formula: num_triangles_i = (A @ A2)[i, i] / 2
        A3_diag = np.sum(adj * A2, axis=1) # This is sum_j A[i,j] * (A^2)[j,i] = sum_j A[i,j] * (A^2)[i,j]
        # Wait, A is symmetric. (A @ A2)[i, i] = sum_j A[i,j] * (A2)[j,i] = sum_j A[i,j] * (A2)[i,j].
        # (A2)[i,j] is number of common neighbors.
        # So A3_diag[i] is sum over neighbors j of #common_neighbors(i,j).
        # This is exactly 2 * number of triangles containing i.
        num_triangles = A3_diag / 2.0
        
        clustering = np.zeros(n)
        denom = degrees * (degrees - 1)
        mask_c = denom > 0
        clustering[mask_c] = num_triangles[mask_c] / denom[mask_c]
    except Exception:
        clustering = np.zeros(n)
    
    # Feature matrix: rows are vertices, columns are features
    # We want to find weights w such that score = X @ w
    # Let's normalize features to [0, 1] range to help optimization
    def normalize(f):
        mn, mx = np.min(f), np.max(f)
        if mx > mn:
            return (f - mn) / (mx - mn)
        return f
    
    deg_norm = normalize(degrees)
    avg_deg_norm = normalize(avg_neigh_deg)
    clus_norm = normalize(clustering)
    
    # Features: 
    # f0: -deg (we want low degree) -> use (1 - deg_norm)
    # f1: -avg_neigh_deg (we want low avg neigh deg?) -> use (1 - avg_deg_norm)
    # f2: -clustering (we want low clustering?) -> use (1 - clus_norm)
    # Let's just let the optimizer decide the signs and magnitudes.
    X = np.column_stack([deg_norm, avg_deg_norm, clus_norm])
    
    # Objective function: Given weights w, compute greedy MIS size
    def greedy_size(weights):
        scores = X @ weights
        # Sort indices by descending score
        order = np.argsort(-scores)
        selected = np.zeros(n, dtype=bool)
        count = 0
        for v in order:
            if not selected[v]:
                # Select v
                selected[v] = True
                count += 1
                # Mark neighbors as selected (blocked)
                neighbors = np.nonzero(adj[v])[0]
                selected[neighbors] = True
        return count
    
    # Simulated Annealing / Random Hill Climbing to find best weights
    # Initialize with a heuristic guess
    # Heuristic: Low degree is good. So weight for deg_norm should be negative.
    best_w = np.array([-1.0, 0.0, 0.0])
    best_score = greedy_size(best_w)
    
    current_w = best_w.copy()
    current_score = best_score
    
    # Temperature schedule
    T = 0.1
    T_min = 0.001
    cooling = 0.99
    
    steps = 0
    max_steps = 5000 # Each step is a greedy pass O(N^2) roughly. 5000 * 40k ops = 200M ops. Might be tight.
    # Let's estimate: N=200. Greedy pass is ~200 iterations, each checking 200 neighbors. 40k ops.
    # 5000 steps * 40k = 200M. In Python loop, this is too slow (10-20s per 100M ops in pure python, but numpy helps).
    # The inner loop `for v in order:` is pure Python. 200 iterations.
    # 5000 * 200 = 1M iterations of the inner loop.
    # Inside: `selected[v]`, `np.nonzero`, `selected[neighbors] = True`.
    # This should be fast enough if optimized.
    
    # Optimize the greedy_size function
    def greedy_size_fast(weights):
        scores = X @ weights
        order = np.argsort(-scores)
        selected = np.zeros(n, dtype=bool)
        count = 0
        # Vectorized selection is hard due to dependency.
        # But N is small.
        for v in order:
            if not selected[v]:
                count += 1
                selected[v] = True
                # Mark neighbors
                # adj[v] is a slice. 
                # We can use: selected[adj[v] > 0] = True
                # But adj is float. 
                # Let's precompute neighbor indices list for speed?
                # Or just use the boolean mask from adj.
                selected[adj[v] > 0.5] = True
        return count
    
    # Precompute neighbor masks as boolean arrays? 
    # adj_bool = adj > 0.5
    adj_bool = adj > 0.5
    
    def greedy_size_v2(weights):
        scores = X @ weights
        order = np.argsort(-scores)
        selected = np.zeros(n, dtype=bool)
        count = 0
        for v in order:
            if not selected[v]:
                count += 1
                selected[v] = True
                selected[adj_bool[v]] = True
        return count
    
    # Update best_score with v2
    best_score = greedy_size_v2(best_w)
    current_score = best_score
    
    for step in range(max_steps):
        if time.time() > deadline - 5.0:
            break
            
        # Perturb weights
        noise = np.random.normal(0, T, size=3)
        new_w = current_w + noise
        
        # Evaluate
        new_score = greedy_size_v2(new_w)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta > 0 or np.random.random() < np.exp(delta / T):
            current_w = new_w
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_w = current_w.copy()
        
        T = max(T_min, T * cooling)
        
    # Final polish: local search around best_w with smaller steps
    T = 0.01
    for step in range(2000):
        if time.time() > deadline - 2.0:
            break
        noise = np.random.normal(0, T, size=3)
        new_w = best_w + noise
        new_score = greedy_size_v2(new_w)
        if new_score > best_score:
            best_score = new_score
            best_w = new_w.copy()
        T = max(0.001, T * 0.99)
        
    # Compute final scores using best weights
    final_scores = X @ best_w
    
    return final_scores.astype(float)
# EVOLVE-BLOCK-END
