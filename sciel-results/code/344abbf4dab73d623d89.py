import numpy as np
import time
from scipy.optimize import minimize
from scipy.sparse.csgraph import shortest_path
import warnings

def priority(adjacency):
    start_time = time.time()
    DEADLINE = 140.0  # 10s safety margin

    A = np.asarray(adjacency, dtype=float)
    N = A.shape[0]

    if N == 0:
        return np.array([])

    # Ensure symmetric
    A = (A + A.T) / 2.0
    np.fill_diagonal(A, 0.0)
    A_bool = A > 0.5

    # Precompute adjacency rows for fast evaluation
    nbrs = [np.where(A_bool[i])[0] for i in range(N)]
    D = np.array([len(nbrs[i]) for i in range(N)], dtype=float)
    
    # Features
    # 1. Degree
    F_deg = D.copy()
    
    # 2. Sum of Neighbor Degrees
    F_snd = np.array([np.sum(D[nbrs[i]]) if len(nbrs[i]) > 0 else 0 for i in range(N)])
    
    # 3. Block Cost: Sum of 1/(1+deg[neighbor])
    F_block_cost = np.array([np.sum(1.0 / (1.0 + D[nbrs[i]])) if len(nbrs[i]) > 0 else 0 for i in range(N)])
    
    # 4. Triangles
    A_sq = A @ A
    F_tri = np.diag(A_sq) / 2.0
    
    # 5. Core Number
    core = np.zeros(N, dtype=int)
    deg = D.copy().astype(int)
    max_deg = int(deg.max()) if N > 0 else 0
    # Simple k-core implementation
    # For N=200, we can do this iteratively
    remaining = np.ones(N, dtype=bool)
    current_deg = deg.copy()
    max_k = max_deg
    while max_k >= 0:
        changed = True
        while changed:
            changed = False
            for i in range(N):
                if remaining[i] and current_deg[i] < max_k:
                    # Remove i
                    remaining[i] = False
                    core[i] = max_k - 1 # Approximation, will refine
                    for j in nbrs[i]:
                        if remaining[j]:
                            current_deg[j] -= 1
                    changed = True
        # Re-evaluate max_k
        active_deg = current_deg[remaining]
        if len(active_deg) > 0:
            max_k = int(active_deg.max())
        else:
            break
            
    # Proper Core Number Calculation (Buckets)
    core = np.zeros(N, dtype=int)
    deg = D.copy().astype(int)
    max_deg = int(deg.max()) if N > 0 else 0
    buckets = [[] for _ in range(max_deg + 1)]
    for i in range(N):
        buckets[min(deg[i], max_deg)].append(i)
    remaining = np.ones(N, dtype=bool)
    current_max = max_deg
    while current_max >= 0:
        while buckets[current_max]:
            i = buckets[current_max].pop()
            if not remaining[i]:
                continue
            core[i] = current_max
            remaining[i] = False
            for j in nbrs[i]:
                if remaining[j]:
                    old_deg = deg[j]
                    deg[j] -= 1
                    # Remove from old bucket
                    try:
                        buckets[old_deg].remove(j)
                    except ValueError:
                        pass
                    new_deg = max(0, min(deg[j], max_deg))
                    buckets[new_deg].append(j)
        current_max -= 1
        
    F_core = core.astype(float)
    
    # 6. Fiedler Vector
    L = np.diag(D) - A
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        idx = np.argsort(eigvals)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        fiedler = eigvecs[:, 1]
        F_fiedler_abs = np.abs(fiedler)
    except Exception:
        F_fiedler_abs = np.zeros(N)
        
    # 7. Eccentricity (Approx)
    # Use shortest paths
    try:
        # dist_matrix[i][j] is distance from i to j
        # scipy.sparse.csgraph.shortest_path returns matrix
        # We need max distance from i to any other node
        dist = shortest_path(A_bool.astype(float), method='D', unweighted=True)
        # Replace inf with 0 (disconnected)
        dist = np.where(np.isinf(dist), 0, dist)
        F_ecc = dist.max(axis=1)
    except Exception:
        F_ecc = np.zeros(N)

    # Normalize features to [0, 1]
    def norm(x):
        x = np.asarray(x, dtype=float)
        min_val = x.min()
        max_val = x.max()
        if max_val == min_val:
            return np.zeros_like(x)
        return (x - min_val) / (max_val - min_val)

    F_norm = [
        norm(F_deg),
        norm(F_snd),
        norm(F_block_cost),
        norm(F_tri),
        norm(F_core),
        norm(F_fiedler_abs),
        norm(F_ecc)
    ]
    
    # Feature Matrix: N x 7
    F_mat = np.column_stack(F_norm)
    
    # Baseline scores
    # Strategy: Low Degree, High SND, Low Block Cost, Low Core, High FiedlerAbs, High Ecc
    # Weights: [-1, 1, -1, 0, -1, 1, 1]
    w_init = np.array([-1.0, 1.0, -1.0, 0.0, -1.0, 1.0, 1.0])
    scores_init = F_mat @ w_init
    
    # Fast evaluation function
    def evaluate_fast(scores):
        indices = np.argsort(-scores, kind='stable')
        blocked = np.zeros(N, dtype=bool)
        count = 0
        for v in indices:
            if not blocked[v]:
                count += 1
                blocked[v] = True
                if len(nbrs[v]) > 0:
                    blocked[nbrs[v]] = True
        return count

    best_size = evaluate_fast(scores_init)
    best_scores = scores_init.copy()
    
    # Optimization
    # Objective: Minimize -size
    def objective(w):
        if time.time() > DEADLINE:
            return best_size
        s = F_mat @ w
        return -evaluate_fast(s)
    
    # Nelder-Mead
    # Multiple restarts
    rng = np.random.default_rng(42)
    
    # Initial candidates
    starts = [
        w_init,
        [-1.0, 0.5, -1.0, 0.0, -1.0, 0.5, 0.5],
        [-0.5, 1.0, -0.5, 0.0, -0.5, 1.0, 1.0],
        [-1.0, 1.0, -1.0, 0.5, -1.0, 1.0, 1.0],
        [-1.0, 1.0, -1.0, -0.5, -1.0, 1.0, 1.0],
        # Random
        rng.normal(0, 1, 7),
        rng.normal(0, 1, 7)
    ]
    
    for w0 in starts:
        if time.time() > DEADLINE - 10:
            break
        try:
            res = minimize(objective, w0, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-4})
            if res.fun < -best_size:
                best_size = -res.fun
                best_scores = F_mat @ res.x
        except Exception:
            pass
            
    # Local Polish: Random perturbations of best scores
    if time.time() < DEADLINE - 5:
        for _ in range(2000):
            if time.time() > DEADLINE - 2:
                break
            # Perturb weights
            w_best = np.linalg.lstsq(F_mat, best_scores, rcond=None)[0]
            w_pert = w_best + rng.normal(0, 0.05, 7)
            s_pert = F_mat @ w_pert
            sz = evaluate_fast(s_pert)
            if sz > best_size:
                best_size = sz
                best_scores = s_pert.copy()
                
    # Final Polish: Direct score perturbations
    if time.time() < DEADLINE - 1:
        for _ in range(1000):
            if time.time() > DEADLINE - 0.5:
                break
            s_pert = best_scores + rng.normal(0, 0.01, N)
            sz = evaluate_fast(s_pert)
            if sz > best_size:
                best_size = sz
                best_scores = s_pert.copy()

    return best_scores
