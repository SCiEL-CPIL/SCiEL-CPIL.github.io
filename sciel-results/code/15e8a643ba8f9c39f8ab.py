import numpy as np
import time
import warnings

def priority(adjacency):
    start_time = time.time()
    DEADLINE = 148.0  # 2s safety margin before 150s kill

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

    # 5. Core Number (Buckets)
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
    try:
        from scipy.sparse.csgraph import shortest_path
        dist = shortest_path(A_bool.astype(float), method='D', unweighted=True)
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

    # Local swap improvement: try swapping pairs of vertices in the selection order
    # to see if it improves the independent set size
    def local_swap_improve(scores, max_iters=5000):
        current_scores = scores.copy()
        current_size = evaluate_fast(current_scores)
        indices = np.argsort(-current_scores, kind='stable')
        
        for _ in range(max_iters):
            if time.time() > DEADLINE - 5:
                break
            
            # Pick a random pair of indices to swap
            i, j = np.random.randint(0, N, 2)
            if i == j:
                continue
            
            # Swap scores at positions i and j
            current_scores[i], current_scores[j] = current_scores[j], current_scores[i]
            new_size = evaluate_fast(current_scores)
            
            if new_size > current_size:
                current_size = new_size
            else:
                # Revert
                current_scores[i], current_scores[j] = current_scores[j], current_scores[i]
                
        return current_scores, current_size

    # Apply local swap improvement
    best_scores, best_size = local_swap_improve(best_scores, max_iters=3000)

    # Additional random restarts with local search
    rng = np.random.default_rng(42)
    
    for restart in range(20):
        if time.time() > DEADLINE - 10:
            break
        
        # Random perturbation of weights
        w_pert = w_init + rng.normal(0, 0.5, 7)
        scores_pert = F_mat @ w_pert
        
        # Local improvement
        scores_pert, size_pert = local_swap_improve(scores_pert, max_iters=1000)
        
        if size_pert > best_size:
            best_size = size_pert
            best_scores = scores_pert.copy()

    # Final polish: more aggressive local search
    if time.time() < DEADLINE - 2:
        best_scores, best_size = local_swap_improve(best_scores, max_iters=5000)

    return best_scores
