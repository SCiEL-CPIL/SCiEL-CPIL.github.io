import numpy as np
import time
import warnings

def priority(adjacency):
    start_time = time.time()
    DEADLINE = 130.0  # 20s safety margin before 150s kill

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
    if max_deg > 0:
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

    # 6. Fiedler Vector (2nd smallest eigenvalue of Laplacian)
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

    # 7. Average neighbor degree (hub-neighbor bonus)
    F_avg_nbr_deg = np.array([np.mean(D[nbrs[i]]) if len(nbrs[i]) > 0 else 0 for i in range(N)])

    # 8. Principal Eigenvector (Perron Vector) - signed
    try:
        eigvals_A, eigvecs_A = np.linalg.eigh(A)
        idx_A = np.argsort(eigvals_A)
        eigvals_A = eigvals_A[idx_A]
        eigvecs_A = eigvecs_A[:, idx_A]
        v1 = eigvecs_A[:, -1]
        max_val = np.max(np.abs(v1))
        if max_val > 0:
            v1 = v1 / max_val
        else:
            v1 = np.zeros(N)
    except Exception:
        v1 = np.zeros(N)
    F_perron = v1

    # Normalize features to [0, 1]
    def norm(x):
        x = np.asarray(x, dtype=float)
        min_val = x.min()
        max_val = x.max()
        if max_val == min_val:
            return np.zeros_like(x)
        return (x - min_val) / (max_val - min_val)

    # For Perron vector, we keep the signed value but normalize its magnitude
    # Actually, let's normalize it to have zero mean and unit variance or just [-1, 1]
    # The current norm function maps to [0, 1]. For signed features, this might be okay
    # if we treat it as a magnitude, but the sign matters for bipartition.
    # Let's create a specific normalization for signed features: center and scale to [-1, 1]
    def norm_signed(x):
        x = np.asarray(x, dtype=float)
        x = x - x.mean()
        max_abs = np.max(np.abs(x))
        if max_abs == 0:
            return np.zeros_like(x)
        return x / max_abs

    F_norm = [
        norm(F_deg),
        norm(F_snd),
        norm(F_block_cost),
        norm(F_tri),
        norm(F_core),
        norm(F_fiedler_abs),
        norm(F_avg_nbr_deg)
    ]
    F_perron_norm = norm_signed(F_perron)

    # Feature Matrix: N x 8
    F_mat = np.column_stack(F_norm + [F_perron_norm])
    n_feat = F_mat.shape[1]

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

    # Multiple initial weight strategies
    rng = np.random.default_rng(42)

    # Strategies: 
    # 0: deg, 1: snd, 2: block_cost, 3: tri, 4: core, 5: fiedler, 6: avg_nbr, 7: perron
    # We want: Low Deg, High SND (maybe?), Low Block Cost, Low Core, High Fiedler? 
    # Perron: signed. Positive side vs Negative side.
    
    strategies = [
        np.array([-1.0, 1.0, -1.0, 0.0, -1.0, 1.0, 1.0, 0.5]),
        np.array([-1.0, 0.5, -1.0, 0.0, -1.0, 0.5, 1.0, 0.5]),
        np.array([-0.5, 1.0, -0.5, 0.0, -0.5, 1.0, 0.5, 0.5]),
        np.array([-1.0, 1.0, -1.0, 0.5, -1.0, 1.0, 0.5, 0.5]),
        np.array([-1.0, 1.0, -1.0, -0.5, -1.0, 1.0, 1.0, 0.5]),
        np.array([-1.5, 1.0, -1.0, 0.0, -1.0, 0.5, 1.5, 0.5]),
        np.array([-1.0, 1.0, -1.0, 0.0, -1.0, 1.0, 1.0, 1.0]),
        np.array([-1.0, 1.0, -1.0, 0.0, -1.0, 1.0, 1.0, -1.0]),
        rng.normal(0, 1, n_feat),
        rng.normal(0, 1, n_feat),
    ]

    best_size = 0
    best_scores = None

    for w in strategies:
        s = F_mat @ w
        sz = evaluate_fast(s)
        if sz > best_size:
            best_size = sz
            best_scores = s.copy()

    # Coordinate descent on weights
    def coordinate_descent(w, max_iter=500):
        current_w = w.copy()
        current_s = F_mat @ current_w
        current_sz = evaluate_fast(current_s)
        step = 0.1
        for _ in range(max_iter):
            if time.time() > DEADLINE - 5:
                break
            improved = False
            for d in range(n_feat):
                for delta in [step, -step]:
                    trial_w = current_w.copy()
                    trial_w[d] += delta
                    trial_s = F_mat @ trial_w
                    trial_sz = evaluate_fast(trial_s)
                    if trial_sz > current_sz:
                        current_w = trial_w
                        current_s = trial_s
                        current_sz = trial_sz
                        improved = True
            if not improved:
                step *= 0.5
                if step < 0.001:
                    break
        return current_w, current_s, current_sz

    # Run coordinate descent from best strategy
    if best_scores is not None:
        # Recover weights via least squares
        w_best = np.linalg.lstsq(F_mat, best_scores, rcond=None)[0]
        w_cd, s_cd, sz_cd = coordinate_descent(w_best, max_iter=300)
        if sz_cd > best_size:
            best_size = sz_cd
            best_scores = s_cd.copy()

    # Random restarts with coordinate descent
    for _ in range(30):
        if time.time() > DEADLINE - 10:
            break
        w_rand = rng.normal(0, 1, n_feat)
        w_cd, s_cd, sz_cd = coordinate_descent(w_rand, max_iter=100)
        if sz_cd > best_size:
            best_size = sz_cd
            best_scores = s_cd.copy()

    # Local polish: direct score perturbations (Gaussian noise)
    if best_scores is not None and time.time() < DEADLINE - 3:
        for _ in range(5000):
            if time.time() > DEADLINE - 1:
                break
            s_pert = best_scores + rng.normal(0, 0.02, N)
            sz = evaluate_fast(s_pert)
            if sz > best_size:
                best_size = sz
                best_scores = s_pert.copy()

    # Local polish: Pairwise Swaps (from Inspiration 1)
    # This is a more structured search than random noise.
    # We try swapping the scores of two vertices to see if it improves the set size.
    # To make this tractable, we limit the number of swaps or use a heuristic to pick pairs.
    # Full O(N^2) is 40k checks. Each check is O(N log N). 40k * 200 * 8 ~ 64M ops. 
    # This might be too slow for 200 nodes if done naively in Python loops.
    # However, we can be smart: only swap pairs that are "close" in the current order?
    # Or just do a limited number of random pair swaps.
    
    if best_scores is not None and time.time() < DEADLINE - 1:
        # Random pair swaps
        for _ in range(2000):
            if time.time() > DEADLINE - 0.5:
                break
            i, j = rng.integers(0, N, 2)
            if i == j:
                continue
            s_swap = best_scores.copy()
            s_swap[i], s_swap[j] = s_swap[j], s_swap[i]
            sz = evaluate_fast(s_swap)
            if sz > best_size:
                best_size = sz
                best_scores = s_swap.copy()
    
    # Final fine-tune with smaller steps
    if best_scores is not None and time.time() < DEADLINE - 1:
        for _ in range(3000):
            if time.time() > DEADLINE - 0.5:
                break
            s_pert = best_scores + rng.normal(0, 0.005, N)
            sz = evaluate_fast(s_pert)
            if sz > best_size:
                best_size = sz
                best_scores = s_pert.copy()

    return best_scores
