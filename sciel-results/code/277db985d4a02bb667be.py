"""QAP solver for tho150 using MDS/Procrustes init with BCD + multi-restart 2-opt polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 120s kill

    a_f = a.astype(np.float64)
    b_f = b.astype(np.float64)

    def compute_cost(p):
        return float(np.sum(a_f * b_f[np.ix_(p, p)]))

    def mds_embedding(D, dim=2):
        nn = D.shape[0]
        J = np.eye(nn) - np.ones((nn, nn)) / nn
        B = -0.5 * J @ (D ** 2) @ J
        vals, vecs = np.linalg.eigh(B)
        idx = np.argsort(vals)[::-1]
        vals = vals[idx]
        vecs = vecs[:, idx]
        vals = vals[:dim]
        vecs = vecs[:, :dim]
        vals = np.maximum(vals, 0)
        coords = vecs * np.sqrt(vals)
        return coords

    def procrustes_alignment(X, Y):
        from scipy.optimize import linear_sum_assignment
        U, S, Vt = np.linalg.svd(Y.T @ X)
        Q = U @ Vt
        X_aligned = X @ Q
        X_norms = np.sum(X_aligned ** 2, axis=1)
        Y_norms = np.sum(Y ** 2, axis=1)
        Cross = X_aligned @ Y.T
        C = X_norms[:, None] + Y_norms[None, :] - 2 * Cross
        row_ind, col_ind = linear_sum_assignment(C)
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        return p

    def local_search(p, current_cost, end_time, seed=12345):
        """Fast 2-swap local search with precomputed a rows/cols."""
        curr_p = p.copy()
        curr_cost = current_cost
        best_p = p.copy()
        best_cost = current_cost
        
        # Precompute a row and col arrays for faster access
        a_rows = a_f  # just use a_f directly
        a_cols = a_f.T  # transpose view
        
        rng = np.random.RandomState(seed)
        pairs = np.array([(i, j) for i in range(n) for j in range(i + 1, n)], dtype=np.int32)
        
        while time.time() < end_time:
            improved = False
            rng.shuffle(pairs)
            
            for idx in range(len(pairs)):
                if time.time() > end_time:
                    break
                
                i, j = pairs[idx]
                li = curr_p[i]
                lj = curr_p[j]
                
                drow = b_f[lj, curr_p] - b_f[li, curr_p]
                dcol = b_f[curr_p, lj] - b_f[curr_p, li]
                
                delta = np.dot(a_rows[i] - a_rows[j], drow) + np.dot(a_cols[i] - a_cols[j], dcol)
                
                if delta < -1e-10:
                    curr_p[i], curr_p[j] = lj, li
                    curr_cost += delta
                    if curr_cost < best_cost:
                        best_cost = curr_cost
                        best_p = curr_p.copy()
                    improved = True
            
            if not improved:
                break
                
        return best_p, best_cost

    def sa_polish(p, current_cost, end_time, seed=111):
        """Simulated annealing polish with adaptive temperature."""
        curr_p = p.copy()
        curr_cost = current_cost
        best_p = p.copy()
        best_cost = current_cost
        
        rng_sa = np.random.RandomState(seed)
        
        # Estimate typical delta magnitude
        i0, j0 = rng_sa.choice(n, 2, replace=False)
        li0 = curr_p[i0]
        lj0 = curr_p[j0]
        drow0 = b_f[lj0, curr_p] - b_f[li0, curr_p]
        dcol0 = b_f[curr_p, lj0] - b_f[curr_p, li0]
        delta0 = np.dot(a_f[i0] - a_f[j0], drow0) + np.dot(a_f[:, i0] - a_f[:, j0], dcol0)
        T = max(abs(delta0) * 2.0, 100.0)
        cooling = 0.995  # Slower cooling for more iterations
        
        a_cols = a_f.T
        
        while time.time() < end_time:
            i, j = rng_sa.choice(n, 2, replace=False)
            li = curr_p[i]
            lj = curr_p[j]
            
            drow = b_f[lj, curr_p] - b_f[li, curr_p]
            dcol = b_f[curr_p, lj] - b_f[curr_p, li]
            delta = np.dot(a_f[i] - a_f[j], drow) + np.dot(a_cols[i] - a_cols[j], dcol)
            
            if delta < 0 or rng_sa.random_sample() < np.exp(-delta / T):
                curr_p[i], curr_p[j] = lj, li
                curr_cost += delta
                if curr_cost < best_cost:
                    best_cost = curr_cost
                    best_p = curr_p.copy()
            
            T *= cooling
            if T < 1.0:
                T = 1.0
        
        return best_p, best_cost

    # 1. MDS Initialization - try multiple dims
    a_sym = (a_f + a_f.T) / 2.0
    b_sym = (b_f + b_f.T) / 2.0
    
    best_p_init = None
    best_cost_init = np.inf
    
    for dim in [2, 3, 5, 8]:
        if time.time() > deadline - 60.0:
            break
        X = mds_embedding(a_sym, dim)
        Y = mds_embedding(b_sym, dim)
        p_try = procrustes_alignment(X, Y)
        cost_try = compute_cost(p_try)
        if cost_try < best_cost_init:
            best_cost_init = cost_try
            best_p_init = p_try
    
    # 2. Rank matching
    a_row_sums = a_f.sum(axis=1)
    b_row_sums = b_f.sum(axis=1)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p_rank = np.empty(n, dtype=np.int64)
    p_rank[heavy] = central
    cost_rank = compute_cost(p_rank)
    if cost_rank < best_cost_init:
        best_cost_init = cost_rank
        best_p_init = p_rank

    # 3. Initial local search
    end_time_lk1 = deadline - 40.0
    best_p, best_cost = local_search(best_p_init, best_cost_init, end_time_lk1, seed=12345)
    
    # 4. Multiple perturbation + local search restarts
    rng_perturb = np.random.RandomState(789)
    num_restarts = 5
    for r in range(num_restarts):
        if time.time() > deadline - 25.0:
            break
        
        p_pert = best_p.copy()
        # Random number of swaps: 2 to 6
        n_swaps = rng_perturb.randint(2, 7)
        for _ in range(n_swaps):
            i, j = rng_perturb.choice(n, 2, replace=False)
            p_pert[i], p_pert[j] = p_pert[j], p_pert[i]
        
        cost_pert = compute_cost(p_pert)
        end_time_lk = deadline - 20.0 - r * 3.0
        if end_time_lk < time.time() + 5:
            break
        p_lk, cost_lk = local_search(p_pert, cost_pert, end_time_lk, seed=456 + r)
        
        if cost_lk < best_cost:
            best_p = p_lk
            best_cost = cost_lk
    
    # 5. SA polish if time remains
    if time.time() < deadline - 5.0:
        end_time_sa = deadline - 2.0
        p_sa, cost_sa = sa_polish(best_p, best_cost, end_time_sa, seed=111)
        if cost_sa < best_cost:
            best_p = p_sa
            best_cost = cost_sa
    
    # 6. Final short local search
    if time.time() < deadline - 1.0:
        end_time_final = deadline - 0.5
        p_final, cost_final = local_search(best_p, best_cost, end_time_final, seed=999)
        if cost_final < best_cost:
            best_p = p_final
            best_cost = cost_final

    return best_p.tolist()
# EVOLVE-BLOCK-END
