"""QAP solver for tho150 using MDS/Procrustes initialization with B_perm-accelerated local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    # Convert to float64 for faster arithmetic
    a_f = a.astype(np.float64)
    b_f = b.astype(np.float64)

    def compute_cost(p):
        """Compute exact cost."""
        return float(np.sum(a_f * b_f[np.ix_(p, p)]))

    def mds_embedding(D, dim=2):
        """Classical MDS embedding."""
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
        """Align X to Y and solve assignment problem."""
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

    def make_B_perm(p):
        """Build permuted B matrix: B_perm[i,j] = b_f[p[i], p[j]]."""
        return b_f[np.ix_(p, p)].copy()

    def swap_in_B_perm(B_perm, i, j):
        """Swap rows i,j and columns i,j of B_perm in place."""
        B_perm[[i, j], :] = B_perm[[j, i], :]
        B_perm[:, [i, j]] = B_perm[:, [j, i]]

    def efficient_local_search(p, current_cost, end_time, seed=12345, tol=-1e-10):
        """
        First-improvement 2-swap local search using B_perm for O(1) row/col access.
        delta = (a[i,:] - a[j,:]) @ (B_perm[j,:] - B_perm[i,:])
              + (a[:,i] - a[:,j]) @ (B_perm[:,j] - B_perm[:,i])
        """
        curr_p = p.copy()
        curr_cost = current_cost
        best_p = p.copy()
        best_cost = current_cost
        B_perm = make_B_perm(curr_p)

        rng = np.random.RandomState(seed)
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

        while time.time() < end_time:
            improved = False
            rng.shuffle(pairs)

            for i, j in pairs:
                if time.time() > end_time:
                    break

                drow = B_perm[j, :] - B_perm[i, :]
                dcol = B_perm[:, j] - B_perm[:, i]

                a_diff_row = a_f[i, :] - a_f[j, :]
                a_diff_col = a_f[:, i] - a_f[:, j]

                delta = np.dot(a_diff_row, drow) + np.dot(a_diff_col, dcol)

                if delta < tol:
                    curr_p[i], curr_p[j] = curr_p[j], curr_p[i]
                    curr_cost += delta
                    swap_in_B_perm(B_perm, i, j)

                    if curr_cost < best_cost:
                        best_cost = curr_cost
                        best_p = curr_p.copy()
                    improved = True

            if not improved:
                break

        return best_p, best_cost

    def sa_polish(curr_p, curr_cost, best_p, best_cost, B_perm, end_time, seed=111):
        """SA polish with two-phase cooling: fast exploration then slow exploitation."""
        best_p_sa = best_p.copy()
        best_cost_sa = best_cost
        rng_sa = np.random.RandomState(seed)
        
        # Phase 1: Higher temperature, faster cooling for exploration
        T = 2000.0
        cooling1 = 0.999
        phase1_end = end_time - 10.0
        
        while time.time() < phase1_end and T > 100.0:
            i, j = rng_sa.choice(n, 2, replace=False)

            drow = B_perm[j, :] - B_perm[i, :]
            dcol = B_perm[:, j] - B_perm[:, i]
            a_diff_row = a_f[i, :] - a_f[j, :]
            a_diff_col = a_f[:, i] - a_f[:, j]
            delta = np.dot(a_diff_row, drow) + np.dot(a_diff_col, dcol)

            if delta < 0 or rng_sa.random_sample() < np.exp(-delta / T):
                curr_p[i], curr_p[j] = curr_p[j], curr_p[i]
                curr_cost += delta
                swap_in_B_perm(B_perm, i, j)
                if curr_cost < best_cost_sa:
                    best_cost_sa = curr_cost
                    best_p_sa = curr_p.copy()
            T *= cooling1

        # Phase 2: Lower temperature, slower cooling for exploitation
        T = min(T, 500.0)
        cooling2 = 0.9995
        
        while time.time() < end_time and T > 1.0:
            i, j = rng_sa.choice(n, 2, replace=False)

            drow = B_perm[j, :] - B_perm[i, :]
            dcol = B_perm[:, j] - B_perm[:, i]
            a_diff_row = a_f[i, :] - a_f[j, :]
            a_diff_col = a_f[:, i] - a_f[:, j]
            delta = np.dot(a_diff_row, drow) + np.dot(a_diff_col, dcol)

            if delta < 0 or rng_sa.random_sample() < np.exp(-delta / T):
                curr_p[i], curr_p[j] = curr_p[j], curr_p[i]
                curr_cost += delta
                swap_in_B_perm(B_perm, i, j)
                if curr_cost < best_cost_sa:
                    best_cost_sa = curr_cost
                    best_p_sa = curr_p.copy()
            T *= cooling2

        return best_p_sa, best_cost_sa

    # 1. MDS Initialization (multi-dimension)
    a_sym = (a_f + a_f.T) / 2.0
    b_sym = (b_f + b_f.T) / 2.0

    best_p_init = None
    best_cost_init = np.inf

    for dim in [2, 3, 5, 7]:
        if time.time() > deadline - 80.0:
            break
        X = mds_embedding(a_sym, dim)
        Y = mds_embedding(b_sym, dim)
        p_try = procrustes_alignment(X, Y)
        cost_try = compute_cost(p_try)
        if cost_try < best_cost_init:
            best_cost_init = cost_try
            best_p_init = p_try

    # 2. Rank Matching Initialization
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

    # 3. Random Initialization
    rng_init = np.random.RandomState(42)
    p_rand = rng_init.permutation(n)
    cost_rand = compute_cost(p_rand)
    if cost_rand < best_cost_init:
        best_cost_init = cost_rand
        best_p_init = p_rand

    # 4. Main Local Search Polish
    end_time_lk = deadline - 30.0
    best_p, best_cost = efficient_local_search(best_p_init, best_cost_init, end_time_lk, seed=12345)

    # 5. Perturb and Local Search if time remains
    if time.time() < deadline - 25.0:
        rng_perturb = np.random.RandomState(789)
        p_pert = best_p.copy()
        for _ in range(6):
            i, j = rng_perturb.choice(n, 2, replace=False)
            p_pert[i], p_pert[j] = p_pert[j], p_pert[i]
        cost_pert = compute_cost(p_pert)
        end_time_lk2 = deadline - 20.0
        p_lk2, cost_lk2 = efficient_local_search(p_pert, cost_pert, end_time_lk2, seed=456)
        if cost_lk2 < best_cost:
            best_p = p_lk2
            best_cost = cost_lk2

    # 6. SA Polish with two-phase cooling if time remains
    if time.time() < deadline - 15.0:
        curr_p = best_p.copy()
        curr_cost = best_cost
        B_perm = make_B_perm(curr_p)
        end_time_sa = deadline - 5.0
        best_p_sa, best_cost_sa = sa_polish(curr_p, curr_cost, best_p, best_cost, B_perm, end_time_sa, seed=111)
        if best_cost_sa < best_cost:
            best_p = best_p_sa
            best_cost = best_cost_sa

    # 7. Final high-precision polish pass (tighter tolerance, extra local search)
    if time.time() < deadline - 3.0:
        end_time_final = deadline - 1.0
        p_final, cost_final = efficient_local_search(best_p, best_cost, end_time_final, seed=999, tol=-1e-12)
        if cost_final < best_cost:
            best_p = p_final
            best_cost = cost_final

    return best_p.tolist()
# EVOLVE-BLOCK-END
