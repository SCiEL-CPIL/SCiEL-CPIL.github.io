"""QAP solver for tho150 using MDS/Procrustes init with linearized-assignment BCD + 2-opt polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0

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

    def make_B_perm(p):
        """Build B_perm[i,j] = b[p[i], p[j]]."""
        return b_f[np.ix_(p, p)]

    def swap_in_B_perm(Bp, i, j):
        """Swap rows/cols i and j of B_perm in-place."""
        Bp[[i, j], :] = Bp[[j, i], :]
        Bp[:, [i, j]] = Bp[:, [j, i]]

    def linearized_assignment(p, Bp, end_time, max_iter=20):
        """
        Block-coordinate descent via linearized assignment.
        At each iteration, build a linear cost matrix from the current Bp,
        solve the LAP, and update. Iterate until convergence or time runs out.
        """
        from scipy.optimize import linear_sum_assignment
        curr_p = p.copy()
        curr_Bp = Bp.copy()
        curr_cost = compute_cost(curr_p)
        best_p = curr_p.copy()
        best_cost = curr_cost

        for it in range(max_iter):
            if time.time() > end_time:
                break

            # Linearize: C[i, j] = sum_k a[i,k] * Bp[k, j]  (facility i to location j)
            # This is the "row" part of the QAP linearization.
            # Full linearization: C[i,j] = sum_k a[i,k]*Bp[k,j] + sum_k Bp[i,k]*a[k,j]
            # But the second term depends on the assignment of row i, so we use:
            # C[i,j] = sum_k a[i,k] * Bp[k,j]  (fixing the column assignment)
            # Plus a symmetric correction.
            
            # Efficient: C = a_f @ Bp  gives C[i,j] = sum_k a[i,k]*Bp[k,j]
            C = a_f @ curr_Bp
            
            # Add the column-interaction term: for each facility i, the cost also depends on
            # which facility k is assigned to the same "row" position. We approximate by
            # adding (Bp_row_i @ a_f.T)[j] = sum_k Bp[i,k] * a[k,j]
            # This makes C more accurate.
            C += curr_Bp @ a_f.T
            
            # Solve LAP
            row_ind, col_ind = linear_sum_assignment(C)
            new_p = np.empty(n, dtype=np.int64)
            new_p[row_ind] = col_ind
            
            new_cost = compute_cost(new_p)
            
            if new_cost < curr_cost - 1e-6:
                # Update Bp by permuting rows and columns
                # New Bp should be b[new_p, new_p]
                # We can rebuild or do row/col permutations
                # Rebuild is O(n^2) which is fine for n=150
                curr_Bp = b_f[np.ix_(new_p, new_p)]
                curr_p = new_p
                curr_cost = new_cost
                
                if curr_cost < best_cost:
                    best_cost = curr_cost
                    best_p = curr_p.copy()
            else:
                # No improvement from full LAP; try a few 2-opt swaps
                break

        return best_p, best_cost, curr_Bp

    def fast_2opt_polish(p, Bp, current_cost, end_time, seed=12345):
        """Fast 2-opt using maintained B_perm."""
        curr_p = p.copy()
        curr_Bp = Bp.copy()
        curr_cost = current_cost
        best_p = p.copy()
        best_cost = current_cost

        rng = np.random.RandomState(seed)
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

        while time.time() < end_time:
            improved = False
            rng.shuffle(pairs)

            for i, j in pairs:
                if time.time() > end_time:
                    break

                # Delta for swapping facilities i and j
                # delta = sum_k a[i,k]*(Bp[k,j]-Bp[k,i]) + sum_k a[k,i]*(Bp[j,k]-Bp[i,k])
                #       + sum_k a[j,k]*(Bp[k,i]-Bp[k,j]) + sum_k a[k,j]*(Bp[i,k]-Bp[j,k])
                # = (a[i,:]-a[j,:]) @ (Bp[:,j]-Bp[:,i]) + (Bp[j,:]-Bp[i,:]) @ (a[:,i]-a[:,j])
                
                dcol = curr_Bp[:, j] - curr_Bp[:, i]
                drow = curr_Bp[j, :] - curr_Bp[i, :]
                a_diff_row = a_f[i, :] - a_f[j, :]
                a_diff_col = a_f[:, i] - a_f[:, j]
                
                delta = np.dot(a_diff_row, dcol) + np.dot(drow, a_diff_col)
                
                if delta < -1e-10:
                    curr_p[i], curr_p[j] = curr_p[j], curr_p[i]
                    swap_in_B_perm(curr_Bp, i, j)
                    curr_cost += delta
                    
                    if curr_cost < best_cost:
                        best_cost = curr_cost
                        best_p = curr_p.copy()
                    improved = True

            if not improved:
                break

        return best_p, best_cost, curr_Bp

    # 1. MDS Initialization
    a_sym = (a_f + a_f.T) / 2.0
    b_sym = (b_f + b_f.T) / 2.0

    best_p_init = None
    best_cost_init = np.inf

    for dim in [2, 3, 5]:
        if time.time() > deadline - 60.0:
            break
        X = mds_embedding(a_sym, dim)
        Y = mds_embedding(b_sym, dim)
        p_try = procrustes_alignment(X, Y)
        cost_try = compute_cost(p_try)
        if cost_try < best_cost_init:
            best_cost_init = cost_try
            best_p_init = p_try

    # 2. Rank Matching
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

    # 3. Random Init
    rng_init = np.random.RandomState(42)
    p_rand = rng_init.permutation(n)
    cost_rand = compute_cost(p_rand)
    if cost_rand < best_cost_init:
        best_cost_init = cost_rand
        best_p_init = p_rand

    # 4. Linearized Assignment BCD
    end_time_lap = deadline - 30.0
    Bp_init = make_B_perm(best_p_init)
    best_p, best_cost, best_Bp = linearized_assignment(best_p_init, Bp_init, end_time_lap)

    # 5. 2-opt polish
    end_time_2opt = deadline - 10.0
    best_p, best_cost, best_Bp = fast_2opt_polish(best_p, best_Bp, best_cost, end_time_2opt)

    # 6. Perturb + LAP + 2-opt if time remains
    if time.time() < deadline - 20.0:
        rng_perturb = np.random.RandomState(789)
        p_pert = best_p.copy()
        for _ in range(5):
            i, j = rng_perturb.choice(n, 2, replace=False)
            p_pert[i], p_pert[j] = p_pert[j], p_pert[i]
        cost_pert = compute_cost(p_pert)
        Bp_pert = make_B_perm(p_pert)
        
        end_time_lap2 = deadline - 15.0
        p_lap2, cost_lap2, Bp_lap2 = linearized_assignment(p_pert, Bp_pert, end_time_lap2)
        
        end_time_2opt2 = deadline - 5.0
        p_2opt2, cost_2opt2, _ = fast_2opt_polish(p_lap2, Bp_lap2, cost_lap2, end_time_2opt2)
        
        if cost_2opt2 < best_cost:
            best_p = p_2opt2
            best_cost = cost_2opt2
            best_Bp = Bp_lap2

    # 7. Final SA polish if time remains
    if time.time() < deadline - 3.0:
        curr_p = best_p.copy()
        curr_cost = best_cost
        best_p_sa = best_p.copy()
        best_cost_sa = best_cost
        
        rng_sa = np.random.RandomState(111)
        T = 500.0
        cooling = 0.9995
        
        end_time_sa = deadline - 1.0
        while time.time() < end_time_sa and T > 1.0:
            i, j = rng_sa.choice(n, 2, replace=False)
            li = curr_p[i]
            lj = curr_p[j]
            
            drow = b_f[lj, curr_p] - b_f[li, curr_p]
            dcol = b_f[curr_p, lj] - b_f[curr_p, li]
            a_diff_row = a_f[i, :] - a_f[j, :]
            a_diff_col = a_f[:, i] - a_f[:, j]
            delta = np.dot(a_diff_row, drow) + np.dot(a_diff_col, dcol)
            
            if delta < 0 or rng_sa.random_sample() < np.exp(-delta / T):
                curr_p[i], curr_p[j] = lj, li
                curr_cost += delta
                
                if curr_cost < best_cost_sa:
                    best_cost_sa = curr_cost
                    best_p_sa = curr_p.copy()
            
            T *= cooling
        
        if best_cost_sa < best_cost:
            best_p = best_p_sa
            best_cost = best_cost_sa

    return best_p.tolist()
# EVOLVE-BLOCK-END
