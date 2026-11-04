"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    n = len(a)
    deadline = time.time() + 115.0  # safety margin before 120s kill

    # Precompute
    a_sym = (a + a.T) / 2.0
    b_sym = (b + b.T) / 2.0

    def compute_cost(p):
        # sum_{i,j} a[i,j] * b[p[i], p[j]]
        bp = b[np.ix_(p, p)]
        return int(np.sum(a * bp))

    # Initial solution: Hungarian algorithm on linear surrogate
    # Surrogate: facility i has flow centrality a.sum(axis=1), location j has distance centrality b.sum(axis=1)
    # We want to assign high-flow facilities to low-distance locations
    # Cost matrix for Hungarian: C[i,j] = -a.sum(axis=1)[i] * b.sum(axis=1)[j]
    # But this is a rank-1 matrix, so Hungarian will give a specific optimal assignment.
    # Better: use a more nuanced surrogate. Let's use the actual QAP structure.
    # For a linear surrogate, we can use: C[i,j] = sum_k a[i,k] * b[j, k] + sum_k a[k, i] * b[k, j]
    # This is the contribution of facility i at location j if all other facilities were optimally placed.
    # Actually, a simpler and effective surrogate: C[i,j] = a.sum(axis=1)[i] * b.sum(axis=1)[j]
    # Since we want to minimize, we negate for Hungarian (which minimizes).
    
    flow_cent = a.sum(axis=1).astype(np.float64)
    dist_cent = b.sum(axis=1).astype(np.float64)
    
    # Build cost matrix for Hungarian: we want high flow_cent matched with low dist_cent
    # So cost[i,j] = flow_cent[i] * dist_cent[j] (we minimize this)
    C = np.outer(flow_cent, dist_cent)
    
    # Hungarian algorithm minimizes the total cost
    row_ind, col_ind = linear_sum_assignment(C)
    p_hung = np.empty(n, dtype=np.int64)
    p_hung[row_ind] = col_ind
    
    best_p = p_hung.copy()
    best_cost = compute_cost(best_p)

    def compute_all_swap_deltas(p):
        """Compute delta for all pairs (i,j) with i<j in a fully vectorized manner."""
        b_p = b[np.ix_(p, p)]  # b_p[i,j] = b[p[i], p[j]]
        
        # M1[i,k] = a[i,:] @ b_p[k,:] = sum_j a[i,j]*b[p[k],p[j]]
        M1 = a @ b_p.T  # (n,n)
        
        # M2[i,k] = a[:,i] @ b_p[:,k] = sum_j a[j,i]*b[p[j],p[k]]
        M2 = a.T @ b_p  # (n,n)
        
        # term1_full[i,j] = M1[i,j] - M1[i,i] - M1[j,j] + M1[j,i]
        # But this is the FULL sum including k=i,j. We need to subtract those.
        
        # For term1, the k=i contribution: (a[i,i]-a[j,i])*(b[p[j],p[i]]-b[p[i],p[i]])
        # = (a[i,i]-a[j,i])*(b_p[j,i]-b_p[i,i])
        # The k=j contribution: (a[i,j]-a[j,j])*(b[p[j],p[j]]-b[p[i],p[j]])
        # = (a[i,j]-a[j,j])*(b_p[j,j]-b_p[i,j])
        
        # For term2, the k=i contribution: (a[i,i]-a[i,j])*(b[p[i],p[j]]-b[p[i],p[i]])
        # = (a[i,i]-a[i,j])*(b_p[i,j]-b_p[i,i])
        # The k=j contribution: (a[j,i]-a[j,j])*(b[p[j],p[j]]-b[p[j],p[i]])
        # = (a[j,i]-a[j,j])*(b_p[j,j]-b_p[j,i])
        
        # term3[i,j] = (a[i,j]-a[j,i])*(b[p[j],p[i]]-b[p[i],p[j]])
        
        # Build all-pairs matrices
        ii = np.arange(n)
        
        # term1 corrections
        t1_ki = (a[ii, ii] - a[np.ix_(ii, ii)][1]) * (b_p[np.ix_(ii, ii)][0] - b_p[np.ix_(ii, ii)][1])
        # This is getting messy. Let me use explicit broadcasting.
        
        # a_ii[i] = a[i,i], a_ij[i,j] = a[i,j]
        a_ii = np.diag(a)
        b_p_ii = np.diag(b_p)
        
        # t1_ki[i,j] = (a[i,i]-a[j,i])*(b_p[j,i]-b_p[i,i])
        # = a_ii[i]*b_p[j,i] - a_ii[i]*b_p[i,i] - a[j,i]*b_p[j,i] + a[j,i]*b_p[i,i]
        # Let's just compute it directly with broadcasting
        ai_i = a_ii[:, None]  # (n,1)
        aj_i = a.T  # a[j,i] -> a.T[i,j]
        bp_ji = b_p.T  # b_p[j,i] -> b_p.T[i,j]
        bp_ii = b_p_ii[:, None]  # (n,1)
        
        t1_ki = (ai_i - aj_i) * (bp_ji - bp_ii)  # (n,n), t1_ki[i,j]
        
        # t1_kj[i,j] = (a[i,j]-a[j,j])*(b_p[j,j]-b_p[i,j])
        ai_j = a  # a[i,j]
        aj_j = a_ii[None, :]  # (1,n)
        bp_jj = b_p_ii[None, :]  # (1,n)
        bp_ij = b_p  # b_p[i,j]
        
        t1_kj = (ai_j - aj_j) * (bp_jj - bp_ij)  # (n,n), t1_kj[i,j]
        
        # term1[i,j] = (M1[i,j] - M1[i,i] - M1[j,j] + M1[j,i]) - t1_ki[i,j] - t1_kj[i,j]
        M1_ii = np.diag(M1)
        M1_jj = M1_ii[None, :]
        M1_ii_col = M1_ii[:, None]
        M1_ji = M1.T
        
        term1 = (M1 - M1_ii_col - M1_jj + M1_ji) - t1_ki - t1_kj
        
        # term2 corrections
        # t2_ki[i,j] = (a[i,i]-a[i,j])*(b_p[i,j]-b_p[i,i])
        ai_i2 = a_ii[:, None]
        ai_j2 = a
        bp_ij2 = b_p
        bp_ii2 = b_p_ii[:, None]
        
        t2_ki = (ai_i2 - ai_j2) * (bp_ij2 - bp_ii2)  # (n,n), t2_ki[i,j]
        
        # t2_kj[i,j] = (a[j,i]-a[j,j])*(b_p[j,j]-b_p[j,i])
        aj_i2 = a.T
        aj_j2 = a_ii[None, :]
        bp_jj2 = b_p_ii[None, :]
        bp_ji2 = b_p.T
        
        t2_kj = (aj_i2 - aj_j2) * (bp_jj2 - bp_ji2)  # (n,n), t2_kj[i,j]
        
        # term2[i,j] = (M2[i,j] - M2[i,i] - M2[j,j] + M2[j,i]) - t2_ki[i,j] - t2_kj[i,j]
        M2_ii = np.diag(M2)
        M2_jj = M2_ii[None, :]
        M2_ii_col = M2_ii[:, None]
        M2_ji = M2.T
        
        term2 = (M2 - M2_ii_col - M2_jj + M2_ji) - t2_ki - t2_kj
        
        # term3[i,j] = (a[i,j]-a[j,i])*(b_p[j,i]-b_p[i,j])
        term3 = (a - a.T) * (b_p.T - b_p)
        
        deltas = term1 + term2 + term3
        
        # Zero out diagonal and make symmetric (only upper triangle matters)
        np.fill_diagonal(deltas, 0)
        
        return deltas

    iu, ju = np.triu_indices(n, k=1)

    def local_search(p, max_iters=500):
        """Best-improvement local search with swap moves."""
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            deltas = compute_all_swap_deltas(p)
            valid_deltas = deltas[iu, ju]
            min_idx = np.argmin(valid_deltas)
            min_delta = valid_deltas[min_idx]
            if min_delta >= 0:
                break
            i_idx = iu[min_idx]
            j_idx = ju[min_idx]
            p[i_idx], p[j_idx] = p[j_idx], p[i_idx]
        return p

    def three_cycle_delta(p, i, j, k):
        """Compute delta for 3-cycle: i->j->k->i (move facility at i to j's pos, j to k's, k to i's)."""
        u, v, w = p[i], p[j], p[k]
        old = b[np.ix_([u, v, w], [u, v, w])]
        # New assignment: i gets v, j gets w, k gets u
        new = b[np.ix_([v, w, u], [v, w, u])]
        a_sub = a[np.ix_([i, j, k], [i, j, k])]
        return int(np.sum(a_sub * new) - np.sum(a_sub * old))

    # Hill climbing with vectorized deltas and random restarts
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break

        if restart == 0:
            p = best_p.copy()
        elif restart % 3 == 1:
            # Moderate perturbation
            p = best_p.copy()
            num_swaps = 2 + restart // 8
            for _ in range(num_swaps):
                i, j = np.random.choice(n, 2, replace=False)
                p[i], p[j] = p[j], p[i]
        else:
            # Aggressive perturbation: random restart with greedy refinement
            p = np.random.permutation(n).astype(np.int64)
            p = local_search(p, max_iters=50)

        # Local search
        p = local_search(p, max_iters=200)

        # Try 3-cycles to escape local optima
        if time.time() < deadline - 5.0:
            improved_3c = True
            while improved_3c and time.time() < deadline - 5.0:
                improved_3c = False
                # Sample random 3-cycles
                for _ in range(200):
                    if time.time() > deadline - 5.0:
                        break
                    i, j, k = np.random.choice(n, 3, replace=False)
                    d = three_cycle_delta(p, i, j, k)
                    if d < 0:
                        # Apply 3-cycle: i->j, j->k, k->i
                        p[i], p[j], p[k] = p[j], p[k], p[i]
                        improved_3c = True
                        break

        # Re-evaluate
        cur_cost = compute_cost(p)
        if cur_cost < best_cost:
            best_cost = cur_cost
            best_p = p.copy()

    # Final polish
    if time.time() < deadline - 2.0:
        p = best_p.copy()
        p = local_search(p, max_iters=300)
        cur_cost = compute_cost(p)
        if cur_cost < best_cost:
            best_cost = cur_cost
            best_p = p.copy()

    return best_p.tolist()
# EVOLVE-BLOCK-END
