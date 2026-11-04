"""QAP tai256c solver: vectorized iterated local search with O(n^2) per-pass 2-opt."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    n = len(a)
    deadline = time.time() + 115.0

    def compute_cost(p):
        return int(np.sum(a * b[np.ix_(p, p)]))

    # --- Initialization via Hungarian (bipartite matching on flow/distance centrality) ---
    # Build a cost matrix for the Hungarian: weight[i, j] = a_row_sum[i] * b_row_sum[j]
    # We want to maximize this, so minimize its negative.
    a_rs = a.sum(axis=1).astype(np.float64)
    b_rs = b.sum(axis=1).astype(np.float64)
    # Also try column sums for a richer signal
    a_cs = a.sum(axis=0).astype(np.float64)
    b_cs = b.sum(axis=0).astype(np.float64)

    # Combined centrality
    a_cent = a_rs + a_cs
    b_cent = b_rs + b_cs

    # Hungarian: minimize -a_cent[i] * b_cent[j]
    cost_mat = -np.outer(a_cent, b_cent)
    # linear_sum_assignment minimizes; we get row -> col assignment
    row_ind, col_ind = linear_sum_assignment(cost_mat)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind

    best_p = p.copy()
    best_cost = compute_cost(best_p)

    # --- Vectorized 2-opt local search ---
    # For swapping p[i] and p[j] (i < j), the delta is:
    #   delta = sum_{k != i,j} (a[i,k] - a[j,k]) * (b[pj,p[k]] - b[pi,p[k]])
    #         + sum_{k != i,j} (a[k,i] - a[k,j]) * (b[p[k],pj] - b[p[k],pi])
    #         + a[i,j]*(b[pj,pi] - b[pi,pj]) + a[j,i]*(b[pi,pj] - b[pj,pi])
    #         + a[i,i]*(b[pj,pj] - b[pi,pi]) + a[j,j]*(b[pi,pi] - b[pj,pj])
    #
    # Vectorized for fixed i: compute delta for all j at once.

    def local_search(p, cost):
        p = p.copy()
        improved = True
        max_passes = 200
        passes = 0
        while improved and time.time() < deadline - 2.0 and passes < max_passes:
            improved = False
            passes += 1
            for i in range(n):
                if time.time() > deadline - 2.0:
                    return p, cost
                pi = p[i]
                # Compute delta for swapping p[i] with p[j] for all j != i
                # We'll compute for all j, then mask out j=i
                pj_all = p  # array of all current locations
                # diff_row = a[i,:] - a[j,:] but j varies... 
                # Let me restructure: for fixed i, delta(i,j) involves:
                #   term1 = sum_k a[i,k]*(b[pj,p[k]] - b[pi,p[k]])  -> for each j, this is a[i,:] @ (b[pj,:] - b[pi,:])
                #   term2 = -sum_k a[j,k]*(b[pj,p[k]] - b[pi,p[k]]) -> -a[j,:] @ (b[pj,:] - b[pi,:])
                #   term3 = sum_k a[k,i]*(b[p[k],pj] - b[p[k],pi])
                #   term4 = -sum_k a[k,j]*(b[p[k],pj] - b[p[k],pi])
                #   + cross/diagonal corrections
                #
                # For term1: b[pj,:] - b[pi,:] for all j -> B_diff = b[p,:] - b[pi,:][:,None]  shape (n, n)
                #   term1_j = a[i,:] @ B_diff[j,:] = (a[i,:] * B_diff).sum(axis=1)
                B_diff = b[p] - b[pi]  # shape (n, n): B_diff[j, :] = b[p[j], :] - b[pi, :]
                # term1 for all j: a[i,:] @ B_diff[j] for each j
                term1_all = (a[i] * B_diff).sum(axis=1)  # shape (n,)
                
                # term2: -a[j,:] @ (b[pj,:] - b[pi,:]) for each j
                # = -B_diff[j,:] @ a[j,:] = -B_diff @ a.T  ... shape (n,)
                term2_all = -(B_diff * a).sum(axis=1)  # B_diff[j,:] * a[j,:] summed over cols
                
                # term3: sum_k a[k,i] * (b[p[k], pj] - b[p[k], pi]) for each j
                # = a[:,i] @ (b[p[:,j],:] - b[p[:,i],:]) ... 
                # For fixed j: sum_k a[k,i] * (b[p[k], j_loc] - b[p[k], pi])
                # where j_loc = p[j]
                # = (a[:,i] * (b[:, j_loc] - b[:, pi]))[p] ... no
                # Let me think: sum_k a[k,i] * b[p[k], j_loc] - sum_k a[k,i] * b[p[k], pi]
                # = a[:,i] @ b[np.ix_(p, [j_loc])] - a[:,i] @ b[np.ix_(p, [pi])]
                # For all j at once: 
                # sum_k a[k,i] * b[p[k], p[j]] for all j = (a[:,i][:,None] * b[np.ix_(p, p)]).sum(axis=0)
                # That's expensive. Alternative:
                # Let M = b[np.ix_(p, p)]  shape (n,n)
                # term3_j = a[:,i] @ M[:, j] - a[:,i] @ M[:, i]  (since p[i]=pi, the "pi" index in p is i)
                # Wait, b[p[k], pi] = M[k, i] because p[i] = pi
                # b[p[k], p[j]] = M[k, j]
                # So term3_j = a[:,i] @ M[:,j] - a[:,i] @ M[:,i]
                # = (M.T @ a[:,i])[j] - (M.T @ a[:,i])[i]
                M = b[np.ix_(p, p)]  # M[k, l] = b[p[k], p[l]]
                a_col_i = a[:, i]
                term3_all = M.T @ a_col_i  # shape (n,): term3_all[j] = sum_k M[k,j]*a[k,i] = sum_k b[p[k],p[j]]*a[k,i]
                # subtract the constant part: sum_k a[k,i]*b[p[k],pi] = term3_all[i]
                term3_all = term3_all - term3_all[i]  # now term3_all[j] = sum_k a[k,i]*(b[p[k],p[j]] - b[p[k],pi])
                
                # term4: -sum_k a[k,j] * (b[p[k], pj] - b[p[k], pi]) for each j
                # = -(sum_k a[k,j]*b[p[k],p[j]] - sum_k a[k,j]*b[p[k],pi])
                # = -(a[:,j] @ M[:,j] - a[:,j] @ M[:,i])
                # For all j: a[:,j] @ M[:,j] = diag(M.T @ a) ... 
                # a[:,j] @ M[:,j] = sum_k a[k,j]*M[k,j] = sum_k a[k,j]*b[p[k],p[j]]
                # Let's compute: for each j, sum_k a[k,j]*M[k,j] = (a * M).sum(axis=0)
                aM_diag = (a * M).sum(axis=0)  # shape (n,): aM_diag[j] = sum_k a[k,j]*M[k,j]
                # sum_k a[k,j]*M[k,i] = M.T @ a ... no: M.T[i,j] = M[j,i], (M.T @ a)[j] = sum_i M[j,i]*a[i,j] ... 
                # sum_k a[k,j]*M[k,i] = (a[:,j] * M[:,i]).sum() for each j
                # = (M[:,i][:,None] * a).sum(axis=0)  shape (n,)
                term4_const = (M[:, i][:, None] * a).sum(axis=0)  # shape (n,): for each j, sum_k a[k,j]*M[k,i]
                term4_all = -(aM_diag - term4_const)
                
                # Now total delta before corrections:
                delta_all = term1_all + term2_all + term3_all + term4_all
                
                # Corrections for k=i and k=j terms (they were included in the sums above but need adjustment)
                # The sums above include k=i and k=j. We need to exclude them and add back the correct pair terms.
                # Actually, let me reconsider. The formula delta(i,j) = 
                #   sum_{k != i,j} (a[i,k] - a[j,k])*(b[pj,p[k]] - b[pi,p[k]])
                #   + sum_{k != i,j} (a[k,i] - a[k,j])*(b[p[k],pj] - b[p[k],pi])
                #   + a[i,j]*(b[pj,pi] - b[pi,pj]) + a[j,i]*(b[pi,pj] - b[pj,pi])
                #   + a[i,i]*(b[pj,pj] - b[pi,pi]) + a[j,j]*(b[pi,pi] - b[pj,pj])
                #
                # My term1+term2 = sum_k (a[i,k] - a[j,k])*(b[pj,p[k]] - b[pi,p[k]])  (includes k=i,j)
                # My term3+term4 = sum_k (a[k,i] - a[k,j])*(b[p[k],pj] - b[p[k],pi])  (includes k=i,j)
                #
                # So I need to subtract the k=i and k=j contributions from these sums, then add the explicit pair terms.
                
                # Subtract k=i from term1+term2:
                # (a[i,i] - a[j,i])*(b[pj,pi] - b[pi,pi])
                ki1 = (a[i, i] - a[j, i]) * (b[pj_all[i], pi] - b[pi, pi])  # this is for j=i, but we need per-j
                # Actually this is getting complicated. Let me use a simpler but still vectorized approach.
                # 
                # Alternative: just compute delta for the best j by evaluating a few candidates.
                # Or: accept that the full vectorized delta is complex and use a hybrid.
                #
                # Let me try a different vectorization: compute the full delta matrix for all (i,j) pairs at once.
                # That would be O(n^3) memory which is too much for n=256 (256^3 = 16.7M floats = 134MB, borderline).
                #
                # Let me just do the per-i vectorization correctly.
                
                # For fixed i, delta(i, j) for all j:
                # Let's define:
                #   X[j] = b[p[j], :] - b[pi, :]   (row difference, shape (n,) for each j)
                #   Y[j] = b[:, p[j]] - b[:, pi]   (column difference, shape (n,) for each j)
                #
                # delta(i,j) = a[i,:] @ X[j] - a[j,:] @ X[j] + a[:,i] @ Y[j] - a[:,j] @ Y[j]
                #             + corrections for k=i,j in the sums + explicit pair terms
                #
                # The sums a[i,:] @ X[j] include k=i and k=j. Let's compute them fully then subtract.
                #
                # Full sum1_j = a[i,:] @ X[j] = sum_k a[i,k] * (b[p[j],p[k]] - b[pi,p[k]])
                #   = a[i,:] @ b[p[j],:] - a[i,:] @ b[pi,:]
                # Full sum2_j = a[j,:] @ X[j] = a[j,:] @ b[p[j],:] - a[j,:] @ b[pi,:]
                # Full sum3_j = a[:,i] @ Y[j] = a[:,i] @ b[:,p[j]] - a[:,i] @ b[:,pi]
                # Full sum4_j = a[:,j] @ Y[j] = a[:,j] @ b[:,p[j]] - a[:,j] @ b[:,pi]
                #
                # delta(i,j) = full_sum1_j - full_sum2_j + full_sum3_j - full_sum4_j
                #   - [k=i terms in sums 1-4] - [k=j terms in sums 1-4]
                #   + a[i,j]*(b[pj,pi] - b[pi,pj]) + a[j,i]*(b[pi,pj] - b[pj,pi])
                #   + a[i,i]*(b[pj,pj] - b[pi,pi]) + a[j,j]*(b[pi,pi] - b[pj,pj])
                #
                # k=i terms:
                #   sum1: a[i,i]*(b[pj,pi] - b[pi,pi])
                #   sum2: a[j,i]*(b[pj,pi] - b[pi,pi])
                #   sum3: a[i,i]*(b[pi,pj] - b[pi,pi])
                #   sum4: a[j,i]*(b[pi,pj] - b[pi,pi])
                # k=j terms:
                #   sum1: a[i,j]*(b[pj,pj] - b[pi,pj])
                #   sum2: a[j,j]*(b[pj,pj] - b[pi,pj])
                #   sum3: a[i,j]*(b[pj,pj] - b[pj,pi])
                #   sum4: a[j,j]*(b[pj,pj] - b[pj,pi])
                #
                # This is still complex. Let me just use the simpler per-pair O(n) delta but vectorize over j using numpy.
                
                # Simplified: for fixed i, compute delta for all j in a vectorized manner
                # using the M matrix approach.
                
                # delta(i,j) = (a[i,:]-a[j,:]) @ (b[p[j],:] - b[pi,:]) 
                #            + (a[:,i]-a[:,j]) @ (b[:,p[j]] - b[:,pi])
                #            + pair corrections
                #
                # = a[i,:] @ b[p[j],:] - a[i,:] @ b[pi,:] - a[j,:] @ b[p[j],:] + a[j,:] @ b[pi,:]
                # + a[:,i] @ b[:,p[j]] - a[:,i] @ b[:,pi] - a[:,j] @ b[:,p[j]] + a[:,j] @ b[:,pi]
                # + corrections
                #
                # Let:
                #   Ab = a @ b  (n x n)  -- precompute
                #   aTb = a.T @ b  (n x n) -- precompute? No, a[:,i] @ b[:,p[j]] is not a simple mat-vec.
                #
                # a[i,:] @ b[p[j],:] = (a @ b.T)[i, p[j]]  ... let's define AbT = a @ b.T
                # a[:,i] @ b[:,p[j]] = (a.T @ b)[i, p[j]]  ... let's define aTb = a.T @ b
                #
                # So:
                #   a[i,:] @ b[p[j],:] = AbT[i, p[j]]
                #   a[i,:] @ b[pi,:] = AbT[i, pi]
                #   a[j,:] @ b[p[j],:] = AbT[j, p[j]]
                #   a[j,:] @ b[pi,:] = AbT[j, pi]
                #   a[:,i] @ b[:,p[j]] = aTb[i, p[j]]
                #   a[:,i] @ b[:,pi] = aTb[i, pi]
                #   a[:,j] @ b[:,p[j]] = aTb[j, p[j]]
                #   a[:,j] @ b[:,pi] = aTb[j, pi]
                #
                # So the main part (before corrections) for fixed i, all j:
                #   main_j = AbT[i, p[j]] - AbT[i, pi] - AbT[j, p[j]] + AbT[j, pi]
                #           + aTb[i, p[j]] - aTb[i, pi] - aTb[j, p[j]] + aTb[j, pi]
                #
                # This is O(n) per i if we have AbT and aTb precomputed!
                #
                # Corrections (subtract k=i and k=j from the full sums, add explicit pair terms):
                # From k=i:
                #   -a[i,i]*(b[pj,pi]-b[pi,pi]) + a[j,i]*(b[pj,pi]-b[pi,pi])
                #   -a[i,i]*(b[pi,pj]-b[pi,pi]) + a[j,i]*(b[pi,pj]-b[pi,pi])
                #   = (a[j,i]-a[i,i])*(b[pj,pi]-b[pi,pi]) + (a[j,i]-a[i,i])*(b[pi,pj]-b[pi,pi])
                #   = (a[j,i]-a[i,i])*(b[pj,pi]+b[pi,pj]-2*b[pi,pi])
                #
                # From k=j:
                #   -a[i,j]*(b[pj,pj]-b[pi,pj]) + a[j,j]*(b[pj,pj]-b[pi,pj])
                #   -a[i,j]*(b[pj,pj]-b[pj,pi]) + a[j,j]*(b[pj,pj]-b[pj,pi])
                #   = (a[j,j]-a[i,j])*(b[pj,pj]-b[pi,pj]) + (a[j,j]-a[i,j])*(b[pj,pj]-b[pj,pi])
                #   = (a[j,j]-a[i,j])*(2*b[pj,pj]-b[pi,pj]-b[pj,pi])
                #
                # Explicit pair terms:
                #   a[i,j]*(b[pj,pi]-b[pi,pj]) + a[j,i]*(b[pi,pj]-b[pj,pi])
                #   + a[i,i]*(b[pj,pj]-b[pi,pi]) + a[j,j]*(b[pi,pi]-b[pj,pj])
                #
                # Total correction for fixed i, all j (vectorized):
                # Let's compute each part:
                #   b_pi_pi = b[pi, pi]  (scalar)
                #   For each j:
                #     c1 = (a[j,i]-a[i,i])*(b[p[j],pi]+b[pi,p[j]]-2*b_pi_pi)
                #     c2 = (a[j,j]-a[i,j])*(2*b[p[j],p[j]]-b[pi,p[j]]-b[p[j],pi])
                #     c3 = a[i,j]*(b[p[j],pi]-b[pi,p[j]]) + a[j,i]*(b[pi,p[j]]-b[p[j],pi])
                #         + a[i,i]*(b[p[j],p[j]]-b_pi_pi) + a[j,j]*(b_pi_pi-b[p[j],p[j]])
                #
                # This is doable vectorized for fixed i.
                
                AbT = a @ b.T  # precompute outside? It's n x n = 65536 floats, fine
                aTb = a.T @ b
                
                # For fixed i:
                AbT_i = AbT[i]  # shape (n,)
                aTb_i = aTb[i]  # shape (n,)
                AbT_pi = AbT[:, pi]  # shape (n,)
                aTb_pi = aTb[:, pi]  # shape (n,)
                
                # main_j = AbT_i[p[j]] - AbT_i[pi] - AbT[j, p[j]] + AbT[j, pi]
                #         + aTb_i[p[j]] - aTb_i[pi] - aTb[j, p[j]] + aTb[j, pi]
                # AbT[j, p[j]] = diag of AbT indexed by p: AbT[np.arange(n), p]
                # aTb[j, p[j]] = aTb[np.arange(n), p]
                AbT_diag_p = AbT[np.arange(n), p]  # shape (n,)
                aTb_diag_p = aTb[np.arange(n), p]  # shape (n,)
                
                main = (AbT_i[p] - AbT_i[pi] - AbT_diag_p + AbT_pi
                        + aTb_i[p] - aTb_i[pi] - aTb_diag_p + aTb_pi)
                
                # Corrections:
                b_pi_pi = b[pi, pi]
                b_p_pi = b[p, pi]  # shape (n,): b[p[j], pi]
                b_pi_p = b[pi, p]  # shape (n,): b[pi, p[j]]
                b_p_p = b[p, p]  # shape (n,): b[p[j], p[j]]
                
                a_col_i = a[:, i]  # shape (n,)
                a_row_i = a[i]     # shape (n,)
                
                # c1 = (a[j,i]-a[i,i])*(b[p[j],pi]+b[pi,p[j]]-2*b_pi_pi)
                c1 = (a_col_i - a[i, i]) * (b_p_pi + b_pi_p - 2 * b_pi_pi)
                
                # c2 = (a[j,j]-a[i,j])*(2*b[p[j],p[j]]-b[pi,p[j]]-b[p[j],pi])
                a_diag = np.diag(a)  # shape (n,)
                c2 = (a_diag - a_row_i) * (2 * b_p_p - b_pi_p - b_p_pi)
                
                # c3 = a[i,j]*(b[p[j],pi]-b[pi,p[j]]) + a[j,i]*(b[pi,p[j]]-b[p[j],pi])
                #     + a[i,i]*(b[p[j],p[j]]-b_pi_pi) + a[j,j]*(b_pi_pi-b[p[j],p[j]])
                c3 = (a_row_i * (b_p_pi - b_pi_p)
                      + a_col_i * (b_pi_p - b_p_pi)
                      + a[i, i] * (b_p_p - b_pi_pi)
                      + a_diag * (b_pi_pi - b_p_p))
                
                delta_all = main + c1 + c2 + c3
                
                # Mask out j=i (swapping with itself is not a move)
                delta_all[i] = 0
                
                # Find the best j (most negative delta)
                best_j = int(np.argmin(delta_all))
                if delta_all[best_j] < -1e-10:
                    p[i], p[best_j] = p[best_j], p[i]
                    cost += delta_all[best_j]
                    improved = True
                    # Recompute M, AbT_diag_p, aTb_diag_p, b_p_p, b_p_pi, b_pi_p after swap
                    # Actually, we need to update p-dependent quantities. Let's just recompute at the top of the i-loop.
                    # But we're inside the i-loop, so let's break and restart the pass.
                    break
        
        return p, cost

    # Precompute AbT and aTb
    AbT = a @ b.T
    aTb = a.T @ b
    a_diag = np.diag(a)

    # Redefine local_search to use precomputed AbT, aTb
    def local_search2(p, cost):
        p = p.copy()
        improved = True
        max_passes = 300
        passes = 0
        while improved and time.time() < deadline - 2.0 and passes < max_passes:
            improved = False
            passes += 1
            for i in range(n):
                if time.time() > deadline - 2.0:
                    return p, cost
                pi = p[i]
                AbT_i = AbT[i]
                aTb_i = aTb[i]
                AbT_pi = AbT[:, pi]
                aTb_pi = aTb[:, pi]
                
                AbT_diag_p = AbT[np.arange(n), p]
                aTb_diag_p = aTb[np.arange(n), p]
                
                main = (AbT_i[p] - AbT_i[pi] - AbT_diag_p + AbT_pi
                        + aTb_i[p] - aTb_i[pi] - aTb_diag_p + aTb_pi)
                
                b_pi_pi = b[pi, pi]
                b_p_pi = b[p, pi]
                b_pi_p = b[pi, p]
                b_p_p = b[p, p]
                
                a_col_i = a[:, i]
                a_row_i = a[i]
                
                c1 = (a_col_i - a[i, i]) * (b_p_pi + b_pi_p - 2 * b_pi_pi)
                c2 = (a_diag - a_row_i) * (2 * b_p_p - b_pi_p - b_p_pi)
                c3 = (a_row_i * (b_p_pi - b_pi_p)
                      + a_col_i * (b_pi_p - b_p_pi)
                      + a[i, i] * (b_p_p - b_pi_pi)
                      + a_diag * (b_pi_pi - b_p_p))
                
                delta_all = main + c1 + c2 + c3
                delta_all[i] = 0
                
                best_j = int(np.argmin(delta_all))
                if delta_all[best_j] < -1e-10:
                    p[i], p[best_j] = p[best_j], p[i]
                    cost += delta_all[best_j]
                    improved = True
                    break
        
        return p, cost

    # Initial local search
    p, cost = local_search2(p.copy(), best_cost)
    if cost < best_cost:
        best_p = p.copy()
        best_cost = cost

    # Iterated local search
    rng = np.random.RandomState(42)
    while time.time() < deadline - 3.0:
        p = best_p.copy()
        # Adaptive perturbation
        n_perturb = rng.randint(2, 10)
        for _ in range(n_perturb):
            i, j = rng.choice(n, 2, replace=False)
            p[i], p[j] = p[j], p[i]

        p, cost = local_search2(p, compute_cost(p))

        if cost < best_cost:
            best_p = p.copy()
            best_cost = cost

    return best_p.tolist()
# EVOLVE-BLOCK-END
