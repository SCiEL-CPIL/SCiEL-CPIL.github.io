"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # 1. Initial Solution: Rank Matching
    # Heavy flows (high row sum in A) to central locations (low row sum in B)
    a_sums = a.sum(axis=1)
    b_sums = b.sum(axis=1)
    
    # Sort indices
    heavy = np.argsort(-a_sums, kind="stable")
    central = np.argsort(b_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # 2. Local Search: Best-Improvement 2-Opt
    # M is B with rows permuted by p
    M = b[p]
    # S[i, l] = sum_k a[i, k] * M[k, l] = sum_k a[i, k] * b[p[k], l]
    S = a @ M
    
    # Precompute constants for delta calculation
    # We need a_ii, a_jj, a_ij
    # b_uu, b_vv, b_uv
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            
            u = p[i]
            # Compute deltas for swapping i with all j
            # Delta_ij = 2(S[i, v] - S[i, u]) + 2(S[j, u] - S[j, v]) + 
            #            a_ii(b_vv - 2b_uv + b_uu) + a_jj(b_uu - 2b_uv + b_vv) + 
            #            a_ij(2b_uv - b_vv - b_uu)
            
            # S[i, u] and S[i, v] are vectors of length n
            # S[i, u] is scalar S[i, u]
            # S[i, v] is vector S[i, :]
            
            # Let's vectorize over j
            # S[i, :] is shape (n,)
            # S[:, u] is shape (n,)
            # S[:, v] is shape (n,) but v depends on j, so S[j, u] is S[:, u][j]
            
            # Term 1: 2 * (S[i, v] - S[i, u])
            # S[i, v] is S[i, :]
            # S[i, u] is scalar
            t1 = 2.0 * (S[i, :] - S[i, u])
            
            # Term 2: 2 * (S[j, u] - S[j, v])
            # S[j, u] is S[:, u][j]
            # S[j, v] is S[:, p[j]][j]? No, S[j, v] where v=p[j].
            # S[j, p[j]] is the diagonal of S? No.
            # S[j, l] = sum_k a[j, k] b[p[k], l].
            # If l = p[j], S[j, p[j]] = sum_k a[j, k] b[p[k], p[j]].
            # This is not simply S[:, u] or S[:, v].
            
            # Wait, S[j, v] where v = p[j].
            # We need S[j, p[j]] for all j.
            # Let D = np.diag(S)? No.
            # S[j, p[j]] is the element at row j, column p[j].
            # We can extract this: S_diag_p = S[np.arange(n), p]
            
            # So Term 2 for j is: 2 * (S[:, u][j] - S_diag_p[j])
            # But v depends on j.
            # S[j, v] = S[j, p[j]] = S_diag_p[j].
            # S[j, u] = S[:, u][j].
            
            t2 = 2.0 * (S[:, u] - S[np.arange(n), p])
            
            # Term 3: a_ii * (b_vv - 2b_uv + b_uu)
            # b_vv = b[p[j], p[j]]
            # b_uv = b[u, p[j]]
            # b_uu = b[u, u]
            b_uu = b[u, u]
            b_uv = b[u, :]  # shape (n,)
            b_vv = b[np.arange(n), p] # b[p[j], p[j]]
            
            t3 = a[i, i] * (b_vv - 2.0 * b_uv + b_uu)
            
            # Term 4: a_jj * (b_uu - 2b_uv + b_vv)
            a_jj = np.diag(a) # shape (n,)
            t4 = a_jj * (b_uu - 2.0 * b_uv + b_vv)
            
            # Term 5: a_ij * (2b_uv - b_vv - b_uu)
            a_ij = a[i, :] # shape (n,)
            t5 = a_ij * (2.0 * b_uv - b_vv - b_uu)
            
            deltas = t1 + t2 + t3 + t4 + t5
            
            # Exclude i
            deltas[i] = np.inf
            
            best_j = np.argmin(deltas)
            best_delta = deltas[best_j]
            
            if best_delta < -1e-9:
                # Perform swap
                j = best_j
                v = p[j]
                
                # Update p
                p[i], p[j] = v, u
                
                # Update M (swap rows i and j)
                M[i], M[j] = M[j], M[i].copy()
                
                # Update S
                # S_new[k, :] = S_old[k, :] + A[k, i] (M[j, :] - M[i, :]) + A[k, j] (M[i, :] - M[j, :])
                # Note: M[i] and M[j] are now swapped in M, so we must use the OLD M[i], M[j] for the delta?
                # No, the formula derived was:
                # S_new[k, :] = S_old[k, :] + A[k, i] (M_old[j, :] - M_old[i, :]) + A[k, j] (M_old[i, :] - M_old[j, :])
                # Let's store old M rows before swapping.
                # I swapped M[i] and M[j] above.
                # So M[i] is now old M[j], M[j] is now old M[i].
                # Diff = M[i] - M[j] = old M[j] - old M[i].
                # S_new = S_old + A[:, i] * Diff + A[:, j] * (-Diff)
                # S_new = S_old + (A[:, i] - A[:, j]) * Diff
                
                diff_M = M[i] - M[j]
                S += (a[:, i] - a[:, j])[:, np.newaxis] * diff_M[np.newaxis, :]
                
                improved = True
        
        # If no improvement in a full pass, break
        if not improved:
            break
            
    return p.tolist()
# EVOLVE-BLOCK-END
