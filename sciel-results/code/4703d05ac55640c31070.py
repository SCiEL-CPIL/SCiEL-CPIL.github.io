"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def solve(a, b):
    n = len(a)
    af = a.astype(np.float64)
    bf = b.astype(np.float64)
    
    # Initialize with rank-matching
    heavy = np.argsort(-af.sum(axis=1), kind="stable")
    central = np.argsort(bf.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # Compute S matrix: S[i, l] = sum_{k != i} (A[i,k] B[l, p[k]] + A[k,i] B[p[k], l])
    # S is n x n
    S = np.zeros((n, n), dtype=np.float64)
    
    # Efficient computation of S
    # S[i, l] = sum_k A[i,k] B[l, p[k]] + sum_k A[k,i] B[p[k], l] - (terms where k=i)
    # Term 1: (A @ B[:, p])? No.
    # Let M1[i, l] = sum_k A[i,k] B[l, p[k]]
    # B_perm = B[:, p] # Shape (n, n), B_perm[l, k] = B[l, p[k]]
    B_perm = bf[:, p]
    # M1 = A @ B_perm? 
    # (A @ B_perm)[i, k] = sum_m A[i,m] B_perm[m, k] = sum_m A[i,m] B[m, p[k]]. 
    # We want sum_m A[i,m] B[l, p[m]].
    # This is A @ B_perm.T? 
    # (A @ B_perm.T)[i, l] = sum_m A[i,m] B_perm.T[m, l] = sum_m A[i,m] B_perm[l, m] = sum_m A[i,m] B[l, p[m]].
    # Yes.
    M1 = af @ B_perm.T
    
    # Term 2: M2[i, l] = sum_k A[k,i] B[p[k], l]
    # Let B_perm2 = B[p, :] # Shape (n, n), B_perm2[k, l] = B[p[k], l]
    B_perm2 = bf[p, :]
    # M2[i, l] = sum_k A[k,i] B_perm2[k, l]
    # This is (A.T @ B_perm2)[i, l] = sum_k A.T[i,k] B_perm2[k, l] = sum_k A[k,i] B[p[k], l].
    M2 = af.T @ B_perm2
    
    S = M1 + M2
    
    # Subtract self terms (k=i)
    # For each i, S[i, l] included A[i,i] B[l, p[i]] + A[i,i] B[p[i], l]
    # We defined S[i, l] for k != i.
    # So we must subtract these.
    # diag_A = A[i, i]
    # B_l_pi = B[l, p[i]]
    # B_pi_l = B[p[i], l]
    # S[i, :] -= A[i,i] * (B[:, p[i]] + B[p[i], :])
    for i in range(n):
        S[i, :] -= af[i, i] * (bf[:, p[i]] + bf[p[i], :])
        
    # Initial cost
    current_cost = np.sum(S[np.arange(n), p])
    best_cost = current_cost
    best_p = p.copy()
    
    start_time = time.time()
    time_limit = 115
    
    while time.time() - start_time < time_limit:
        best_delta = 0
        best_i = -1
        best_j = -1
        
        # Evaluate all swaps i < j
        # Delta_ij = S[i, p[j]] + S[j, p[i]] - S[i, p[i]] - S[j, p[j]] + Corr
        # We iterate i, and vectorize j.
        
        for i in range(n):
            u = p[i]
            # We want Delta[i, j] for all j != i
            # Let's create an array of j's
            js = np.arange(n)
            js = js[js != i]
            vs = p[js] # v = p[j]
            
            # S[i, v] -> S[i, vs]
            term1 = S[i, vs]
            # S[j, u] -> S[js, u]
            term2 = S[js, u]
            # S[i, u] -> S[i, u] (scalar)
            term3 = S[i, u]
            # S[j, v] -> S[js, vs]
            term4 = S[js, vs]
            
            base_delta = term1 + term2 - term3 - term4
            
            # Correction terms
            # A[i, j] B[u, v] * 4 - A[i, j] (B[v, v] + B[u, u]) * 2  (assuming symmetric, but let's use general)
            # General:
            # - (A[i,j] + A[j,i]) (B[v,v] + B[u,u])
            # + 2 (A[i,j] B[u,v] + A[j,i] B[v,u])
            # + (A[i,j] - A[j,i]) (B[v,u] - B[u,v])
            
            Ai_js = af[i, js]
            Aj_is = af[js, i]
            
            Bu_v = bf[u, vs]
            Bv_u = bf[vs, u]
            Bu_u = bf[u, u]
            Bv_v = bf[vs, vs]
            
            corr1 = -(Ai_js + Aj_is) * (Bv_v + Bu_u)
            corr2 = 2 * (Ai_js * Bu_v + Aj_is * Bv_u)
            corr3 = (Ai_js - Aj_is) * (Bv_u - Bu_v)
            
            deltas = base_delta + corr1 + corr2 + corr3
            
            # Find best delta in this row
            min_idx = np.argmin(deltas)
            min_val = deltas[min_idx]
            
            if min_val < best_delta:
                best_delta = min_val
                best_i = i
                best_j = js[min_idx]
                
        if best_i == -1:
            break
            
        # Perform swap
        i, j = best_i, best_j
        u, v = p[i], p[j]
        
        p[i], p[j] = v, u
        current_cost += best_delta
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            
        # Update S
        # S[k, l] changes for all k, l.
        # Delta S[k, l] = A[k,i](B[l, v] - B[l, u]) + A[i,k](B[v, l] - B[u, l]) + A[k,j](B[l, u] - B[l, v]) + A[j,k](B[u, l] - B[v, l])
        
        # Vectorized update
        # Let D = B[:, v] - B[:, u]  (Shape n, D[l] = B[l, v] - B[l, u])
        # Let E = B[v, :] - B[u, :]  (Shape n, E[l] = B[v, l] - B[u, l])
        
        D = bf[:, v] - bf[:, u]
        E = bf[v, :] - bf[u, :]
        
        # Term for k interacting with i (which moved u->v)
        # A[k, i] * D[l] + A[i, k] * E[l]
        # This adds to S[k, l] for all k.
        # But for k=i and k=j, the formula is slightly different because p[i] and p[j] changed.
        # Actually, the formula derived:
        # For k != i, j:
        # Delta S[k, l] = A[k,i](B[l, v] - B[l, u]) + A[i,k](B[v, l] - B[u, l]) + A[k,j](B[l, u] - B[l, v]) + A[j,k](B[u, l] - B[v, l])
        # Note: B[l, u] - B[l, v] = -D[l]. B[u, l] - B[v, l] = -E[l].
        # So Delta S[k, l] = (A[k,i] - A[k,j]) D[l] + (A[i,k] - A[j,k]) E[l]
        
        # For k = i:
        # S[i, l] = sum_{m != i} A[i,m] B[l, p[m]] + A[m,i] B[p[m], l]
        # p[j] changed from v to u.
        # Delta S[i, l] = A[i,j](B[l, u] - B[l, v]) + A[j,i](B[u, l] - B[v, l])
        # = -A[i,j] D[l] - A[j,i] E[l]
        
        # For k = j:
        # p[i] changed from u to v.
        # Delta S[j, l] = A[j,i](B[l, v] - B[l, u]) + A[i,j](B[v, l] - B[u, l])
        # = A[j,i] D[l] + A[i,j] E[l]
        
        # So we can update S for all k using the general formula, then fix i and j?
        # General formula for k:
        # Delta S[k, l] = A[k,i] D[l] + A[i,k] E[l] + A[k,j] (-D[l]) + A[j,k] (-E[l])
        # = (A[k,i] - A[k,j]) D[l] + (A[i,k] - A[j,k]) E[l]
        
        # Let's check k=i with this formula:
        # (A[i,i] - A[i,j]) D[l] + (A[i,i] - A[j,i]) E[l]
        # We derived: -A[i,j] D[l] - A[j,i] E[l]
        # Difference: A[i,i] D[l] + A[i,i] E[l].
        # Why? Because in S[i, l], the term for m=i is excluded.
        # In the "sum over m" derivation, we assumed m ranges over all.
        # So for k=i, we must add back the self term change?
        # Actually, S[i, l] definition excludes m=i.
        # The update formula derived from "sum over m != k" is valid for k=i if we just consider the change in p[j].
        # The change in p[j] affects S[i, l] only through the term m=j.
        # So Delta S[i, l] = A[i,j](B[l, u] - B[l, v]) + A[j,i](B[u, l] - B[v, l]).
        # The general formula gives: (A[i,i] - A[i,j]) D[l] + (A[i,i] - A[j,i]) E[l].
        # This is incorrect for k=i.
        
        # So:
        # 1. Update S for all k using general formula: S += (A[:, i] - A[:, j])[:, None] * D[None, :] + (A[i, :] - A[j, :])[None, :] * E[None, :]
        # 2. Correct S[i, :] and S[j, :]
        
        # General update
        coeff_D = af[:, i] - af[:, j]
        coeff_E = af[i, :] - af[j, :]
        
        S += coeff_D[:, None] * D[None, :] + coeff_E[None, :] * E[None, :]
        
        # Correct i
        # We added: (A[i,i] - A[i,j]) D + (A[i,i] - A[j,i]) E
        # We want: -A[i,j] D - A[j,i] E
        # Difference: A[i,i] D + A[i,i] E
        # So subtract A[i,i] (D + E) from S[i, :]
        S[i, :] -= af[i, i] * (D + E)
        
        # Correct j
        # We added: (A[j,i] - A[j,j]) D + (A[j,i] - A[j,j])? No.
        # coeff_D[j] = A[j,i] - A[j,j].
        # coeff_E[j] = A[i,j] - A[j,j].
        # Added: (A[j,i] - A[j,j]) D + (A[i,j] - A[j,j]) E
        # We want: A[j,i] D + A[i,j] E
        # Difference: -A[j,j] D - A[j,j] E
        # So add A[j,j] (D + E) to S[j, :]?
        # Wait.
        # Desired: A[j,i] D + A[i,j] E
        # Added: (A[j,i] - A[j,j]) D + (A[i,j] - A[j,j]) E
        # Added = Desired - A[j,j] D - A[j,j] E
        # So S[j, :] currently has Desired - A[j,j](D+E).
        # We want Desired. So add A[j,j](D+E).
        S[j, :] += af[j, j] * (D + E)
        
    return best_p.tolist()
# EVOLVE-BLOCK-END
