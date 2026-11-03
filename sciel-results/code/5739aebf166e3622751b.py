"""QAP solver for tho150 using vectorized local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve(a, b):
    n = len(a)
    deadline = time.time() + 145.0  # Safety margin before 150s kill
    
    # Ensure matrices are float64 for faster arithmetic and to avoid overflow issues 
    # in intermediate steps, though int64 is used for final cost check if needed.
    # The problem states a and b are int64. 
    # Using float64 for delta calculation is safe as values are not huge.
    a_f = a.astype(np.float64)
    b_f = b.astype(np.float64)
    
    # Initial solution: rank matching
    # Heavy facilities (high flow sum) to central locations (low distance sum)
    a_row_sums = a_f.sum(axis=1)
    b_row_sums = b_f.sum(axis=1)
    
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # Precompute B rows and cols as views for faster access? 
    # B[x, :] and B[:, x] are contiguous in memory if B is C-contiguous.
    # We will just index B directly.
    
    # Local Search
    # We use a "First Improvement" strategy.
    # To speed up, we vectorize the delta calculation for a given pair (i, j).
    
    # We will loop i from 0 to n-1, and j from i+1 to n-1.
    # If a swap is accepted, we continue with the new p.
    
    # Optimization: Pre-compute B rows and columns? 
    # B is (n, n). B[x, :] is a row. B[:, x] is a column.
    # Accessing B[x, P] requires gathering.
    
    while time.time() < deadline - 5.0:
        improved = False
        # Optional: Randomize start order if we want to escape symmetries, 
        # but standard order is fine for first improvement.
        
        for i in range(n):
            if time.time() > deadline - 5.0:
                break
            pi = p[i]
            # Precompute vectors for location pi
            # B[pi, P] and B[P, pi]
            B_pi_P = b_f[pi, p]
            B_P_pi = b_f[p, pi]
            
            # a[i, :] and a[:, i]
            a_i = a_f[i, :]
            a_col_i = a_f[:, i]
            
            for j in range(i + 1, n):
                pj = p[j]
                if pi == pj: # Should not happen in a permutation, but safety
                    continue
                
                # Precompute vectors for location pj
                B_pj_P = b_f[pj, p]
                B_P_pj = b_f[p, pj]
                
                a_j = a_f[j, :]
                a_col_j = a_f[:, j]
                
                # Vectorized Delta Calculation
                # Terms for k not in {i, j}:
                # a[i, k] * (b[pj, p[k]] - b[pi, p[k]])  -> a_i * (B_pj_P - B_pi_P)
                # a[k, i] * (b[p[k], pj] - b[p[k], pi])  -> a_col_i * (B_P_pj - B_P_pi)
                # a[j, k] * (b[pi, p[k]] - b[pj, p[k]])  -> a_j * (B_pi_P - B_pj_P)
                # a[k, j] * (b[p[k], pi] - b[p[k], pj])  -> a_col_j * (B_P_pi - B_P_pj)
                
                diff_row = B_pj_P - B_pi_P
                diff_col = B_P_pj - B_P_pi
                
                delta_vec = (a_i * diff_row) + (a_col_i * diff_col) + (a_j * (-diff_row)) + (a_col_j * (-diff_col))
                
                # Sum over all k
                delta_sum = np.sum(delta_vec)
                
                # Correction for k=i and k=j
                # We summed terms for k=i and k=j using the "other" index fixed assumption.
                # We need to subtract the incorrect contributions for k=i and k=j from delta_sum
                # and add the correct exact delta for the 4 corner terms.
                
                # Incorrect contribution for k=i in delta_vec:
                # a[i, i]*(b[pj, pi] - b[pi, pi]) + a[i, i]*(b[pi, pj] - b[pi, pi]) + 
                # a[j, i]*(b[pi, pi] - b[pj, pi]) + a[i, j]*(b[pi, pi] - b[pi, pj])
                # Note: In delta_vec[i], the indices are:
                # a_i[i] * (B_pj_P[i] - B_pi_P[i]) = a[i,i] * (b[pj, pi] - b[pi, pi])
                # a_col_i[i] * (B_P_pj[i] - B_P_pi[i]) = a[i,i] * (b[pi, pj] - b[pi, pi])
                # a_j[i] * (B_pi_P[i] - B_pj_P[i]) = a[j,i] * (b[pi, pi] - b[pj, pi])
                # a_col_j[i] * (B_P_pi[i] - B_P_pj[i]) = a[i,j] * (b[pi, pi] - b[pi, pj])
                
                # It is easier to just compute the exact delta for the 4 terms (i,i), (i,j), (j,i), (j,j)
                # and the terms (i,k), (k,i), (j,k), (k,j) for k!=i,j are handled by the vector sum 
                # IF we remove the i and j entries from the vector sum.
                
                # Let's subtract delta_vec[i] and delta_vec[j] from delta_sum.
                delta_sum -= delta_vec[i]
                delta_sum -= delta_vec[j]
                
                # Now add exact delta for corners:
                # Old: a[i,i]*b[pi,pi] + a[i,j]*b[pi,pj] + a[j,i]*b[pj,pi] + a[j,j]*b[pj,pj]
                # New: a[i,i]*b[pj,pj] + a[i,j]*b[pj,pi] + a[j,i]*b[pi,pj] + a[j,j]*b[pi,pi]
                
                delta_corners = (
                    a_f[i, i] * (b_f[pj, pj] - b_f[pi, pi]) +
                    a_f[i, j] * (b_f[pj, pi] - b_f[pi, pj]) +
                    a_f[j, i] * (b_f[pi, pj] - b_f[pj, pi]) +
                    a_f[j, j] * (b_f[pi, pi] - b_f[pj, pj])
                )
                
                total_delta = delta_sum + delta_corners
                
                if total_delta < 0:
                    # Perform swap
                    p[i], p[j] = p[j], p[i]
                    # Update pi, pj and precomputed vectors?
                    # Since we are in a "First Improvement" loop, after a swap, 
                    # the precomputed vectors for i (which is pi) are now stale if we 
                    # intended to use them for subsequent j's in the same i-loop?
                    # No, we fixed i. p[i] changed to pj. 
                    # So pi is now the old pj.
                    # The vectors B_pi_P etc. are now wrong for the new pi.
                    # We should break the inner loop and restart the i-loop? 
                    # Or update the vectors?
                    # Updating vectors for all j is expensive.
                    # Restarting the outer loop is safer and simpler.
                    improved = True
                    break
            
            if improved:
                break
        
        if not improved:
            # Local optimum reached
            break

    return p.tolist()
# EVOLVE-BLOCK-END
