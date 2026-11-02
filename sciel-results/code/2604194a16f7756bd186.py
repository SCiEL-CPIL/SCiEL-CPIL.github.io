"""Matrix multiplication 2x5x6 - rank-minimal tensor decomposition search."""
import numpy as np
import time
import itertools

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook as fallback
    rank_school = n * m * p
    U_sb = np.zeros((n*m, rank_school))
    V_sb = np.zeros((m*p, rank_school))
    W_sb = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    start_time = time.time()
    deadline = start_time + 130  # safety margin
    
    # Try to find lower rank decompositions
    # The tensor T has T[i,j,k]=1 for all i in [0,n), j in [0,m), k in [0,p)
    # Rank-r decomposition: T = sum_c U[:,c] o V[:,c] o W[:,c]
    # i.e., for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Strategy: For small n, we can fix W and solve for U,V or vice versa.
    # Alternatively, use iterative methods.
    
    # Try ranks from lower bound up
    # Lower bound: rank >= max(n*m, m*p, p*n) / something, but actually rank >= n*m for "full" tensors
    # For this specific tensor, rank >= 10 (since U has 10 rows and must be "full" in some sense)
    # Actually, the rank is at least the dimension of the space spanned by slices.
    
    # Let's try a structured approach:
    # Since n=2, W has 12 rows. Let's try to find W first, then solve for U,V.
    
    # For a given W (12 x r), the constraint becomes:
    # For each (j,k): sum_i sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # This is complex. Let me try a different parameterization.
    
    # Alternative: use the fact that this is a bilinear map. 
    # C[i,k] = sum_j A[i,j] B[j,k]
    # We want to compute all C[i,k] using r scalar multiplications.
    
    # Let me try random initialization + gradient descent on the residual
    # with the constraint that we want exact (or near-exact) decomposition.
    
    # Actually, let's try a more structured search.
    # For rank r, we have r terms. Each term is U_c (length 10) o V_c (length 30) o W_c (length 12).
    # The sum must equal the all-ones tensor.
    
    # Let me try: fix the W matrix to have a specific structure.
    # Since n=2, W has 12 entries per column. Let's try W with entries in {-1, 0, 1} or {-0.5, 0, 0.5, 1}.
    
    # Better idea: use the known result that for 2x2x2, Strassen gives rank 7 < 8.
    # For 2x5x6, we can try to embed Strassen-like decompositions.
    
    # Let me try a direct approach: assume a specific form for the decomposition.
    # Since all entries of T are 1, maybe we can find a decomposition where
    # the columns of U, V, W have simple patterns.
    
    # Try: for each possible rank r from 10 to 60, try random initializations
    # and see if we can drive the residual to zero.
    
    np.random.seed(42)
    
    for r in range(10, min(61, rank_school + 1)):
        if time.time() > deadline:
            break
        # Try several random initializations
        for trial in range(20):
            if time.time() > deadline:
                break
            # Random initialization
            U_r = np.random.randn(n*m, r) * 0.5
            V_r = np.random.randn(m*p, r) * 0.5
            W_r = np.random.randn(p*n, r) * 0.5
            
            # Iteratively refine: for each term, try to minimize residual
            # Use a simple iterative method: fix two matrices, optimize the third.
            
            residual = 1.0
            for it in range(100):
                if time.time() > deadline:
                    break
                # Compute current tensor from U_r, V_r, W_r
                # T_computed[i,j,k] = sum_c U_r[i*m+j, c] * V_r[j*p+k, c] * W_r[k*n+i, c]
                
                # Optimize W_r: for each (i,k), we need sum_c U_r[i*m+j,c]*V_r[j*p+k,c]*W_r[k*n+i,c] summed over j = 1
                # This is: for each (i,k), sum_j sum_c (U_r[i*m+j,c] * V_r[j*p+k,c]) * W_r[k*n+i,c] = 1
                # Let M_ik[c] = sum_j U_r[i*m+j,c] * V_r[j*p+k,c]
                # Then sum_c M_ik[c] * W_r[k*n+i,c] = 1 for all i,k
                # This is a system: M_ik (1 x r) dot W_r[k*n+i, :] = 1
                # So M_all (12 x r) dot W_r^T = ones(12)
                # Solve for W_r: W_r^T = M_all^T (M_all M_all^T)^{-1} ones(12) if possible
                
                M_all = np.zeros((p*n, r))
                for i in range(n):
                    for k in range(p):
                        # M_ik[c] = sum_j U_r[i*m+j, c] * V_r[j*p+k, c]
                        for j in range(m):
                            M_all[k*n+i, :] += U_r[i*m+j, :] * V_r[j*p+k, :]
                
                # Solve M_all @ W_r.T = ones(p*n)
                try:
                    # W_r.T = M_all.T @ pinv(M_all @ M_all.T) @ ones
                    MMT = M_all @ M_all.T
                    # Regularize
                    MMT_reg = MMT + 1e-6 * np.eye(p*n)
                    W_new_T = M_all.T @ np.linalg.solve(MMT_reg, np.ones(p*n))
                    W_r = W_new_T.T
                except:
                    break
                
                # Similarly optimize V_r: for each (j,k), sum_i sum_c U_r[i*m+j,c]*V_r[j*p+k,c]*W_r[k*n+i,c] = 1
                # For fixed j: sum_i sum_c U_r[i*m+j,c]*W_r[k*n+i,c] * V_r[j*p+k,c] = 1 for all k
                # Let N_jk[c] = sum_i U_r[i*m+j,c] * W_r[k*n+i,c]
                # Then sum_c N_jk[c] * V_r[j*p+k,c] = 1 for all j,k
                # N_all (m*p x r), V_r (m*p x r): N_all @ V_r.T = ones(m*p)
                
                N_all = np.zeros((m*p, r))
                for j in range(m):
                    for k in range(p):
                        for i in range(n):
                            N_all[j*p+k, :] += U_r[i*m+j, :] * W_r[k*n+i, :]
                
                try:
                    NNT = N_all @ N_all.T
                    NNT_reg = NNT + 1e-6 * np.eye(m*p)
                    V_new_T = N_all.T @ np.linalg.solve(NNT_reg, np.ones(m*p))
                    V_r = V_new_T.T
                except:
                    break
                
                # Optimize U_r: for each (i,j), sum_k sum_c U_r[i*m+j,c]*V_r[j*p+k,c]*W_r[k*n+i,c] = 1
                # For fixed i: sum_j sum_k sum_c U_r[i*m+j,c]*V_r[j*p+k,c]*W_r[k*n+i,c] = 1
                # Let P_ij[c] = sum_k V_r[j*p+k,c] * W_r[k*n+i,c]
                # Then sum_c P_ij[c] * U_r[i*m+j,c] = 1 for all i,j
                # P_all (n*m x r), U_r (n*m x r): P_all @ U_r.T = ones(n*m)
                
                P_all = np.zeros((n*m, r))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            P_all[i*m+j, :] += V_r[j*p+k, :] * W_r[k*n+i, :]
                
                try:
                    PPT = P_all @ P_all.T
                    PPT_reg = PPT + 1e-6 * np.eye(n*m)
                    U_new_T = P_all.T @ np.linalg.solve(PPT_reg, np.ones(n*m))
                    U_r = U_new_T.T
                except:
                    break
                
                # Compute residual
                T_computed = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_computed[i,j,k] = np.sum(U_r[i*m+j, :] * V_r[j*p+k, :] * W_r[k*n+i, :])
                residual = np.max(np.abs(T_computed - 1.0))
                
                if residual < 1e-6:
                    break
            
            if residual < 1e-6:
                # Found a valid decomposition at rank r
                best_rank = r
                best_U, best_V, best_W = U_r, V_r, W_r
                break  # No need to try higher ranks
        
        if best_rank <= r:
            break
    
    # Convert to lists with half-integer rounding if needed
    def to_list(mat):
        # Round to nearest 0.5
        rounded = np.round(mat * 2) / 2
        return rounded.tolist()
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}

