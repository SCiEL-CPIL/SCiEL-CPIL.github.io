"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # The tensor for matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    # We need U (n*m x rank), V (m*p x rank), W (p*n x rank) such that:
    # For each output position (i,k), the coefficient of A[i,j]*B[j,k] is 1
    # i.e., sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Strategy: Use a construction that exploits the small dimensions.
    # For 2x3x6, we can try ranks from 18 upward (Landsberg bound suggests ~18).
    
    # Let's build a structured decomposition.
    # Key idea: group the 6 columns of B into pairs or use linear combinations.
    
    # Attempt: rank 18 construction
    # Split the 6 k-indices into 3 groups of 2.
    # For each group of 2 k-values, we can use 2 rank-1 tensors per (i,j) pair? 
    # That would be 2*3*3 = 18.
    
    # More carefully: 
    # For fixed i, j: we need the contribution to all 6 k's.
    # If we use a "block" approach where for each (i,j) we allocate some ranks 
    # to handle all 6 k-values simultaneously using linear combinations.
    
    # Let me try: for each i (2 values), j (3 values), we need to cover k=0..5.
    # If we use 3 ranks per (i,j) pair, that's 18 total.
    # Each rank handles 2 k-values via a linear combination.
    
    # Construction:
    # For rank r corresponding to (i, j, k_group) where k_group in {0,1,2}:
    #   U[i*m+j, r] = 1
    #   V[j*p + k0, r] = 1, V[j*p + k1, r] = 1  (for the two k's in the group)
    #   W[k0*n+i, r] = 1, W[k1*n+i, r] = 1
    # But this gives coefficient 2 for those k's, not 1.
    
    # Better: use 1/2 entries.
    #   U[i*m+j, r] = 1
    #   V[j*p + k0, r] = 1/2, V[j*p + k1, r] = 1/2
    #   W[k0*n+i, r] = 1, W[k1*n+i, r] = 1
    # Then contribution to k0: 1 * (1/2) * 1 = 1/2. Not 1.
    
    # Hmm, need to be more careful. The product U*V*W must equal 1.
    
    # Let me think differently. The tensor T[i,j,k] = 1 (coefficient in the trilinear form).
    # We want T = sum_r u_r ⊗ v_r ⊗ w_r where:
    #   u_r[i,j], v_r[j,k], w_r[k,i]
    # and sum_r u_r[i,j] * v_r[j,k] * w_r[k,i] = 1 for all i,j,k.
    
    # For rank 18: let's try a systematic approach.
    # Since n=2, m=3, p=6, note that 6 = 2*3.
    # Maybe use the structure: for each i (2), the 6 k's can be paired with j's.
    
    # Alternative: use the fact that this is a "thin" matrix multiplication.
    # Let me just search for the minimal rank by trying constructions.
    
    # Actually, let me try a direct construction at rank 18.
    # Label ranks by (i, j, s) where i in {0,1}, j in {0,1,2}, s in {0,1,2}.
    # For rank r = (i, j, s):
    #   U[i*m+j, r] = 1
    #   V[j*p + (2*s), r] = 1, V[j*p + (2*s+1), r] = 1  -- no, this couples k's
    #   W[(2*s)*n+i, r] = 1, W[(2*s+1)*n+i, r] = 1
    # This gives for fixed (i,j,k): if k = 2s or 2s+1, contribution is 1*1*1 = 1. 
    # But if k = 2s' for s' != s, no contribution. So each (i,j,k) gets exactly one 
    # rank contributing to it. That works! Rank = 2*3*3 = 18.
    
    # Wait, let me verify: for (i,j,k) with k = 2s or 2s+1:
    #   The rank (i,j,s) has U[i*m+j] = 1, V[j*p+k] = 1, W[k*n+i] = 1.
    #   Product = 1. 
    #   Any other rank (i',j',s'): if i'!=i, U[i*m+j, r]=0. If j'!=j, V[j*p+k,r]=0.
    #   If s'!=s, then V[j*p+k, r'] = 0 (since k is not in group s').
    #   So exactly one term contributes, giving 1. Correct!
    
    # But wait - this is essentially the schoolbook with k split into groups of 2.
    # The schoolbook has rank 2*3*6 = 36. This construction has rank 2*3*3 = 18.
    # Is this valid? Let me double-check the W indexing.
    
    # W index is k*n + i. For rank (i,j,s), W[(2s)*n+i] = 1 and W[(2s+1)*n+i] = 1.
    # For a given (i,j,k), we need sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1.
    # The only r with U[i*m+j,r]=1 AND V[j*p+k,r]=1 is r=(i,j,s) where s = k//2.
    # For that r, W[k*n+i, r] = 1 (since k is in group s). So product = 1. ✓
    
    # This gives rank 18. Can we do better?
    
    # Let's also verify no lower rank works by trying to construct rank 17 or lower.
    # For now, let's go with 18 and verify it's correct.
    
    rank = 18
    U = [[0.0] * rank for _ in range(n * m)]  # 6 x 18
    V = [[0.0] * rank for _ in range(m * p)]  # 18 x 18
    W = [[0.0] * rank for _ in range(p * n)]  # 12 x 18
    
    col = 0
    for i in range(n):
        for j in range(m):
            for s in range(3):  # 3 groups of 2 k-values
                k0 = 2 * s
                k1 = 2 * s + 1
                U[i * m + j][col] = 1.0
                V[j * p + k0][col] = 1.0
                V[j * p + k1][col] = 1.0
                W[k0 * n + i][col] = 1.0
                W[k1 * n + i][col] = 1.0
                col += 1
    
    # Verify the decomposition
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
    # Check: for all i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s_val = sum(U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r] 
                           for r in range(rank))
                if abs(s_val - 1.0) > 1e-9:
                    # Fallback to schoolbook
                    rank_fb = n * m * p
                    U_fb = [[0.0] * rank_fb for _ in range(n * m)]
                    V_fb = [[0.0] * rank_fb for _ in range(m * p)]
                    W_fb = [[0.0] * rank_fb for _ in range(p * n)]
                    c = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U_fb[ii * m + jj][c] = 1.0
                                V_fb[jj * p + kk][c] = 1.0
                                W_fb[kk * n + ii][c] = 1.0
                                c += 1
                    return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
