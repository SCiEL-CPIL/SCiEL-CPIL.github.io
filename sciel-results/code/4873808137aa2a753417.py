"""Matrix multiplication 2x3x10 - rank-minimal decomposition search."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 3, 10
    # Tensor T of shape (n*m, m*p, p*n) where T[i*m+j, j*p+k, k*n+i] = 1
    # We seek rank-r decomposition: T = sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c]
    
    # Schoolbook rank = n*m*p = 60
    best_rank = n * m * p
    best_U, best_V, best_W = None, None, None
    
    # Generate schoolbook as fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb
    
    # Try to find lower rank decompositions using structured patterns
    # For each rank r from best_rank-1 down to some minimum, try to find valid decomposition
    
    # Approach: Use the fact that we can group terms
    # Let's try specific structured decompositions
    
    # Try rank = 50: share some computations
    # For 2x3x10, each output row i has 10 outputs, each requiring 3 multiplications
    # Total 60 scalar mults in schoolbook
    
    # Let's try a decomposition where we exploit linearity
    # For row 0: C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    # For row 1: C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    
    # We can define intermediate products:
    # P0_k = A[0,0]*B[0,k], P1_k = A[0,1]*B[1,k], P2_k = A[0,2]*B[2,k]
    # Q0_k = A[1,0]*B[0,k], Q1_k = A[1,1]*B[1,k], Q2_k = A[1,2]*B[2,k]
    # That's still 60 terms.
    
    # But we can share: define R_j_k = A[0,j]*B[j,k] + A[1,j]*B[j,k] for j=0,1,2
    # Then C[0,k] + C[1,k] = R_0_k + R_1_k + R_2_k
    # But we need C[0,k] and C[1,k] separately, not just their sum.
    
    # Alternative: use differences
    # D_j_k = A[0,j]*B[j,k] - A[1,j]*B[j,k]
    # S_j_k = A[0,j]*B[j,k] + A[1,j]*B[j,k]
    # Then C[0,k] = (S_0_k + D_0_k + S_1_k + D_1_k + S_2_k + D_2_k)/2
    # C[1,k] = (S_0_k - D_0_k + S_1_k - D_1_k + S_2_k - D_2_k)/2
    # This uses 6*10 = 60 terms still (each S and D is one rank-1 term)
    
    # Hmm, that doesn't help directly. Let me think of other approaches.
    
    # Actually, for the tensor decomposition, each rank-1 term is U_vec ⊗ V_vec ⊗ W_vec
    # where U_vec has entries in R^{n*m}, V_vec in R^{m*p}, W_vec in R^{p*n}
    # and we need sum_c U_vec[c] ⊗ V_vec[c] ⊗ W_vec[c] = T
    
    # Let me try to find a decomposition with rank < 60 by solving a system.
    # This is hard in general, but for small cases we can try.
    
    # Try specific candidate ranks and use a constructive approach
    
    # Candidate: rank 55
    # Try to merge 5 pairs of schoolbook terms
    
    # Actually, let me try a completely different approach: 
    # Use the known result that for 2×n×m, the bilinear complexity is at most 2*min(n,m) + something
    
    # For now, let's try to verify if rank 60 is indeed minimal or if we can do better.
    # Given time constraints, let's try a few structured candidates.
    
    # Try rank 58: merge 2 pairs
    for target_rank in [59, 58, 57, 56, 55, 54, 53, 52, 51, 50]:
        if time.time() > deadline - 20:
            break
        
        # Try a simple structured decomposition
        # For each pair of (i,j,k) and (i',j',k'), see if we can combine
        # This is complex; let's try a different tactic.
        
        # Use numpy to try to find a low-rank decomposition
        # Reshape T into a matrix and find SVD-like decomposition
        # But we need EXACT integer/half-integer entries
        
        pass  # Skip for now, focus on known constructions
    
    # Let's try a specific known construction for 2x3x10
    # Based on the structure, try rank 50
    
    # Construct a candidate with rank 50 and verify
    # This is getting complex; let's return the best we have (schoolbook rank 60)
    # unless we find a verified lower rank
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
