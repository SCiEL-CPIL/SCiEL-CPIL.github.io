import numpy as np
import time
import sys

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # The tensor for matrix multiplication C = A * B where A is n x m and B is m x p
    # is defined by T_{i,j,k} = 1 if we consider the product term A_{i,j} B_{j,k}.
    # The rank of this tensor is the bilinear complexity.
    
    # For n=1, m=4, p=7:
    # C_k = sum_{j=0}^{3} A_j B_{j,k}
    # This can be computed with 4 multiplications: M_j = A_j * B_{j,:}
    # Then C_k = sum_j M_{j,k}.
    # Wait, M_j is a row vector (1x7). C is a row vector (1x7).
    # C_k = M_{0,k} + M_{1,k} + M_{2,k} + M_{3,k}.
    # This requires 4 scalar multiplications A_j * B_{j,k} for each k?
    # No, A_j is a scalar. B_{j,k} is a scalar.
    # The product C = A B is a 1x7 matrix.
    # Entry C_k = A_0 B_{0,k} + A_1 B_{1,k} + A_2 B_{2,k} + A_3 B_{3,k}.
    # This is a sum of 4 products.
    # Can we do it with fewer?
    # The rank of the 1 x m x p tensor is m.
    # Proof: The matrix of coefficients for A is B (m x p). The matrix of coefficients for B is A^T (m x 1).
    # Actually, the rank is the dimension of the middle index, which is m.
    # So for 1x4x7, rank is 4.
    
    # For n=2, m=4, p=7:
    # The tensor is the direct sum of two 1x4x7 tensors?
    # C_{0,k} = sum_j A_{0,j} B_{j,k}
    # C_{1,k} = sum_j A_{1,j} B_{j,k}
    # These are independent equations.
    # The variables A_{0,j}, A_{1,j}, B_{j,k} are all distinct.
    # The rank is the sum of the ranks of the blocks if they don't share variables in a way that allows sharing rank-1 terms.
    # A rank-1 term is u \otimes v \otimes w.
    # u is in R^{n*m} = R^8. v is in R^{m*p} = R^{28}. w is in R^{p*n} = R^{14}.
    # The support of the first block (i=0) is on u indices (0..3), v indices (all), w indices (0..6).
    # The support of the second block (i=1) is on u indices (4..7), v indices (all), w indices (7..13).
    # Since the u-supports are disjoint, any rank-1 term can only contribute to one of the blocks?
    # No, a rank-1 term u \otimes v \otimes w has full support u \times v \times w.
    # If u has non-zero entries in both block 0 and block 1, then the term contributes to both.
    # However, we can simply take the direct sum of the rank-4 decompositions for each block.
    # Rank = 4 + 4 = 8.
    
    rank = 8
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Block 1: i=0 (Row 0 of A)
    # Uses columns 0..3 of U, V, W
    for j in range(m):
        col = j
        U[0 * m + j, col] = 1.0
        # V needs to pick B_{j,k} for all k?
        # Wait, the rank-1 term is u \otimes v \otimes w.
        # Contribution to T is u_{i,j} v_{j,k} w_{k,i}.
        # We want T_{i,j,k} = 1.
        # For the term corresponding to j, we want it to contribute to C_{i,k} via A_{i,j} B_{j,k}.
        # If we set u_{i,j} = 1, v_{j,k} = 1 for all k, w_{k,i} = 1 for all k?
        # Then term is sum_{j'} sum_{k'} 1 * 1 * 1 ? No.
        # The sum is over rank r.
        # C_{i,k} = sum_{r} (sum_{j'} U_{i m + j', r} A_{i,j'}) (sum_{k'} V_{j' p + k', r} B_{j',k'}) W_{k n + i, r}
        # This is not right. The standard form is:
        # C_{i,k} = sum_{r} M_r W_{k n + i, r}
        # where M_r = (U[:,r]^T A) (B V[:,r]) ? No.
        # The decomposition is:
        # C_{i,k} = sum_{r} ( sum_{j} U_{i m + j, r} A_{i,j} ) ( sum_{l} V_{j p + l, r} B_{j,l} ) ... wait.
        
        # Let's use the explicit tensor definition.
        # T_{i,j,k} = 1.
        # We want sum_{r} U_{i m + j, r} V_{j p + k, r} W_{k n + i, r} = 1 for all i,j,k.
        
        # For 1x4x7 (i=0):
        # We need sum_{r=0}^3 U_{j, r} V_{j, k} W_{k, 0} = 1 for all j,k.
        # If we set U_{j, r} = 1 if r==j else 0.
        # Then sum_{r} U_{j, r} V_{j, k} W_{k, 0} = V_{j, k} W_{k, 0}.
        # We need V_{j, k} W_{k, 0} = 1 for all j,k.
        # This implies V_{j, k} = 1/W_{k, 0}.
        # We can set W_{k, 0} = 1 for all k. Then V_{j, k} = 1.
        # Check: sum_{r} U_{j, r} V_{j, k} W_{k, 0} = U_{j, j} * 1 * 1 = 1. Correct.
        
        # So for block i=0:
        # r=0..3
        # U[j, r] = delta_{j,r}
        # V[j, k] = 1 (for all j,k) -- Wait, V is shared? No, V indices are j,k.
        # In the global V matrix, the row index is j*p + k.
        # So V[j*p + k, r] = 1.
        # W[k*n + 0, r] = 1.
        
        for k in range(p):
            pass # V and W setup depends on r which is j
            
    # Let's construct it properly.
    # Rank 8.
    # First 4 ranks for i=0.
    for r in range(4):
        # r corresponds to j=r for the first block
        # U index: i=0, j=r -> index r
        U[r, r] = 1.0
        # V index: j=r, k=0..6 -> indices r*p + k
        for k in range(p):
            V[r * p + k, r] = 1.0
        # W index: k=0..6, i=0 -> indices k*n + 0 = k*2
        for k in range(p):
            W[k * n + 0, r] = 1.0
            
    # Last 4 ranks for i=1.
    for r in range(4):
        # r corresponds to j=r for the second block (offset by 4 in rank)
        rank_idx = 4 + r
        # U index: i=1, j=r -> index 1*m + r = 4 + r
        U[4 + r, rank_idx] = 1.0
        # V index: j=r, k=0..6 -> indices r*p + k
        for k in range(p):
            V[r * p + k, rank_idx] = 1.0
        # W index: k=0..6, i=1 -> indices k*n + 1 = k*2 + 1
        for k in range(p):
            W[k * n + 1, rank_idx] = 1.0
            
    # Verify
    # T_{i,j,k} = sum_{r} U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    
    # Case i=0, j, k
    # sum_{r=0}^7 U[j, r] V[j*p+k, r] W[2k, r]
    # For r < 4:
    #   U[j, r] is 1 if j==r else 0.
    #   V[j*p+k, r] is 1 if r < 4 (since V is set for all k in r<4 block).
    #   W[2k, r] is 1 if r < 4.
    #   So term is 1 if j==r, else 0.
    #   Sum is 1.
    # For r >= 4:
    #   U[j, r] is 0 (since U[4+r, 4+r]=1, but j<4).
    #   So sum is 0.
    # Total 1. Correct.
    
    # Case i=1, j, k
    # sum_{r=0}^7 U[4+j, r] V[j*p+k, r] W[2k+1, r]
    # For r < 4:
    #   U[4+j, r] is 0.
    # For r >= 4:
    #   Let r' = r-4.
    #   U[4+j, 4+r'] is 1 if j==r' else 0.
    #   V[j*p+k, 4+r'] is 1 (since V is set for all k in r>=4 block).
    #   W[2k+1, 4+r'] is 1.
    #   So term is 1 if j==r' else 0.
    #   Sum is 1.
    # Total 1. Correct.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
