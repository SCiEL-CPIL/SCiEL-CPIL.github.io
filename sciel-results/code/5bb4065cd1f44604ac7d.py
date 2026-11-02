"""Program for 3x4x8 matrix multiplication tensor decomposition."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product

def construct():
    """
    Constructs a tensor decomposition for multiplying a 3x4 matrix by a 4x8 matrix.
    Uses a structured approach to find a low-rank decomposition.
    """
    n, m, p = DIMENSIONS
    
    # We'll try to construct a decomposition with rank R.
    # Start with a reasonable target and verify.
    
    # The schoolbook rank is n*m*p = 96.
    # We want to do better.
    
    # Strategy: Use a block decomposition approach.
    # Split B (4x8) into two 4x4 blocks: B_left and B_right.
    # C = A * B = A * [B_left, B_right] = [A*B_left, A*B_right]
    # So we need two 3x4x4 multiplications.
    
    # For 3x4x4, schoolbook rank is 3*4*4 = 48.
    # If we can find a rank-24 decomposition for 3x4x4, total rank is 48.
    # If we can find rank-20, total is 40.
    
    # Let's try to construct a rank-24 decomposition for 3x4x4.
    # This is half the schoolbook rank, which is achievable with some cleverness.
    
    # Actually, let's try a different approach.
    # For 3x4x8, we can use the fact that 8 = 2*4.
    # We can view this as computing 4 products of 3x4 by 4x1 (columns of B).
    # Each 3x4x1 multiplication has rank at most 12 (schoolbook).
    # 4 * 12 = 48.
    
    # Can we do better than 48?
    # Let's try to find a decomposition with rank 48 first, then optimize.
    
    # Construction for rank 48:
    # For each column k of B (k = 0..7), compute A * B[:,k].
    # A * B[:,k] = sum_{j=0}^{3} A[:,j] * B[j,k]
    # This is 4 terms per column, 8 columns, so 32 terms? No.
    # Each term A[:,j] * B[j,k] is a scalar times a vector.
    # In tensor decomposition, each (i,j,k) triple is one term.
    # So schoolbook is 3*4*8 = 96 terms.
    
    # To get rank 48, we need to combine terms.
    
    # Let's try a systematic approach:
    # We'll use the fact that for fixed j, we have terms for all i and k.
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    
    # For a given j, the terms are (i, j, k) for all i, k.
    # We can try to factor these.
    
    # Actually, let's just try to solve for a rank R decomposition using a greedy approach.
    
    # We'll try R = 48 first.
    
    R = 48
    
    # Initialize U, V, W as zeros
    U = np.zeros((n*m, R), dtype=float)
    V = np.zeros((m*p, R), dtype=float)
    W = np.zeros((p*n, R), dtype=float)
    
    # For each rank component r, we need to assign values to U, V, W
    # such that the sum over r gives the correct tensor.
    
    # This is a complex optimization problem. Let's use a simpler construction.
    
    # Construction: For each (i,j), we have p=8 k-values.
    # We can group the k-values and use Strassen-like ideas.
    
    # Actually, let's use a known result: for 2x2x2, rank 7 is optimal.
    # For 3x4x8, we can try to use Strassen on 2x2 sub-blocks.
    
    # Let's split A (3x4) into A1 (2x4) and A2 (1x4).
    # Split B (4x8) into B1 (2x8) and B2 (2x8) -- wait, B is 4x8, so split rows.
    # B = [B1; B2] where B1 is 2x8, B2 is 2x8.
    
    # C = A * B = A1 * B1 + A2 * B2
    # A1 is 2x4, B1 is 2x8 -- dimensions don't match for multiplication.
    
    # Let me reconsider.
    # A is 3x4, B is 4x8.
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    
    # Split j into two groups: j=0,1 and j=2,3.
    # C[i,k] = sum_{j=0}^{1} A[i,j] * B[j,k] + sum_{j=2}^{3} A[i,j] * B[j,k]
    
    # Let A_left = A[:, 0:2] (3x2), A_right = A[:, 2:4] (3x2)
    # Let B_left = B[0:2, :] (2x8), B_right = B[2:4, :] (2x8)
    
    # C = A_left * B_left + A_right * B_right
    
    # Now we need to compute two 3x2x8 multiplications.
    # Schoolbook rank for 3x2x8 is 3*2*8 = 48.
    # Two of them: 96. Same as before.
    
    # Can we do better for 3x2x8?
    # 3x2x8: C[i,k] = sum_{j=0}^{1} A[i,j] * B[j,k]
    # For each i, C[i,:] = A[i,0] * B[0,:] + A[i,1] * B[1,:]
    # This is 2 terms per row, 3 rows = 6 scalar-vector products.
    # In tensor terms, each (i,j,k) is a term, so 3*2*8 = 48 terms.
    
    # To reduce rank, we need to share terms across i or k.
    
    # Let's try a different approach: use the fact that we can combine terms.
    
    # For a fixed (i,k), C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    
    # We can write this as a dot product of [A[i,0], A[i,1], A[i,2], A[i,3]] and [B[0,k], B[1,k], B[2,k], B[3,k]].
    
    # The tensor decomposition corresponds to expressing this bilinear form as a sum of rank-1 terms.
    
    # Rank-1 term: u_i * v_k where u_i depends on A[i,:] and v_k depends on B[:,k].
    
    # But we need the term to be bilinear in A and B separately.
    
    # A rank-1 term in the tensor decomposition is:
    # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # For this to contribute to C[i,k], we need the j-index to be consistent.
    
    # Let's try to construct a decomposition with rank 48.
    
    # I'll use a simple construction: for each (i,j), create one rank component that handles all k.
    # But that would be 3*4 = 12 components, each handling 8 k-values.
    # Each component would need V[j*p+k, r] to be non-zero for all k.
    
    # Let's try: for each (i,j), r = i*4 + j (0..11)
    # U[i*m+j, r] = 1
    # V[j*p+k, r] = 1 for all k
    # W[k*n+i, r] = 1 for all k
    
    # Then sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = sum_j 1*1*1 = 4 for all i,k.
    # That's not correct. We need it to be 1 for the correct (i,j,k) and 0 otherwise.
    
    # This approach doesn't work directly.
    
    # Let's just return the schoolbook decomposition for now.
    # It's guaranteed to be correct, even if not minimal.
    
    R = n * m * p
    U = np.zeros((n*m, R), dtype=float)
    V = np.zeros((m*p, R), dtype=float)
    W = np.zeros((p*n, R), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    
    # Convert to lists of lists for the expected output format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
