import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=6
    
    # We'll use Strassen's algorithm on 2x2x2 sub-blocks.
    # For each j in range(m), we have A[:, j] (2-vector) and B[j, :] (6-vector).
    # Group the 6 k-values into 3 pairs: (0,1), (2,3), (4,5).
    # For each (j, k_pair), we have a 2x1x2 problem: A[:,j] * B[j, k_pair].
    # This is actually just an outer product of a 2-vector and a 2-vector = rank 4 trivially.
    # But we can use Strassen if we had 2x2x2. We only have 2x1x2 per (j,k_pair).
    
    # Alternative: View the whole thing as 3 separate 2x3x2 tensors (one for each k-pair).
    # Each 2x3x2 tensor: C[i, k] = sum_j A[i,j] B[j,k] for k in a pair.
    # For a 2x3x2 tensor, we can use rank 10 (known result) or trivial 12.
    # Trivial for 2x3x2 is 12, so 3*12=36.
    
    # Let me try a direct Strassen-like approach on the full problem.
    # The tensor T[i,j,k] = e_i e_j^T e_k^T for the multiplication.
    # We want to write T = sum_r U_r V_r W_r where U_r in R^{nm}, V_r in R^{mp}, W_r in R^{pn}.
    
    # Use the following decomposition based on Strassen for the 2x2 part in i and k:
    # For each j, define M_1 = (A[0,j]+A[1,j])(B[j,0]+B[j,1]) etc.
    
    # Let's build rank 33: 3 j-values, each with 3 k-pairs, each k-pair uses Strassen 7-term
    # but that gives 63. Too much.
    
    # Better: For each k-pair (3 pairs), we have a 2x3x2 subproblem.
    # Known: 2x3x2 has rank 10 (I believe). 3*10 = 30.
    # Let me implement rank 10 for 2x3x2.
    
    # 2x3x2: C[i,k] = sum_{j=0}^{2} A[i,j] B[j,k], i in {0,1}, k in {0,1}
    # Trivial rank = 12. Can we do better?
    # Use Strassen on the 2x2 part (i, k) by treating j as the sum dimension.
    # M1 = A[0,0]B[0,0], M2 = A[0,1]B[1,0], M3 = A[1,0]B[0,1], M4 = A[1,1]B[1,1]
    # M5 = (A[0,0]+A[0,1])(B[1,0]+B[1,1]) - M2 - M4  ... this is for 2x2x2.
    
    # For 2x3x2, I'll use a known decomposition of rank 10.
    # Terms for one 2x3x2 block:
    def strassen_2x3x2(A_block, B_block):
        """A_block is 2x3, B_block is 3x2. Returns 2x2."""
        # Use 10 multiplications
        # This is a standard result; let me just use rank 12 for now and optimize later.
        pass
    
    # Let me just use the trivial rank 36 first, then try to reduce.
    # Actually, let me implement a simple Strassen on the (i,k) 2x2 structure.
    
    rank = 36
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
