"""Program for 4x4x4 matrix multiplication tensor decomposition."""
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-49 decomposition for 4x4x4 matrix multiplication
    using the Strassen tensor square.
    
    For 2x2x2, Strassen gives rank 7. The tensor square for 4x4x4 has rank 7^2 = 49.
    We construct the 49 rank-1 tensors by taking all pairs of Strassen's 7 terms.
    
    Verification: We check that the decomposition exactly reproduces matrix
    multiplication for random test cases.
    """
    n, m, p = 4, 4, 4
    
    # Strassen for 2x2x2: 7 rank-1 terms
    # Each term has U (2x2), V (2x2), W (2x2) entries
    # Strassen's 7 terms for computing C = A*B where A,B,C are 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A01-A10)(B00+B10)
    # M7 = (A01-A11)(B11+B01)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    def strassen_2x2():
        """Return list of 7 (U, V, W) tuples, each a 2x2 matrix (list of lists)."""
        terms = []
        # M1 = (A00+A11)(B00+B11)
        U = [[1,0],[0,1]]; V = [[1,0],[0,1]]; W = [[1,0],[0,1]]
        terms.append((U,V,W))
        # M2 = (A10+A11)B00
        U = [[0,0],[1,1]]; V = [[1,0],[0,0]]; W = [[0,0],[1,0]]
        terms.append((U,V,W))
        # M3 = A00(B01-B11)
        U = [[1,0],[0,0]]; V = [[0,1],[0,-1]]; W = [[0,1],[0,0]]
        terms.append((U,V,W))
        # M4 = A11(B10-B00)
        U = [[0,0],[0,1]]; V = [[-1,0],[1,0]]; W = [[1,0],[0,1]]
        terms.append((U,V,W))
        # M5 = (A00+A10)B11
        U = [[1,0],[1,0]]; V = [[0,0],[0,1]]; W = [[-1,0],[0,-1]]
        terms.append((U,V,W))
        # M6 = (A01-A10)(B00+B10)
        U = [[0,1],[-1,0]]; V = [[1,0],[1,0]]; W = [[0,0],[0,1]]
        terms.append((U,V,W))
        # M7 = (A01-A11)(B11+B01)
        U = [[0,1],[0,-1]]; V = [[0,1],[1,1]]; W = [[1,0],[0,0]]
        terms.append((U,V,W))
        return terms
    
    s2x2 = strassen_2x2()
    
    # For 4x4x4, the indices are:
    # A is 4x4: A[i,j] where i,j in {0,1,2,3}
    # Block structure: i = 2*i_hi + i_lo, j = 2*j_hi + j_lo
    #   where i_hi, j_hi in {0,1} select the block and i_lo, j_lo in {0,1} select within block.
    #
    # The 49 terms from the tensor square:
    # For each pair (r1, r2) where r1, r2 in {0..6}:
    # U_4x4[i,j] = U_s[r1][i_hi, j_hi] * U_s[r2][i_lo, j_lo]
    # V_4x4[j,k] = V_s[r1][j_hi, k_hi] * V_s[r2][j_lo, k_lo]
    # W_4x4[k,i] = W_s[r1][k_hi, i_hi] * W_s[r2][k_lo, i_lo]
    
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for r1 in range(7):
        for r2 in range(7):
            # Extract the 2x2 matrices for this pair of Strassen terms
            U1 = s2x2[r1][0]
            V1 = s2x2[r1][1]
            W1 = s2x2[r1][2]
            U2 = s2x2[r2][0]
            V2 = s2x2[r2][1]
            W2 = s2x2[r2][2]
            
            # Fill in the 4x4 matrices
            for i_hi in range(2):
                for j_hi in range(2):
                    for i_lo in range(2):
                        for j_lo in range(2):
                            i = 2 * i_hi + i_lo
                            j = 2 * j_hi + j_lo
                            U[i * 4 + j][col] = U1[i_hi][j_hi] * U2[i_lo][j_lo]
            
            for j_hi in range(2):
                for k_hi in range(2):
                    for j_lo in range(2):
                        for k_lo in range(2):
                            j = 2 * j_hi + j_lo
                            k = 2 * k_hi + k_lo
                            V[j * 4 + k][col] = V1[j_hi][k_hi] * V2[j_lo][k_lo]
            
            for k_hi in range(2):
                for i_hi in range(2):
                    for k_lo in range(2):
                        for i_lo in range(2):
                            k = 2 * k_hi + k_lo
                            i = 2 * i_lo + i_hi  # Wait, need to be careful with index mapping
                            # W is indexed as W[k*n + i, r] where k is the row of B/C and i is the row of A/C
                            # Actually W[k*4+i, col] corresponds to C[i,k] in the output
                            # Let me re-examine: W[k*n+i, r] means for output position (i,k)
                            # In the tensor square, W_4x4[i,k] = W1[i_hi, k_hi] * W2[i_lo, k_lo]
                            # But the indexing is W[k*4+i, col], so:
                            i_idx = 2 * i_hi + i_lo
                            k_idx = 2 * k_hi + k_lo
                            W[k_idx * 4 + i_idx][col] = W1[i_hi][k_hi] * W2[i_lo][k_lo]
            
            col += 1
    
    # Verify the decomposition is correct
    # For random A, B: check that sum_r W[r] * (U[r] . A) * (V[r] . B) = A*B
    # where (U[r] . A) = sum_{i,j} U[i*4+j, r] * A[i,j]
    
    import random
    random.seed(42)
    for trial in range(5):
        A = [[random.randint(-3, 3) for _ in range(4)] for _ in range(4)]
        B = [[random.randint(-3, 3) for _ in range(4)] for _ in range(4)]
        
        # Compute true product
        AB_true = [[0] * 4 for _ in range(4)]
        for i in range(4):
            for k in range(4):
                s = 0
                for j in range(4):
                    s += A[i][j] * B[j][k]
                AB_true[i][k] = s
        
        # Compute using decomposition
        # For each r, compute u_r = sum_{i,j} U[i*4+j, r] * A[i,j]
        # and v_r = sum_{j,k} V[j*4+k, r] * B[j,k]
        # Then C_decomp[i,k] = sum_r W[k*4+i, r] * u_r * v_r
        AB_decomp = [[0] * 4 for _ in range(4)]
        for r in range(rank):
            u_r = 0
            for i in range(4):
                for j in range(4):
                    u_r += U[i * 4 + j][r] * A[i][j]
            v_r = 0
            for j in range(4):
                for k in range(4):
                    v_r += V[j * 4 + k][r] * B[j][k]
            for i in range(4):
                for k in range(4):
                    AB_decomp[i][k] += W[k * 4 + i][r] * u_r * v_r
        
        # Check equality
        for i in range(4):
            for k in range(4):
                if AB_decomp[i][k] != AB_true[i][k]:
                    raise ValueError(f"Decomposition failed at ({i},{k}): {AB_decomp[i][k]} != {AB_true[i][k]}")
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
