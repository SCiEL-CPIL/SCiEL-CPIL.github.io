"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

import time
import random
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p  # 32

    # Try to find a lower-rank decomposition using known results or search
    # For 2x4x4 matrix multiplication, we can try various approaches
    
    # Approach 1: Use Strassen-like decomposition for 2x2 blocks
    # The 2x2x2 matrix multiplication has rank 7 (Strassen)
    # We can decompose 2x4x4 as blocks of 2x2x2
    
    # Let's think about this differently.
    # A 2x4 matrix times a 4x4 matrix gives a 2x4 result.
    # We can view this as:
    # - Split the 4 columns of the first matrix into two 2-column blocks
    # - Split the 4 rows of the second matrix into two 2-row blocks
    # - This gives us a 2x2 grid of 2x2x2 multiplications
    
    # Actually, let's be more careful.
    # A is 2x4, B is 4x4, C = AB is 2x4.
    # Split A into A1 (2x2) and A2 (2x2) horizontally.
    # Split B into B1 (2x4) and B2 (2x4) vertically... no, that's not right.
    
    # Let me reconsider. B is 4x4. If we split by columns:
    # B = [B_left (4x2) | B_right (4x2)]
    # A = [A_left (2x2) | A_right (2x2)]
    # C = A_left * B_left + A_right * B_right
    # where B_left is 4x2 and B_right is 4x2.
    
    # So we have two products: (2x2) * (4x2) -> (2x2)
    # Each (2x2)*(4x2) can be seen as two (2x2)*(2x2) products.
    
    # Actually, let's just try a direct search or use known lower bounds.
    
    # Known result: The rank of 2x4x4 matrix multiplication is at most 28
    # (using Strassen on 2x2 blocks)
    
    # Let's construct the rank-28 decomposition.
    # We'll use the fact that 2x2x2 has rank 7.
    
    # Decomposition strategy:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # where i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}
    
    # Split j into j1 in {0,1} and j2 in {2,3}
    # C[i,k] = sum_{j1=0,1} A[i,j1] * B[j1,k] + sum_{j2=2,3} A[i,j2] * B[j2,k]
    
    # For each half, we have a (2x2)*(2x4) product.
    # Let's further split k: k in {0,1} and k in {2,3}
    
    # So we have 4 sub-problems, each being (2x2)*(2x2) -> (2x2)
    # Each (2x2)*(2x2) has rank 7 (Strassen).
    # Total rank = 4 * 7 = 28.
    
    # Strassen decomposition for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    #
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M4
    # C10 = M2 + M7
    # C11 = M1 - M3 + M6 - M5
    
    # Let's build the U, V, W matrices for rank 28.
    # We'll have 28 columns (one per Strassen bilinear form across the 4 blocks).
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 2*4 = 8 rows
    V = [[0] * rank for _ in range(m * p)]  # 4*4 = 16 rows
    W = [[0] * rank for _ in range(p * n)]  # 4*2 = 8 rows
    
    # Strassen coefficients for 2x2x2
    # Each entry is (u_coeff, v_coeff, w_coeff) for the bilinear form
    # The bilinear form is sum over i,j,k of t[i,j,k] * U[i,j,r] * V[j,k,r]
    # and W[k,n+i,r] contributes to C[i,k]
    
    # Strassen 7 forms:
    # M1: (A00+A11)(B00+B11) -> contributes to C00 with +1, C11 with +1
    # M2: (A10+A11)B00 -> contributes to C10 with +1
    # M3: A00(B10-B11) -> contributes to C01 with +1, C11 with -1
    # M4: A11(B01-B00) -> contributes to C00 with -1, C01 with +1
    # M5: (A00+A01)B11 -> contributes to C00 with +1, C11 with -1
    # M6: (A10-A00)(B00+B10) -> contributes to C11 with +1
    # M7: (A01-A11)(B01+B11) -> contributes to C00 with -1, C10 with +1
    
    # For each of the 4 blocks (block_row, block_col):
    # block_row in {0,1} corresponds to which half of j (0-1 or 2-3)
    # block_col in {0,1} corresponds to which half of k (0-1 or 2-3)
    
    # Within a block, local indices:
    # i_local in {0,1}, j_local in {0,1}, k_local in {0,1}
    # Global: i = i_local, j = block_row*2 + j_local, k = block_col*2 + k_local
    
    # Strassen forms as (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: dict from (i_local, j_local) to coeff
    # v_coeffs: dict from (j_local, k_local) to coeff
    # w_coeffs: dict from (k_local, i_local) to coeff (since W is indexed by k*n+i)
    
    strassen_forms = [
        # M1: (A00+A11)(B00+B11)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2: (A10+A11)B00
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,1): 1}),
        # M3: A00(B10-B11)
        ({(0,0): 1}, {(1,0): 1, (1,1): -1}, {(0,0): 1, (1,0): -1}),
        # M4: A11(B01-B00)
        ({(1,1): 1}, {(0,1): 1, (0,0): -1}, {(1,0): 1, (0,0): -1}),
        # M5: (A00+A01)B11
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(1,0): 1, (1,1): -1}),
        # M6: (A10-A00)(B00+B10)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}, {(1,1): 1}),
        # M7: (A01-A11)(B01+B11)
        ({(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}, {(0,0): -1, (0,1): 1}),
    ]
    
    col = 0
    for block_row in range(2):
        for block_col in range(2):
            for (u_coeffs, v_coeffs, w_coeffs) in strassen_forms:
                for (i_loc, j_loc), coeff in u_coeffs.items():
                    i_global = i_loc
                    j_global = block_row * 2 + j_loc
                    U[i_global * m + j_global][col] += coeff
                
                for (j_loc, k_loc), coeff in v_coeffs.items():
                    j_global = block_row * 2 + j_loc
                    k_global = block_col * 2 + k_loc
                    V[j_global * p + k_global][col] += coeff
                
                for (k_loc, i_loc), coeff in w_coeffs.items():
                    k_global = block_col * 2 + k_loc
                    i_global = i_loc
                    W[k_global * n + i_global][col] += coeff
                
                col += 1
    
    # Verify the decomposition
    def verify(U_mat, V_mat, W_mat, n, m, p, rank):
        # Check that for all i, j, k:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if this is a valid decomposition
        # Actually, the condition is:
        # sum_r U[i*m+j, r] * V[j*p+k, r] = delta_{i,k} * something... 
        # No, the tensor condition is:
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # where T[i,j,k] = 1 for all i,j,k (the matrix multiplication tensor)
        
        # Wait, that's not right either. Let me reconsider.
        # The matrix multiplication tensor T has T[i,j,k] = 1 for all i in [n], j in [m], k in [p]
        # A decomposition of rank r means:
        # T[i,j,k] = sum_{r=1}^r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # where the indices are: U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r)
        # And the indexing is U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
        
        # So we need: for all i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        U_arr = np.array(U_mat)
        V_arr = np.array(V_mat)
        W_arr = np.array(W_mat)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(rank):
                        val += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
                    if abs(val - 1) > 1e-9:
                        return False
        return True
    
    if verify(U, V, W, n, m, p, rank):
        return {"U": U, "V": V, "W": W}
    
    # If verification failed, fall back to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
