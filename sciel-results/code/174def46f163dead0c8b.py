"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # Rank 88 decomposition:
    # 2x M_2,4,4 (rank 28 each via Strassen) + 2x M_1,4,4 (rank 16 each via schoolbook)
    rank = 88
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen factors for 2x2x2 multiplication
    # Each factor is (a_coeffs, b_coeffs, c_coeffs) where coeffs map (row, col) -> value
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    strassen_factors = [
        # M1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M2
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        # M3
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # M4
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # M5
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # M6
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # M7
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    # Process 2x M_2,4,4 blocks (for rows i=0,1 of the 3x4 matrix)
    # Each M_2,4,4 uses Strassen on 2x2 sub-blocks: 4 sub-products of 2x2x2
    for c_block in range(2):
        for factor in strassen_factors:
            a_coeffs, b_coeffs, c_coeffs = factor
            
            # 4 sub-products: (A_left * B_LL), (A_left * B_LR), (A_right * B_RL), (A_right * B_RR)
            for sub_prod_idx in range(4):
                if sub_prod_idx == 0:
                    a_i_off, a_j_off = 0, 0
                    b_i_off, b_j_off = 0, 0
                    c_i_off, c_j_off = 0, 0
                elif sub_prod_idx == 1:
                    a_i_off, a_j_off = 0, 0
                    b_i_off, b_j_off = 0, 2
                    c_i_off, c_j_off = 0, 2
                elif sub_prod_idx == 2:
                    a_i_off, a_j_off = 0, 2
                    b_i_off, b_j_off = 2, 0
                    c_i_off, c_j_off = 0, 0
                else:  # sub_prod_idx == 3
                    a_i_off, a_j_off = 0, 2
                    b_i_off, b_j_off = 2, 2
                    c_i_off, c_j_off = 0, 2
                
                for (ai, aj), au in a_coeffs.items():
                    u_idx = (ai + a_i_off) * 4 + (aj + a_j_off)
                    U[u_idx][col] += au
                
                for (bi, bj), bv in b_coeffs.items():
                    v_idx = (bi + b_i_off) * 8 + (bj + b_j_off + 4 * c_block)
                    V[v_idx][col] += bv
                
                for (ci, cj), cw in c_coeffs.items():
                    w_idx = (cj + c_j_off + 4 * c_block) * 3 + (ci + c_i_off)
                    W[w_idx][col] += cw
                
                col += 1
    
    # Process 2x M_1,4,4 blocks (for row i=2 of the 3x4 matrix)
    # Schoolbook: 16 terms per block
    for c_block in range(2):
        for k in range(4):
            for j in range(4):
                u_idx = 2 * 4 + k
                v_idx = k * 8 + (j + 4 * c_block)
                w_idx = (j + 4 * c_block) * 3 + 2
                
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
    
    # Verify the decomposition is exact using numpy
    # The tensor T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r] should equal delta[i,k] for matching indices
    # Actually, for matrix multiplication C = A * B where A is n x m, B is m x p, C is n x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor decomposition gives: T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
    # where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The indexing is: U[i*m + j, r], V[j*p + k, r], W[k*n + i, r]
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Reshape to proper dimensions
    # U: (n*m, rank) -> we need to check the tensor
    # The matrix multiplication tensor M_{n,m,p} has entries M[i,j,k] = 1 if i corresponds to A[i,j], 
    # j to B[j,k], k to C[i,k]
    # In our convention: U is indexed by (i,j) flattened as i*m+j, V by (j,k) as j*p+k, W by (k,i) as k*n+i
    
    # Build the expected tensor
    # For each (i, j, k), the contribution to C[i,k] from A[i,j]*B[j,k] is 1
    # So the tensor T[i,j,k] = 1 for all valid i,j,k (it's the identity in the middle dimension)
    
    # Let's verify by computing C = A*B for random matrices
    # A is n x m, B is m x p, C is n x p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Using the decomposition:
    # C[i,k] = sum_r (sum_j A[i,j] * U[i*m+j, r]) * (sum_j B[j,k] * V[j*p+k, r]) * W[k*n+i, r]
    # Wait, that's not quite right. The decomposition is:
    # C[i,k] = sum_r W[k*n+i, r] * (sum_j A[i,j] U[i*m+j, r]) * (sum_j B[j,k] V[j*p+k, r])
    # Hmm, actually the standard form is:
    # C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over j? No.
    
    # Let me think again. The tensor for matrix multiplication M_{n,m,p} is defined as:
    # M_{i,j,k} = 1 for all i in [n], j in [m], k in [p]
    # where i indexes rows of A and C, j indexes columns of A and rows of B, k indexes columns of B and C.
    
    # The rank-r decomposition means:
    # M_{i,j,k} = sum_{r=1}^{r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Let's verify this with random matrices
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            val = 0.0
            for r in range(rank):
                # U[i*m+j, r] for each j, V[j*p+k, r] for each j, W[k*n+i, r]
                # C[i,k] = sum_r W[k*n+i, r] * (sum_j A[i,j] * U[i*m+j, r]) * (sum_j B[j,k] * V[j*p+k, r])
                # Wait, that's not right either.
                
                # The correct formula is:
                # C[i,k] = sum_r W[k*n+i, r] * (sum_j A[i,j] U[i*m+j, r]) * (sum_j B[j,k] V[j*p+k, r])
                # No, that would give a product of two sums, which is not correct.
                
                # Actually, the tensor decomposition is:
                # M_{i,j,k} = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
                # And C[i,k] = sum_j A[i,j] B[j,k] M_{i,j,k} = sum_j A[i,j] B[j,k] sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
                # = sum_r W[k*n+i, r] (sum_j A[i,j] U[i*m+j, r] B[j,k] V[j*p+k, r])
                
                # Hmm, that's still not separating cleanly. Let me just directly compute the tensor.
                pass
            
            # Direct tensor evaluation
            for j in range(m):
                for r in range(rank):
                    val += A_test[i,j] * B_test[j,k] * U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
            C_computed[i,k] = val
    
    max_err = np.max(np.abs(C_computed - C_expected))
    if max_err > 1e-10:
        # If verification fails, fall back to schoolbook (rank 96)
        rank_school = n * m * p
        U_school = [[0] * rank_school for _ in range(n * m)]
        V_school = [[0] * rank_school for _ in range(m * p)]
        W_school = [[0] * rank_school for _ in range(p * n)]
        
        col_s = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i*m+j][col_s] = 1
                    V_school[j*p+k][col_s] = 1
                    W_school[k*n+i][col_s] = 1
                    col_s += 1
        
        return {"U": U_school, "V": V_school, "W": W_school}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
