"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill

    # Rank 88 decomposition:
    # 2x M_2,4,4 (rank 28 each via Strassen) + 2x M_1,4,4 (rank 16 each via schoolbook)
    rank = 88
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Strassen factors for 2x2x2 multiplication
    strassen_factors = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    # Process 2x M_2,4,4 blocks (for rows i=0,1 of the 3x4 matrix)
    for c_block in range(2):
        for factor in strassen_factors:
            a_coeffs, b_coeffs, c_coeffs = factor
            
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
                else:
                    a_i_off, a_j_off = 0, 2
                    b_i_off, b_j_off = 2, 2
                    c_i_off, c_j_off = 0, 2
                
                for (ai, aj), au in a_coeffs.items():
                    u_idx = (ai + a_i_off) * 4 + (aj + a_j_off)
                    U[u_idx, col] += au
                
                for (bi, bj), bv in b_coeffs.items():
                    v_idx = (bi + b_i_off) * 8 + (bj + b_j_off + 4 * c_block)
                    V[v_idx, col] += bv
                
                for (ci, cj), cw in c_coeffs.items():
                    w_idx = (cj + c_j_off + 4 * c_block) * 3 + (ci + c_i_off)
                    W[w_idx, col] += cw
                
                col += 1
    
    # Process 2x M_1,4,4 blocks (for row i=2 of the 3x4 matrix)
    for c_block in range(2):
        for k in range(4):
            for j in range(4):
                u_idx = 2 * 4 + k
                v_idx = k * 8 + (j + 4 * c_block)
                w_idx = (j + 4 * c_block) * 3 + 2
                
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
    
    # Verify with vectorized numpy
    if time.time() < deadline - 5:
        np.random.seed(42)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        
        # Vectorized verification:
        # C[i,k] = sum_j A[i,j] * B[j,k] * sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # = sum_r W[k*n+i, r] * (sum_j A[i,j]*U[i*m+j, r]) * (sum_j B[j,k]*V[j*p+k, r])
        # Wait, that's not right. The tensor is T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # And C[i,k] = sum_j A[i,j] * B[j,k] * T[i,j,k]
        
        # Let's compute T directly:
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        C_computed = np.zeros((n, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # T[i,j,k]
                    t_val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    C_computed[i, k] += A_test[i, j] * B_test[j, k] * t_val
        
        max_err = np.max(np.abs(C_computed - C_expected))
        if max_err > 1e-10:
            # Fallback to schoolbook
            rank_school = n * m * p
            U_school = np.zeros((n * m, rank_school))
            V_school = np.zeros((m * p, rank_school))
            W_school = np.zeros((p * n, rank_school))
            
            col_s = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_school[i*m+j, col_s] = 1
                        V_school[j*p+k, col_s] = 1
                        W_school[k*n+i, col_s] = 1
                        col_s += 1
            
            return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
