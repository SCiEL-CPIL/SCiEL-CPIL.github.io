"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 150  # seconds, with margin
    
    n, m, p = DIMENSIONS  # 4, 4, 4
    
    # Strassen 2x2 decomposition terms: (u_coeffs, v_coeffs, w_coeffs)
    # Positions indexed as (row, col) in {0,1}x{0,1}
    # u_coeffs: dict {(r,c): coeff} for the left 2x2 matrix
    # v_coeffs: dict {(r,c): coeff} for the right 2x2 matrix  
    # w_coeffs: dict {(r,c): coeff} for the output 2x2 matrix
    strassen_2x2 = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
    ]
    
    # Top-level Strassen for 4x4: 7 products, each is a 2x2 matrix product
    # Each product: (X_block_coeffs, Y_block_coeffs, C_block_coeffs)
    # Block indices: (0,0), (0,1), (1,0), (1,1)
    top_level = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),      # M1
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),                # M2
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),                # M3
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),               # M4
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),               # M5
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),               # M6
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):1}),      # M7
    ]
    
    rank = 49
    U = [[0] * rank for _ in range(n * m)]  # 16 x 49
    V = [[0] * rank for _ in range(m * p)]  # 16 x 49
    W = [[0] * rank for _ in range(p * n)]  # 16 x 49
    
    col = 0
    for ti, (x_blocks, y_blocks, c_blocks) in enumerate(top_level):
        for u_coeffs, v_coeffs, w_coeffs in strassen_2x2:
            # Build u: linear form in A entries
            # For each block (bi) in x_blocks with coeff cx,
            # for each position (pi) in u_coeffs with coeff cu:
            # A entry at row = bi[0]*2 + pi[0], col = bi[1]*2 + pi[1]
            u_dict = {}
            for (bi_r, bi_c), cx in x_blocks.items():
                for (pi_r, pi_c), cu in u_coeffs.items():
                    a_row = bi_r * 2 + pi_r
                    a_col = bi_c * 2 + pi_c
                    a_idx = a_row * m + a_col  # i*m + j
                    coeff = cx * cu
                    u_dict[a_idx] = u_dict.get(a_idx, 0) + coeff
            
            # Build v: linear form in B entries
            v_dict = {}
            for (bj_r, bj_c), cy in y_blocks.items():
                for (pj_r, pj_c), cv in v_coeffs.items():
                    b_row = bj_r * 2 + pj_r
                    b_col = bj_c * 2 + pj_c
                    b_idx = b_row * p + b_col  # j*p + k
                    coeff = cy * cv
                    v_dict[b_idx] = v_dict.get(b_idx, 0) + coeff
            
            # Build w: linear form in C entries (stored as W[k*n + i])
            w_dict = {}
            for (bc_r, bc_c), cz in c_blocks.items():
                for (pc_r, pc_c), cw in w_coeffs.items():
                    c_row = bc_r * 2 + pc_r  # i (output row)
                    c_col = bc_c * 2 + pc_c  # k (output col)
                    w_idx = c_col * n + c_row  # k*n + i
                    coeff = cz * cw
                    w_dict[w_idx] = w_dict.get(w_idx, 0) + coeff
            
            # Fill in U, V, W for this term
            for idx, coeff in u_dict.items():
                if coeff != 0:
                    U[idx][col] = coeff
            for idx, coeff in v_dict.items():
                if coeff != 0:
                    V[idx][col] = coeff
            for idx, coeff in w_dict.items():
                if coeff != 0:
                    W[idx][col] = coeff
            
            col += 1
    
    # Verify: check that the decomposition is correct
    if time.time() - start_time < DEADLINE - 5:
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should give (A*B)[i,k]
        # Test with random matrices
        rng = np.random.RandomState(42)
        A = rng.randint(-3, 4, size=(n, m)).astype(float)
        B = rng.randint(-3, 4, size=(m, p)).astype(float)
        C_expected = A @ B
        
        # Compute using tensor decomposition
        C_computed = np.zeros((n, p))
        for c in range(rank):
            u_vec = U_np[:, c]  # length n*m
            v_vec = V_np[:, c]  # length m*p
            w_vec = W_np[:, c]  # length p*n
            
            # u_vec[i*m+j] * v_vec[j*p+k] * w_vec[k*n+i]
            # Reshape u to (n,m), v to (m,p), w to (p,n)
            u_mat = u_vec.reshape(n, m)
            v_mat = v_vec.reshape(m, p)
            w_mat = w_vec.reshape(p, n)
            
            # The product u_mat[i,j] * v_mat[j,k] * w_mat[k,i] summed over j,k
            # = sum_j,k u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
            # This is like a tensor contraction
            # = (u_mat * v_mat) contracted with w_mat
            # = sum_j (u_mat[i,j] * sum_k v_mat[j,k] * w_mat[k,i])
            # = sum_j u_mat[i,j] * (v_mat @ w_mat)[j,i]
            # = (u_mat @ (v_mat @ w_mat.T)).T ... let me just do it directly
            
            # Actually: sum_{j,k} u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
            # Let temp[j,i] = sum_k v_mat[j,k] * w_mat[k,i] = (v_mat @ w_mat)[j,i]... 
            # Wait, w_mat is (p,n) = (k,i), so w_mat[k,i]. v_mat[j,k]. 
            # sum_k v_mat[j,k] * w_mat[k,i] = (v_mat @ w_mat)[j, i] where w_mat is (p,n)=(k,i)
            # So temp = v_mat @ w_mat, shape (m, n)
            # Then sum_j u_mat[i,j] * temp[j,i] = (u_mat @ temp)[i,i]? No...
            # sum_j u_mat[i,j] * temp[j,i] = (u_mat @ temp.T)[i,i]? 
            # u_mat is (n,m), temp is (m,n). u_mat @ temp.T would be (n,n). 
            # (u_mat @ temp.T)[i,i] = sum_j u_mat[i,j] * temp.T[j,i] = sum_j u_mat[i,j] * temp[i,j]... no
            # temp.T is (n,m), temp.T[j,i] doesn't make sense if j is row of temp.T...
            # Let me just do it element-wise for small matrices
            for i in range(n):
                for k in range(p):
                    s = 0.0
                    for j in range(m):
                        for kk in range(p):
                            s += u_mat[i, j] * v_mat[j, kk] * w_mat[kk, i]
                    C_computed[i, k] += s
        
        max_err = np.max(np.abs(C_computed - C_expected))
        if max_err > 1e-10:
            # Fallback to schoolbook if Strassen is wrong
            rank_sb = n * m * p
            U_sb = [[0] * rank_sb for _ in range(n * m)]
            V_sb = [[0] * rank_sb for _ in range(m * p)]
            W_sb = [[0] * rank_sb for _ in range(p * n)]
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_sb[i * m + j][column] = 1
                        V_sb[j * p + k][column] = 1
                        W_sb[k * n + i][column] = 1
                        column += 1
            return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
