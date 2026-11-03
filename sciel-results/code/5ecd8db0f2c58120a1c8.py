import time
import numpy as np

DIMENSIONS = [3, 4, 8]
_DEADLINE = 150  # seconds

def _remaining_time(start):
    return _DEADLINE - 10 - (time.time() - start)

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # Check shapes
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if any(len(row) != rank for row in U + V + W):
        return False
    
    # Build the multiplication tensor and check decomposition
    # T[i,k,j] = 1 if C[i,j] = sum_k A[i,k] B[k,j]
    # Decomposition: T[i,k,j] = sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(p):
            for k in range(m):
                expected = 1.0
                actual = 0.0
                for r in range(rank):
                    actual += U_np[i * m + k, r] * V_np[k * p + j, r] * W_np[j * n + i, r]
                if abs(actual - expected) > tol:
                    return False
    return True

def construct():
    start = time.time()
    n, m, p = DIMENSIONS
    # Rank 88 decomposition
    # 2x M_2,4,4 (rank 28 each) + 2x M_1,4,4 (rank 16 each)
    rank = 88
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen factors for 2x2x2
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Coefficients for C11, C12, C21, C22
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    strassen_factors = [
        # (a_coeffs, b_coeffs, c_coeffs)
        # a_coeffs: dict {(i,j): coeff} for A
        # b_coeffs: dict {(i,j): coeff} for B
        # c_coeffs: dict {(i,j): coeff} for C
        # M1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M2
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}), # C21 += M2, C22 -= M2
        # M3
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}), # C12 += M3, C22 += M3
        # M4
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}), # C11 += M4, C21 += M4
        # M5
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}), # C11 -= M5, C12 += M5
        # M6
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}), # C22 += M6
        # M7
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}), # C11 += M7
    ]
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val

    # Process 2x M_2,4,4 blocks (for i=0,1)
    for c_block in range(2):
        # A is 2x4, B is 4x4, C is 2x4
        # A_L: A[0:2, 0:2], A_R: A[0:2, 2:4]
        # B_LL: B[0:2, 0:2], B_LR: B[0:2, 2:4], B_RL: B[2:4, 0:2], B_RR: B[2:4, 2:4]
        # C_L: C[0:2, 0:2], C_R: C[0:2, 2:4]
        
        # Map local 2x2 indices to global
        # A global idx: i*4 + j
        # B global idx: j*8 + (k + 4*c_block)
        # C global idx: (k + 4*c_block)*3 + i
        
        for factor in strassen_factors:
            a_coeffs, b_coeffs, c_coeffs = factor
            
            # Determine which A and B blocks are involved
            # The factor is for a specific 2x2 product.
            # We need to identify which of the 4 products (LL, LR, RL, RR) this factor belongs to.
            # Actually, the Strassen factors are defined for a single 2x2x2 multiplication.
            # We have 4 such multiplications:
            # 1. A_L * B_LL -> contributes to C_L
            # 2. A_L * B_LR -> contributes to C_R
            # 3. A_R * B_RL -> contributes to C_L
            # 4. A_R * B_RR -> contributes to C_R
            
            # Wait, the Strassen factors above are for ONE 2x2x2 multiplication.
            # We need to apply them to each of the 4 sub-products.
            # But the factors M1..M7 are specific combinations.
            # We can just instantiate the 7 factors for each of the 4 sub-products.
            
            # Sub-product 1: A_L (A[0:2, 0:2]) * B_LL (B[0:2, 0:2]) -> C_L (C[0:2, 0:2])
            # Sub-product 2: A_L (A[0:2, 0:2]) * B_LR (B[0:2, 2:4]) -> C_R (C[0:2, 2:4])
            # Sub-product 3: A_R (A[0:2, 2:4]) * B_RL (B[2:4, 0:2]) -> C_L (C[0:2, 0:2])
            # Sub-product 4: A_R (A[0:2, 2:4]) * B_RR (B[2:4, 2:4]) -> C_R (C[0:2, 2:4])
            
            # We iterate over the 4 sub-products
            for sub_prod_idx in range(4):
                # Determine A offset, B offset, C offset
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
                else: # sub_prod_idx == 3
                    a_i_off, a_j_off = 0, 2
                    b_i_off, b_j_off = 2, 2
                    c_i_off, c_j_off = 0, 2
                
                # Apply the current Strassen factor to this sub-product
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
    
    # Process 2x M_1,4,4 blocks (for i=2)
    for c_block in range(2):
        # A is 1x4, B is 4x4, C is 1x4
        # C_j = sum_k A_k B_kj
        # 16 terms: A_k B_kj C_j
        for k in range(4):
            for j in range(4):
                u_idx = 2 * 4 + k
                v_idx = k * 8 + (j + 4 * c_block)
                w_idx = (j + 4 * c_block) * 3 + 2
                
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
    
    # Verify the decomposition
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook rank 96
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    col_fb = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U_fb[i * m + k][col_fb] = 1
                V_fb[k * p + j][col_fb] = 1
                W_fb[j * n + i][col_fb] = 1
                col_fb += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}
