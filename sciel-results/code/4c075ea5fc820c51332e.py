import time
import random
import numpy as np

DIMENSIONS = [2, 4, 8]
START_TIME = time.time()
TIME_LIMIT = 140  # Leave 10s margin before 150s kill

# EVOLVE-BLOCK-START
def construct():
    if time.time() - START_TIME > TIME_LIMIT:
        # Return schoolbook as fallback if we're running out of time
        n, m, p = DIMENSIONS
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    n, m, p = DIMENSIONS
    # Target rank 56 using Strassen for 2x2 blocks
    # Structure: 4 column blocks in B, each requiring rank 14 (2x Strassen)
    target_rank = 56
    
    U = [[0] * target_rank for _ in range(n * m)]
    V = [[0] * target_rank for _ in range(m * p)]
    W = [[0] * target_rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # M1 = (A00 - A11)(B11 - B00)
    # M2 = (A00 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 + A11)(B10 - B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4  <- CHECK THIS
    # C11 = M1 - M2 + M3 + M6
    
    # Let's define the rank-1 terms (u, v, w) for the 7 M's
    # u: indices (i, j) in A
    # v: indices (j, k) in B
    # w: indices (k, i) in C (stored as k*n+i)
    
    def get_strassen_terms():
        terms = []
        # M1: (A00 - A11)(B11 - B00)
        # This is a difference of products, so it's not a single rank-1 term in the standard sense?
        # No, Strassen defines 7 products. Each product is a scalar multiplication of matrices.
        # M1 = (A00 - A11) * (B11 - B00).
        # This is one "multiplication" step.
        # In tensor decomposition, a "multiplication" $XY$ corresponds to a rank-1 tensor if X, Y are scalars.
        # Here X, Y are matrices? No, in the 2x2x2 tensor, the inputs are scalars $A_{ij}, B_{jk}$.
        # The formula $M_1 = (A_{00} - A_{11})(B_{11} - B_{00})$ involves 2 multiplications of scalars?
        # No, it is a bilinear form.
        # $M_1$ is a single scalar value computed from 4 inputs.
        # It corresponds to 1 rank-1 term in the decomposition of the tensor?
        # Yes, the rank of the tensor is the number of such scalar multiplications.
        
        # The rank-1 term for $M_1$ is:
        # u: vector with 1 at (0,0), -1 at (1,1)
        # v: vector with 1 at (1,1), -1 at (0,0)
        # w: vector with 1 at all C positions where M1 contributes?
        # No, M1 is a scalar intermediate. It is added to C with coefficients.
        # So the rank-1 term is $u \otimes v \otimes w_{M1}$?
        # No, the decomposition is $\sum_{r=1}^7 u_r \otimes v_r \otimes w_r$.
        # Here $u_r, v_r$ are the vectors for the inputs to the scalar multiplication.
        # And $w_r$ is the vector that distributes this scalar to the output C.
        
        # So for M1:
        # u1 = [1, 0, 0, -1] (indices 00, 01, 10, 11)
        # v1 = [0, 0, -1, 1] (indices 00, 01, 10, 11) -> B00=0, B01=0, B10=-1, B11=1. (B11-B00)
        # w1: M1 contributes to C00 (+1), C11 (+1).
        # w1 = [1, 0, 0, 1] (C00, C01, C10, C11)
        
        # Let's list all 7.
        
        # M1: (A00+A11)(B00+B11)
        u1 = [1, 0, 0, 1]
        v1 = [1, 0, 0, 1]
        w1 = [1, 0, 0, 1] # C00 += M1, C11 += M1
        
        # M2: (A10+A11)B00
        u2 = [0, 0, 1, 1]
        v2 = [1, 0, 0, 0]
        w2 = [0, 0, -1, -1] # C10 -= M2, C11 -= M2
        
        # M3: A00(B10-B11)
        u3 = [1, 0, 0, 0]
        v3 = [0, 0, 1, -1]
        w3 = [0, 1, 0, 1] # C01 += M3, C11 += M3
        
        # M4: A11(B01-B00)
        u4 = [0, 0, 0, 1]
        v4 = [0, 1, -1, 0]
        w4 = [1, 0, 1, 0] # C00 += M4, C10 += M4
        
        # M5: (A00+A01)B11
        u5 = [1, 1, 0, 0]
        v5 = [0, 0, 0, 1]
        w5 = [-1, 1, 0, 0] # C00 -= M5, C01 += M5
        
        # M6: (A10-A00)(B00+B01)
        u6 = [-1, 0, 1, 0]
        v6 = [1, 1, 0, 0]
        w6 = [0, 0, 0, 1] # C11 += M6
        
        # M7: (A01-A11)(B10+B11)
        u7 = [0, 1, 0, -1]
        v7 = [0, 0, 1, 1]
        w7 = [1, 0, 0, -1] # C00 += M7, C11 -= M7
        
        return [
            (u1, v1, w1),
            (u2, v2, w2),
            (u3, v3, w3),
            (u4, v4, w4),
            (u5, v5, w5),
            (u6, v6, w6),
            (u7, v7, w7)
        ]
    
    terms = get_strassen_terms()
    
    # Verify Strassen locally
    def verify_strassen():
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8]])
        C_ref = A @ B
        C_calc = np.zeros((2, 2))
        for u, v, w in terms:
            u_a = np.array(u).reshape(2, 2)
            v_b = np.array(v).reshape(2, 2)
            w_c = np.array(w).reshape(2, 2)
            # M = sum(u_a * A * v_b * B)? No.
            # M = (u_a * A) ... no.
            # The term is scalar = (u dot A_flat) * (v dot B_flat).
            # Then add scalar * w to C.
            scalar_u = np.sum(u_a * A)
            scalar_v = np.sum(v_b * B)
            C_calc += scalar_u * scalar_v * w_c
        return np.allclose(C_calc, C_ref)
    
    if not verify_strassen():
        # Fallback to schoolbook rank 64
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
        
    # Proceed with rank 56
    current_col = 0
    for b in range(4):
        # Left part: A_L (cols 0,1) * B_b_T (rows 0,1, cols 2b, 2b+1)
        # Map local indices to global
        for u, v, w in terms:
            # u: 2x2 local A_L -> Global A cols 0,1
            for i in range(2):
                for j in range(2):
                    if u[i*2+j] != 0:
                        U[i * m + j][current_col] += u[i*2+j]
                        
            # v: 2x2 local B_b_T -> Global B rows 0,1, cols 2b, 2b+1
            for j in range(2):
                for k in range(2):
                    if v[j*2+k] != 0:
                        V[j * p + (2*b + k)][current_col] += v[j*2+k]
                        
            # w: 2x2 local C_b -> Global C cols 2b, 2b+1
            for i in range(2):
                for k in range(2):
                    if w[i*2+k] != 0:
                        W[(2*b + k) * n + i][current_col] += w[i*2+k]
                        
            current_col += 1
            
        # Right part: A_R (cols 2,3) * B_b_B (rows 2,3, cols 2b, 2b+1)
        for u, v, w in terms:
            # u: 2x2 local A_R -> Global A cols 2,3
            for i in range(2):
                for j in range(2):
                    if u[i*2+j] != 0:
                        U[i * m + (2 + j)][current_col] += u[i*2+j]
                        
            # v: 2x2 local B_b_B -> Global B rows 2,3, cols 2b, 2b+1
            for j in range(2):
                for k in range(2):
                    if v[j*2+k] != 0:
                        V[(2 + j) * p + (2*b + k)][current_col] += v[j*2+k]
                        
            # w: 2x2 local C_b -> Global C cols 2b, 2b+1
            for i in range(2):
                for k in range(2):
                    if w[i*2+k] != 0:
                        W[(2*b + k) * n + i][current_col] += w[i*2+k]
                        
            current_col += 1
            
    # Final Verification
    def verify_full():
        A = np.random.randint(-10, 10, (n, m))
        B = np.random.randint(-10, 10, (m, p))
        C_ref = A @ B
        
        U_mat = np.array(U).T # rank x n*m
        V_mat = np.array(V).T # rank x m*p
        W_mat = np.array(W).T # rank x p*n
        
        # C[i, k] = sum_{j, r} A[i, j] V_mat[r, j*p+k] W_mat[r, k*n+i]
        # Efficiently:
        # Temp = U_mat @ A_flat? No.
        # C = sum_r (u_r * A_flat) (v_r * B_flat) w_r? No.
        
        # C[i, k] = sum_r (sum_j A[i, j] U_mat[r, i*m+j]) (sum_{j',k'} ... )
        # Let's do it directly for small sizes.
        C_calc = np.zeros((n, p))
        A_flat = A.flatten()
        B_flat = B.flatten()
        
        # U_mat is rank x (n*m)
        # V_mat is rank x (m*p)
        # W_mat is rank x (p*n)
        
        # proj_U = U_mat @ A_flat  (rank vector)
        # proj_V = V_mat @ B_flat  (rank vector)
        # proj_W = W_mat @ C_flat? No, we compute C.
        
        # C_flat = sum_r (proj_U[r] * proj_V[r]) * W_mat[r, :]
        
        proj_U = U_mat @ A_flat
        proj_V = V_mat @ B_flat
        coeff = proj_U * proj_V
        C_flat = coeff @ W_mat
        
        return np.allclose(C_flat.reshape(n, p), C_ref)
        
    if not verify_full():
        # Fallback to schoolbook rank 64
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
