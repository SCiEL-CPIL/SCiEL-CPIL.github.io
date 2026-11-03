"""
Matrix multiplication 2x4 by 4x4, exact tensor decomposition.
Uses Strassen's algorithm on four 2x2 sub-blocks to achieve rank 28 (vs schoolbook 32).
"""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]
TIME_LIMIT = 150.0
START_TIME = time.time()


def check_time_and_return(best_result):
    if best_result is not None and (time.time() - START_TIME) > TIME_LIMIT - 10.0:
        return best_result
    return None


def construct():
    # Directly return the Strassen 28 decomposition
    # This is the known best rank for 2x4x4
    return _construct_strassen_28()


def _verify_decomposition(result, trials=5):
    """Verify that U @ diag(V) @ W^T reconstructs the matmul tensor."""
    import random
    n, m, p = DIMENSIONS
    U = np.array(result['U'], dtype=float)
    V = np.array(result['V'], dtype=float)
    W = np.array(result['W'], dtype=float)
    rank = U.shape[1]
    for _ in range(trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        # Reconstruct C from decomposition
        # C[i,k] = sum_r U[i*m+j... wait, need to be careful
        # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
        # C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_l V[j*p+l, r] * B[j,l]) ... no
        # Actually: C[i,k] = sum_r U_col_r(A) * V_col_r(B) * W_col_r[C_index]
        # where U_col_r(A) = sum_{i,j} U[i*m+j, r] * A[i,j]
        #       V_col_r(B) = sum_{j,k} V[j*p+k, r] * B[j,k]
        #       and W contributes: C[i,k] = sum_r W[k*n+i, r] * U_col_r(A) * V_col_r(B)
        # Wait, that's not right either. The tensor decomposition is:
        # T_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # So C[i,k] = sum_j A[i,j] * B[j,k] = sum_r (sum_{i,j} U[i*m+j,r]*A[i,j]) * ... 
        # No. The correct reconstruction:
        # C[i,k] = sum_{j} A[i,j] * B[j,k]
        # = sum_r (sum_{i',j} U[i'*m+j, r] * A[i',j]) * (sum_{j',k'} V[j'*p+k', r] * B[j',k']) * W[k*n+i, r]
        # That's not right. Let me think again.
        # The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # The matmul: C[i,k] = sum_j A[i,j] * B[j,k]
        # We need: sum_j A[i,j]*B[j,k] = sum_r (sum_{i',j'} U[i'*m+j',r]*A[i',j']) * ... 
        # Actually no. The way it works:
        # For each r, compute:
        #   u_r = sum_{i,j} U[i*m+j, r] * A[i,j]  (scalar)
        #   v_r = sum_{j,k} V[j*p+k, r] * B[j,k]  (scalar)
        #   Then C[i,k] += W[k*n+i, r] * u_r * v_r
        # Let's verify: C[i,k] = sum_r W[k*n+i,r] * (sum_{i',j} U[i'*m+j,r]*A[i',j]) * (sum_{j',k'} V[j'*p+k',r]*B[j',k'])
        # This should equal sum_j A[i,j]*B[j,k] if the decomposition is correct.
        C_computed = np.zeros((n, p))
        for r in range(rank):
            u_r = 0.0
            for i in range(n):
                for j in range(m):
                    u_r += U[i * m + j, r] * A[i, j]
            v_r = 0.0
            for j in range(m):
                for k in range(p):
                    v_r += V[j * p + k, r] * B[j, k]
            for i in range(n):
                for k in range(p):
                    C_computed[i, k] += W[k * n + i, r] * u_r * v_r
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            return False
    return True


def _construct_schoolbook_32():
    n, m, p = DIMENSIONS
    rank = n * m * p  # 32
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


def _construct_strassen_28():
    """
    2x4 * 4x4 using four 2x2 Strassen blocks.
    
    Partition:
      A is 2x4: A = [A1 | A2] where A1, A2 are 2x2
      B is 4x4: B = [B11 B12; B21 B22] all 2x2
      C = A * B is 2x4: C = [C1 | C2] where
        C1 = A1*B11 + A2*B21
        C2 = A1*B12 + A2*B22
    
    For C1 = A1*B11 + A2*B21:
      We compute Strassen for (A1, B11) and (A2, B21) separately.
      Each contributes 7 rank-1 terms. Total 14 for C1.
    Similarly for C2: 14 terms.
    Total rank 28.
    """
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28
    
    # Indices in U: row = i*m + j = i*4 + j, where i in {0,1}, j in {0,1,2,3}
    # Indices in V: row = j*p + k = j*4 + k, where j in {0..3}, k in {0..3}
    # Indices in W: row = k*n + i = k*2 + i, where k in {0..3}, i in {0,1}
    
    # Strassen coefficients for 2x2 product X*Y:
    # P1 = (x00+x11)(y00+y11)
    # P2 = (x10+x11)y00
    # P3 = x00(y01-y11)
    # P4 = x11(y10-y00)
    # P5 = (x00+x01)y11
    # P6 = (x10-x00)(y00+y01)
    # P7 = (x01-x11)(y10+y11)
    
    # For each Strassen term t in 1..7, we define:
    #   U_coeffs: dict mapping local A index (0..3) to coefficient
    #   V_coeffs: dict mapping local B index (0..3) to coefficient
    #   W_coeffs: dict mapping local C index (0..3) to coefficient
    
    # Local indices for 2x2 matrix [a b; c d]: a=0, b=1, c=2, d=3
    
    # Strassen term definitions (U, V, W coefficients for local indices)
    # Note: W coefficients are how each P contributes to c00, c01, c10, c11
    # c00 = P1 + P4 - P5 + P7
    # c01 = P3 + P5
    # c10 = P2 + P4
    # c11 = P1 - P2 + P3 + P6
    
    strassen_terms = [
        # term, U_coeffs, V_coeffs, W_coeffs
        # P1: (x00+x11)(y00+y11) -> c00:+1, c11:+1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # P2: (x10+x11)y00 -> c10:+1, c11:-1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # P3: x00(y01-y11) -> c01:+1, c11:+1
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # P4: x11(y10-y00) -> c00:+1, c10:+1
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # P5: (x00+x01)y11 -> c00:-1, c01:+1
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # P6: (x10-x00)(y00+y01) -> c11:+1
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # P7: (x01-x11)(y10+y11) -> c00:+1
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    # We have 4 sub-products to compute:
    # 1. A_left * B_tl -> contributes to C_left (cols 0,1)
    # 2. A_right * B_bl -> contributes to C_left (cols 0,1)
    # 3. A_left * B_tr -> contributes to C_right (cols 2,3)
    # 4. A_right * B_br -> contributes to C_right (cols 2,3)
    
    # For each sub-product, we apply Strassen.
    # Sub-product 1: A_left (A[:,0:2]) * B_tl (B[0:2,0:2]) -> C_left (C[:,0:2])
    #   A_left local (r,c) -> global U[(r)*4 + (c)]
    #   B_tl local (r,c) -> global V[(r)*4 + (c)]
    #   C_left local (r,c) -> global W[(c)*2 + (r)]
    
    # Sub-product 2: A_right (A[:,2:4]) * B_bl (B[2:4,0:2]) -> C_left (C[:,0:2])
    #   A_right local (r,c) -> global U[(r)*4 + (c+2)]
    #   B_bl local (r,c) -> global V[(r+2)*4 + (c)]
    #   C_left local (r,c) -> global W[(c)*2 + (r)]
    
    # Sub-product 3: A_left (A[:,0:2]) * B_tr (B[0:2,2:4]) -> C_right (C[:,2:4])
    #   A_left local (r,c) -> global U[(r)*4 + (c)]
    #   B_tr local (r,c) -> global V[(r)*4 + (c+2)]
    #   C_right local (r,c) -> global W[(c+2)*2 + (r)]
    
    # Sub-product 4: A_right (A[:,2:4]) * B_br (B[2:4,2:4]) -> C_right (C[:,2:4])
    #   A_right local (r,c) -> global U[(r)*4 + (c+2)]
    #   B_br local (r,c) -> global V[(r+2)*4 + (c+2)]
    #   C_right local (r,c) -> global W[(c+2)*2 + (r)]
    
    sub_products = [
        # (a_offset, b_offset, c_offset) where offsets are (row_off, col_off) for A, B, C
        # A: (row_off, col_off), B: (row_off, col_off), C: (col_off, row_off) -- careful with W indexing
        # Let's define mappings explicitly for clarity
        
        # Sub-product 1
        {
            'a_map': lambda r, c: r * 4 + c,
            'b_map': lambda r, c: r * 4 + c,
            'c_map': lambda r, c: c * 2 + r,
        },
        # Sub-product 2
        {
            'a_map': lambda r, c: r * 4 + (c + 2),
            'b_map': lambda r, c: (r + 2) * 4 + c,
            'c_map': lambda r, c: c * 2 + r,
        },
        # Sub-product 3
        {
            'a_map': lambda r, c: r * 4 + c,
            'b_map': lambda r, c: r * 4 + (c + 2),
            'c_map': lambda r, c: (c + 2) * 2 + r,
        },
        # Sub-product 4
        {
            'a_map': lambda r, c: r * 4 + (c + 2),
            'b_map': lambda r, c: (r + 2) * 4 + (c + 2),
            'c_map': lambda r, c: (c + 2) * 2 + r,
        },
    ]
    
    col = 0
    for sp in sub_products:
        for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
            for loc_u, coeff in u_coeffs.items():
                r, c = loc_u // 2, loc_u % 2
                global_u = sp['a_map'](r, c)
                U[global_u][col] += coeff
            for loc_v, coeff in v_coeffs.items():
                r, c = loc_v // 2, loc_v % 2
                global_v = sp['b_map'](r, c)
                V[global_v][col] += coeff
            for loc_w, coeff in w_coeffs.items():
                r, c = loc_w // 2, loc_w % 2
                global_w = sp['c_map'](r, c)
                W[global_w][col] += coeff
            col += 1
    
    return {"U": U, "V": V, "W": W}
