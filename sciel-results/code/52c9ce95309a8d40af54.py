import numpy as np
import time

def construct():
    n, m, p = 2, 4, 8
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    # First, try the known rank-56 Strassen-based decomposition
    result = _strassen_decomp(n, m, p)
    if result is not None:
        return result
    
    # Fallback to schoolbook (rank 64)
    return _schoolbook(n, m, p)

def _strassen_decomp(n, m, p):
    """
    Strassen-based rank-56 decomposition for 2x4 * 4x8.
    
    We split the 4x8 matrix B into two 4x4 blocks: B = [B1 | B2]
    Then C = A * B = [A*B1 | A*B2]
    
    For each 2x4 * 4x4 multiplication, we use a rank-28 decomposition
    (derived from Strassen's 7-term 2x2 recursion applied to the 4x4 part).
    Total rank: 28 * 2 = 56.
    
    Actually, let me think more carefully.
    
    For 2x4 * 4x4:
    A is 2x4, B is 4x4, C is 2x4.
    
    We can view this as:
    - Split A into two 2x2 blocks: A = [A1 | A2] where A1, A2 are 2x2
    - Split B into four 2x2 blocks: B = [[B11, B12], [B21, B22]]
    
    Then C = [A1*B11 + A2*B21, A1*B12 + A2*B22]
    
    Each term A1*B11 is 2x2*2x2 = 2x2, which can be done in rank 7 (Strassen).
    So we need:
    - A1*B11: rank 7
    - A1*B12: rank 7  
    - A2*B21: rank 7
    - A2*B22: rank 7
    Total: 28 per 2x4*4x4 block.
    
    For the full 2x4*4x8: 28 * 2 = 56.
    """
    A_dim = n * m  # 8
    B_dim = m * p  # 32
    C_dim = p * n  # 16
    
    rank = 56
    U = np.zeros((A_dim, rank), dtype=int)
    V = np.zeros((B_dim, rank), dtype=int)
    W = np.zeros((C_dim, rank), dtype=int)
    
    # Strassen's 7 terms for 2x2*2x2
    # For X (2x2) * Y (2x2) = Z (2x2):
    # M1 = (X00+X11)*(Y00+Y11)
    # M2 = (X10+X11)*Y00
    # M3 = X00*(Y10-Y11)
    # M4 = X11*(Y01-Y00)
    # M5 = (X00+X01)*Y11
    # M6 = (X10-X00)*(Y00+Y01)
    # M7 = (X11-X01)*(Y10+Y11)
    # Z00 = M1+M4-M5+M7
    # Z01 = M3+M5
    # Z10 = M2+M4
    # Z11 = M1-M2+M3+M6
    
    # Each term has coefficients for X, Y, Z
    # X indices: (row, col) in {0,1}x{0,1}
    # Y indices: (row, col) in {0,1}x{0,1}  
    # Z indices: (row, col) in {0,1}x{0,1}
    
    strassen_terms = [
        # (X_coeffs, Y_coeffs, Z_coeffs)
        # M1 = (X00+X11)*(Y00+Y11)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}),
        # M2 = (X10+X11)*Y00
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3 = X00*(Y10-Y11)
        ({(0,0): 1}, {(1,0): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4 = X11*(Y01-Y00)
        ({(1,1): 1}, {(0,1): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5 = (X00+X01)*Y11
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6 = (X10-X00)*(Y00+Y01)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7 = (X11-X01)*(Y10+Y11)
        ({(1,1): 1, (0,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    # For 2x4 * 4x4:
    # A = [A_left | A_right] where A_left, A_right are 2x2
    # B = [[B_topleft, B_topright], [B_botleft, B_botright]] all 2x2
    # C = [C_left | C_right] where C_left = A_left*B_topleft + A_right*B_botleft
    #                         C_right = A_left*B_topright + A_right*B_botright
    
    # For the full 2x4 * 4x8:
    # A is 2x4, B is 4x8
    # Split B into two 4x4 blocks: B = [B_block0 | B_block1]
    # For each block:
    #   Split A into A_left (2x2) and A_right (2x2)
    #   Split B_block into four 2x2 quadrants
    #   Compute 4 Strassen products (each rank 7)
    
    col = 0
    for b_block in range(2):  # two 4x4 blocks of B
        for a_block in range(2):  # A_left or A_right
            for b_quad in range(4):  # four quadrants of B_block
                # a_block: 0 = A_left (cols 0-1), 1 = A_right (cols 2-3)
                # b_quad: 0 = top-left, 1 = top-right, 2 = bot-left, 3 = bot-right
                # b_quad row: 0 = top (rows 0-1), 1 = bottom (rows 2-3)
                # b_quad col: 0 = left (cols 0-1), 1 = right (cols 2-3)
                
                b_row = b_quad // 2  # 0 or 1
                b_col = b_quad % 2    # 0 or 1
                
                for term_idx in range(7):
                    x_coeffs, y_coeffs, z_coeffs = strassen_terms[term_idx]
                    
                    # Fill U (A coefficients)
                    # A is 2x4, indexed by (i, j) where i in {0,1}, j in {0,1,2,3}
                    # a_block determines which 2 cols of A
                    for (xi, xj), coeff in x_coeffs.items():
                        global_i = xi
                        global_j = a_block * 2 + xj
                        U[global_i * m + global_j, col] += coeff
                    
                    # Fill V (B coefficients)
                    # B is 4x8, indexed by (j, k) where j in {0,1,2,3}, k in {0,1,2,3,4,5,6,7}
                    # b_block determines which 4 cols of B
                    # b_quad determines which 2x2 within the 4x4
                    for (yi, yj), coeff in y_coeffs.items():
                        global_j = b_row * 2 + yi
                        global_k = b_block * 4 + b_col * 2 + yj
                        V[global_j * p + global_k, col] += coeff
                    
                    # Fill W (C coefficients)
                    # C is 2x8, indexed by (k, i) where k in {0,1,2,3,4,5,6,7}, i in {0,1}
                    # But wait, the output factor follows W[k*n+i, r]
                    # So C index is (k, i) -> k*n + i = k*2 + i
                    # The z_coeffs give (zi, zj) where zi is row of C (i in {0,1}), zj is col of C
                    # For C_left: cols 0-1 (within the 4x4 block), for C_right: cols 2-3
                    # But we're computing A_block * B_quad, which contributes to specific C entries
                    
                    # Actually, let me reconsider the structure.
                    # C = A * B
                    # C[i, k] = sum_j A[i, j] * B[j, k]
                    
                    # When we compute A_left * B_topleft:
                    # A_left has entries A[i, 0], A[i, 1] for i in {0,1}
                    # B_topleft has entries B[j, k] for j in {0,1}, k in {0,1}
                    # Product gives contribution to C[i, k] for i in {0,1}, k in {0,1}
                    
                    # So for a_block=0 (A_left), b_quad=0 (B_topleft, b_row=0, b_col=0):
                    #   Contributes to C[i, k] for i in {0,1}, k in {0,1}
                    #   C index: k*2 + i
                    
                    for (zi, zj), coeff in z_coeffs.items():
                        # zi is the row of the 2x2 result, which corresponds to global row i of C
                        # zj is the col of the 2x2 result, which corresponds to local col within the quadrant
                        global_i = zi
                        # The col of C depends on b_block and b_col
                        # b_block=0: C cols 0-3, b_block=1: C cols 4-7
                        # b_col=0: local cols 0-1, b_col=1: local cols 2-3
                        global_k = b_block * 4 + b_col * 2 + zj
                        W[global_k * n + global_i, col] += coeff
                    
                    col += 1
    
    # Verify
    if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify that the decomposition is exact."""
    if not U or not V or not W:
        return False
    rank = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
        return False
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            u_row = U_arr[i * m + j]
            for k in range(p):
                val = np.dot(u_row, V_arr[j * p + k] * W_arr[k * n + i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
