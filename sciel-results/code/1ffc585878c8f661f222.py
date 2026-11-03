import time
import numpy as np

def construct():
    deadline = time.time() + 140
    n, m, p = 4, 4, 9
    
    # Build 4x4x4 block schoolbook decomposition
    def build_4x4x4(col_offset):
        """Build U, V, W for 4x4 x 4x4 -> 4x4, with B columns offset by col_offset."""
        U = np.zeros((16, 16))
        V = np.zeros((36, 16))
        W = np.zeros((36, 16))
        col = 0
        # Block structure: A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
        # C11 = A11*B11 + A12*B21
        # C12 = A11*B12 + A12*B22
        # C21 = A21*B11 + A22*B21
        # C22 = A21*B12 + A22*B22
        
        blocks = [
            # (A_row_block, A_col_block, B_row_block, B_col_block, C_row_block, C_col_block)
            (0, 0, 0, 0, 0, 0),  # A11*B11 -> C11
            (0, 1, 1, 0, 0, 0),  # A12*B21 -> C11
            (0, 0, 0, 1, 0, 1),  # A11*B12 -> C12
            (0, 1, 1, 1, 0, 1),  # A12*B22 -> C12
            (1, 0, 0, 0, 1, 0),  # A21*B11 -> C21
            (1, 1, 1, 0, 1, 0),  # A22*B21 -> C21
            (1, 0, 0, 1, 1, 1),  # A21*B12 -> C22
            (1, 1, 1, 1, 1, 1),  # A22*B22 -> C22
        ]
        
        for (ar, ac, br, bc, cr, cc) in blocks:
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        u_idx = (ar*2 + i) * m + (ac*2 + j)
                        v_idx = (br*2 + j) * p + (bc*2 + k + col_offset)
                        w_idx = (bc*2 + k + col_offset) * n + (cr*2 + i)
                        U[u_idx, col] = 1
                        V[v_idx, col] = 1
                        W[w_idx, col] = 1
                        col += 1
        
        return U, V, W
    
    # Build 4x4x1 schoolbook decomposition
    def build_4x4x1(col_offset):
        """Build U, V, W for 4x4 x 4x1 -> 4x1, with B column offset by col_offset."""
        U = np.zeros((16, 16))
        V = np.zeros((36, 16))
        W = np.zeros((36, 16))
        col = 0
        for i in range(4):
            for j in range(4):
                U[i*m + j, col] = 1
                V[j*p + (col_offset), col] = 1
                W[(col_offset)*n + i, col] = 1
                col += 1
        return U, V, W
    
    # Part 1: B columns 0-3
    U1, V1, W1 = build_4x4x4(0)
    # Part 2: B columns 4-7
    U2, V2, W2 = build_4x4x4(4)
    # Part 3: B column 8
    U3, V3, W3 = build_4x4x1(8)
    
    U = np.hstack([U1, U2, U3])
    V = np.hstack([V1, V2, V3])
    W = np.hstack([W1, W2, W3])
    
    # Validate
    def validate(U_arr, V_arr, W_arr, tol=1e-9):
        r = U_arr.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_arr[i*m+j, :], V_arr[j*p+k, :] * W_arr[k*n+i, :])
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    if not validate(U, V, W):
        # Fallback to schoolbook
        rank_sb = n * m * p
        U_sb = np.zeros((n*m, rank_sb))
        V_sb = np.zeros((m*p, rank_sb))
        W_sb = np.zeros((p*n, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m + j, col] = 1
                    V_sb[j*p + k, col] = 1
                    W_sb[k*n + i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
