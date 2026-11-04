import time
import numpy as np

DIMENSIONS = [2, 4, 4]
TIME_LIMIT = 150.0
START_TIME = time.time()

def construct():
    global START_TIME
    START_TIME = time.time()
    deadline = START_TIME + TIME_LIMIT - 10.0
    
    try:
        result = _construct_strassen_28()
        if result is not None:
            if _verify_decomposition(result):
                return result
    except Exception:
        pass
    
    return _construct_schoolbook_32()

def _verify_decomposition(result, trials=5):
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

def _construct_strassen_28():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_block(col_offset, X_rows, Y_rows, Z_rows):
        # 7-term Strassen decomposition for 2x2x2 block
        # X = [X00, X01, X10, X11] -> X_rows[0], X_rows[1], X_rows[2], X_rows[3]
        # Y = [Y00, Y01, Y10, Y11] -> Y_rows[0], Y_rows[1], Y_rows[2], Y_rows[3]
        # Z = [Z00, Z01, Z10, Z11] -> Z_rows[0], Z_rows[1], Z_rows[2], Z_rows[3]
        
        # P1 = (X00+X11)*(Y00+Y11)
        U[X_rows[0]][col_offset + 0] += 1
        U[X_rows[3]][col_offset + 0] += 1
        V[Y_rows[0]][col_offset + 0] += 1
        V[Y_rows[3]][col_offset + 0] += 1
        W[Z_rows[0]][col_offset + 0] += 1
        W[Z_rows[3]][col_offset + 0] += 1
        
        # P2 = (X10+X11)*Y00
        U[X_rows[2]][col_offset + 1] += 1
        U[X_rows[3]][col_offset + 1] += 1
        V[Y_rows[0]][col_offset + 1] += 1
        W[Z_rows[2]][col_offset + 1] += 1
        W[Z_rows[3]][col_offset + 1] -= 1
        
        # P3 = X00*(Y01-Y11)
        U[X_rows[0]][col_offset + 2] += 1
        V[Y_rows[1]][col_offset + 2] += 1
        V[Y_rows[3]][col_offset + 2] -= 1
        W[Z_rows[1]][col_offset + 2] += 1
        W[Z_rows[3]][col_offset + 2] += 1
        
        # P4 = X11*(Y10-Y00)
        U[X_rows[3]][col_offset + 3] += 1
        V[Y_rows[2]][col_offset + 3] += 1
        V[Y_rows[0]][col_offset + 3] -= 1
        W[Z_rows[0]][col_offset + 3] += 1
        W[Z_rows[2]][col_offset + 3] += 1
        
        # P5 = (X00+X01)*Y11
        U[X_rows[0]][col_offset + 4] += 1
        U[X_rows[1]][col_offset + 4] += 1
        V[Y_rows[3]][col_offset + 4] += 1
        W[Z_rows[0]][col_offset + 4] -= 1
        W[Z_rows[1]][col_offset + 4] += 1
        
        # P6 = (X10-X00)*(Y00+Y01)
        U[X_rows[2]][col_offset + 5] += 1
        U[X_rows[0]][col_offset + 5] -= 1
        V[Y_rows[0]][col_offset + 5] += 1
        V[Y_rows[1]][col_offset + 5] += 1
        W[Z_rows[3]][col_offset + 5] += 1
        
        # P7 = (X01-X11)*(Y10+Y11)
        U[X_rows[1]][col_offset + 6] += 1
        U[X_rows[3]][col_offset + 6] -= 1
        V[Y_rows[2]][col_offset + 6] += 1
        V[Y_rows[3]][col_offset + 6] += 1
        W[Z_rows[0]][col_offset + 6] += 1
    
    # Four 2x2x2 blocks:
    # Block 1: A_left * B_TL -> C_left
    # A_left: A[0:2, 0:2] -> indices [0,1,4,5]
    # B_TL:  B[0:2, 0:2] -> indices [0,1,4,5]
    # C_left: C[0:2, 0:2] -> W indices: c*2+i where c in {0,1}, i in {0,1}
    #   Z00 -> C[0,0] -> W[0*2+0] = W[0]
    #   Z01 -> C[0,1] -> W[1*2+0] = W[2]
    #   Z10 -> C[1,0] -> W[0*2+1] = W[1]
    #   Z11 -> C[1,1] -> W[1*2+1] = W[3]
    X1 = [0, 1, 4, 5]
    Y1 = [0, 1, 4, 5]
    Z1 = [0, 2, 1, 3]
    add_block(0, X1, Y1, Z1)
    
    # Block 2: A_right * B_BL -> C_left
    # A_right: A[0:2, 2:4] -> indices [2,3,6,7]
    # B_BL:  B[2:4, 0:2] -> indices [8,9,12,13]
    # C_left: same as above
    X2 = [2, 3, 6, 7]
    Y2 = [8, 9, 12, 13]
    Z2 = [0, 2, 1, 3]
    add_block(7, X2, Y2, Z2)
    
    # Block 3: A_left * B_TR -> C_right
    # A_left: A[0:2, 0:2] -> indices [0,1,4,5]
    # B_TR:  B[0:2, 2:4] -> indices [2,3,6,7]
    # C_right: C[0:2, 2:4] -> W indices: c*2+i where c in {2,3}, i in {0,1}
    #   Z00 -> C[0,2] -> W[2*2+0] = W[4]
    #   Z01 -> C[0,3] -> W[3*2+0] = W[6]
    #   Z10 -> C[1,2] -> W[2*2+1] = W[5]
    #   Z11 -> C[1,3] -> W[3*2+1] = W[7]
    X3 = [0, 1, 4, 5]
    Y3 = [2, 3, 6, 7]
    Z3 = [4, 6, 5, 7]
    add_block(14, X3, Y3, Z3)
    
    # Block 4: A_right * B_BR -> C_right
    # A_right: A[0:2, 2:4] -> indices [2,3,6,7]
    # B_BR:  B[2:4, 2:4] -> indices [10,11,14,15]
    # C_right: same as above
    X4 = [2, 3, 6, 7]
    Y4 = [10, 11, 14, 15]
    Z4 = [4, 6, 5, 7]
    add_block(21, X4, Y4, Z4)
    
    return {"U": U, "V": V, "W": W}
