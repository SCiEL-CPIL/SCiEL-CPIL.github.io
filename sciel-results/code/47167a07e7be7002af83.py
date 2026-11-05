import numpy as np
import time

def construct():
    n, m, p = 3, 4, 5
    rank = 56
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    col = 0
    
    def add_block_222(u_rows, v_rows, w_rows, start_col):
        # u_rows, v_rows, w_rows are lists of 4 indices
        # Returns new col
        # Strassen vectors
        u_vecs = [
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, -1],
            [0, 1, 0, -1]
        ]
        v_vecs = [
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, -1],
            [0, 0, 1, -1],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1]
        ]
        w_vecs = [
            [1, 0, 0, 1],
            [0, 0, 1, -1],
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 1]
        ]
        
        for t in range(7):
            for i in range(4):
                U[u_rows[i], start_col + t] = u_vecs[t][i]
                V[v_rows[i], start_col + t] = v_vecs[t][i]
                W[w_rows[i], start_col + t] = w_vecs[t][i]
        return start_col + 7

    def add_block_221(u_rows, v_rows, w_rows, start_col):
        # Schoolbook 2x2x1
        # A (2x2), B (2x1), C (2x1)
        # C0 = A00 B0 + A01 B1
        # C1 = A10 B0 + A11 B1
        # u_rows: [A00, A01, A10, A11]
        # v_rows: [B0, B1]
        # w_rows: [C0, C1]
        
        # Term 1: A00 B0 -> C0
        U[u_rows[0], start_col] = 1
        V[v_rows[0], start_col] = 1
        W[w_rows[0], start_col] = 1
        # Term 2: A01 B1 -> C0
        U[u_rows[1], start_col+1] = 1
        V[v_rows[1], start_col+1] = 1
        W[w_rows[0], start_col+1] = 1
        # Term 3: A10 B0 -> C1
        U[u_rows[2], start_col+2] = 1
        V[v_rows[0], start_col+2] = 1
        W[w_rows[1], start_col+2] = 1
        # Term 4: A11 B1 -> C1
        U[u_rows[3], start_col+3] = 1
        V[v_rows[1], start_col+3] = 1
        W[w_rows[1], start_col+3] = 1
        
        return start_col + 4

    def add_block_145(u_rows, v_rows, w_rows, start_col):
        # Schoolbook 1x4x5
        # A (1x4), B (4x5), C (1x5)
        # Ck = sum_j A_j B_jk
        # u_rows: [A0, A1, A2, A3]
        # v_rows: [B00, B01, B02, B03, B04, B10, ..., B34]
        # w_rows: [C0, C1, C2, C3, C4]
        
        for j in range(4):
            for k in range(5):
                U[u_rows[j], start_col + j*5 + k] = 1
                V[v_rows[j*5 + k], start_col + j*5 + k] = 1
                W[w_rows[k], start_col + j*5 + k] = 1
        return start_col + 20

    # Block 1: i in {0,1}, j in {0,1}, k in {0..4}
    u1 = [0, 1, 4, 5]
    v1 = [0, 1, 5, 6]
    w1_full = [0, 1, 3, 4, 6, 7, 9, 10, 12, 13]
    
    # Sub 1.1: k in {0,1}
    w11 = [0, 1, 3, 4]
    col = add_block_222(u1, v1, w11, col)
    
    # Sub 1.2: k in {2,3}
    w12 = [6, 7, 9, 10]
    col = add_block_222(u1, v1, w12, col)
    
    # Sub 1.3: k in {4}
    w13 = [12, 13]
    col = add_block_221(u1, v1, w13, col)
    
    # Block 2: i in {0,1}, j in {2,3}, k in {0..4}
    u2 = [2, 3, 6, 7]
    v2 = [10, 11, 15, 16]
    
    # Sub 2.1: k in {0,1}
    col = add_block_222(u2, v2, w11, col)
    
    # Sub 2.2: k in {2,3}
    col = add_block_222(u2, v2, w12, col)
    
    # Sub 2.3: k in {4}
    col = add_block_221(u2, v2, w13, col)
    
    # Block 3: i in {2}, j in {0..3}, k in {0..4}
    u3 = [8, 9, 10, 11]
    v3 = list(range(20))
    w3 = [2, 5, 8, 11, 14]
    col = add_block_145(u3, v3, w3, col)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
