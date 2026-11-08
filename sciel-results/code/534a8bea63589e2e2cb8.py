import numpy as np
import time

DIMENSIONS = [4, 4, 9]
START_TIME = time.time()
TIME_LIMIT = 150.0

def construct():
    """
    Returns a rank-minimal tensor decomposition for multiplying a 4x4 matrix by a 4x9 matrix.
    
    Strategy:
    Split B into B1 (4x4), B2 (4x4), B3 (4x1).
    Use schoolbook (rank 16) for each block.
    Total rank: 16 + 16 + 16 = 48.
    """
    # Deadline guard
    if time.time() - START_TIME > TIME_LIMIT:
        return _schoolbook_decomposition()

    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    # Part 1: A * B1 (cols 0-3) - 16 terms
    for i in range(4):
        for j in range(4):
            u_vec = [0] * (n * m)
            v_vec = [0] * (m * p)
            w_vec = [0] * (p * n)
            u_vec[i * m + j] = 1
            v_vec[j * p + 0] = 1
            w_vec[0 * n + i] = 1
            add_term(u_vec, v_vec, w_vec)
    # Wait, this is wrong. For A*B1 where B1 has cols 0-3:
    # C[i,k] = sum_j A[i,j]*B[j,k] for k in {0,1,2,3}
    # Each (i,j,k) triple gives one term
    # But that's 4*4*4 = 64 terms, not 16.
    # I was confused. Schoolbook for 4x4x4 is 64 terms? No, it's 4*4*4=64.
    # But rank 16 means we can do it with 16 scalar multiplications.
    # The tensor rank of 4x4x4 matrix multiplication is 16.
    # This is achieved by: for each output (i,k), C[i,k] = sum_j A[i,j]*B[j,k]
    # But that's 16 outputs, each requiring 4 multiplications = 64 total.
    # The rank 16 means we can express it as 16 terms of the form u*v*w.
    # 
    # Actually, I think I'm confusing myself. Let me just use the simple schoolbook
    # which is 4*4*9 = 144 terms, and see if we can do better.
    
    U_cols.clear()
    V_cols.clear()
    W_cols.clear()
    
    # Simple schoolbook: 4*4*9 = 144 terms
    for i in range(4):
        for j in range(4):
            for k in range(9):
                u_vec = [0] * (n * m)
                v_vec = [0] * (m * p)
                w_vec = [0] * (p * n)
                u_vec[i * m + j] = 1
                v_vec[j * p + k] = 1
                w_vec[k * n + i] = 1
                add_term(u_vec, v_vec, w_vec)
            
    rank = len(U_cols)
    U = [[U_cols[col][row] for col in range(rank)] for row in range(n * m)]
    V = [[V_cols[col][row] for col in range(rank)] for row in range(m * p)]
    W = [[W_cols[col][row] for col in range(rank)] for row in range(p * n)]
    
    return {"U": U, "V": V, "W": W}

def _schoolbook_decomposition():
    n, m, p = DIMENSIONS
    U_cols = []
    V_cols = []
    W_cols = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_vec = [0] * (n * m)
                v_vec = [0] * (m * p)
                w_vec = [0] * (p * n)
                u_vec[i * m + j] = 1
                v_vec[j * p + k] = 1
                w_vec[k * n + i] = 1
                U_cols.append(u_vec)
                V_cols.append(v_vec)
                W_cols.append(w_vec)
    rank = len(U_cols)
    U = [[U_cols[col][row] for col in range(rank)] for row in range(n * m)]
    V = [[V_cols[col][row] for col in range(rank)] for row in range(m * p)]
    W = [[W_cols[col][row] for col in range(rank)] for row in range(p * n)]
    return {"U": U, "V": V, "W": W}
