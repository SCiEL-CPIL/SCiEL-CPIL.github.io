import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    
    total_rank = 40
    
    U = [[0] * total_rank for _ in range(n * m)]  # 8 x 40
    V = [[0] * total_rank for _ in range(m * p)]  # 32 x 40
    W = [[0] * total_rank for _ in range(p * n)]  # 16 x 40
    
    def strassen_4x4_assign(U_mat, V_mat, W_mat, base_col, X_idx, Y_idx, Z_idx):
        # M1 = (X11+X22)*(Y11+Y22)
        for idx in X_idx['X11']:
            U_mat[idx][base_col + 0] += 1
        for idx in X_idx['X22']:
            U_mat[idx][base_col + 0] += 1
        for idx in Y_idx['Y11']:
            V_mat[idx][base_col + 0] += 1
        for idx in Y_idx['Y22']:
            V_mat[idx][base_col + 0] += 1
        for idx in Z_idx['Z11']:
            W_mat[idx][base_col + 0] += 1
        for idx in Z_idx['Z22']:
            W_mat[idx][base_col + 0] += 1
        
        # M2 = (X21+X22)*Y11
        for idx in X_idx['X21']:
            U_mat[idx][base_col + 1] += 1
        for idx in X_idx['X22']:
            U_mat[idx][base_col + 1] += 1
        for idx in Y_idx['Y11']:
            V_mat[idx][base_col + 1] += 1
        for idx in Z_idx['Z21']:
            W_mat[idx][base_col + 1] += 1
        for idx in Z_idx['Z22']:
            W_mat[idx][base_col + 1] -= 1
        
        # M3 = X11*(Y12-Y22)
        for idx in X_idx['X11']:
            U_mat[idx][base_col + 2] += 1
        for idx in Y_idx['Y12']:
            V_mat[idx][base_col + 2] += 1
        for idx in Y_idx['Y22']:
            V_mat[idx][base_col + 2] -= 1
        for idx in Z_idx['Z12']:
            W_mat[idx][base_col + 2] += 1
        for idx in Z_idx['Z22']:
            W_mat[idx][base_col + 2] += 1
        
        # M4 = X22*(Y21-Y11)
        for idx in X_idx['X22']:
            U_mat[idx][base_col + 3] += 1
        for idx in Y_idx['Y21']:
            V_mat[idx][base_col + 3] += 1
        for idx in Y_idx['Y11']:
            V_mat[idx][base_col + 3] -= 1
        for idx in Z_idx['Z11']:
            W_mat[idx][base_col + 3] += 1
        for idx in Z_idx['Z21']:
            W_mat[idx][base_col + 3] += 1
        
        # M5 = (X11+X12)*Y22
        for idx in X_idx['X11']:
            U_mat[idx][base_col + 4] += 1
        for idx in X_idx['X12']:
            U_mat[idx][base_col + 4] += 1
        for idx in Y_idx['Y22']:
            V_mat[idx][base_col + 4] += 1
        for idx in Z_idx['Z11']:
            W_mat[idx][base_col + 4] -= 1
        for idx in Z_idx['Z12']:
            W_mat[idx][base_col + 4] += 1
        
        # M6 = (X21-X11)*(Y11+Y12)
        for idx in X_idx['X21']:
            U_mat[idx][base_col + 5] += 1
        for idx in X_idx['X11']:
            U_mat[idx][base_col + 5] -= 1
        for idx in Y_idx['Y11']:
            V_mat[idx][base_col + 5] += 1
        for idx in Y_idx['Y12']:
            V_mat[idx][base_col + 5] += 1
        for idx in Z_idx['Z22']:
            W_mat[idx][base_col + 5] += 1
        
        # M7 = (X12-X22)*(Y21+Y22)
        for idx in X_idx['X12']:
            U_mat[idx][base_col + 6] += 1
        for idx in X_idx['X22']:
            U_mat[idx][base_col + 6] -= 1
        for idx in Y_idx['Y21']:
            V_mat[idx][base_col + 6] += 1
        for idx in Y_idx['Y22']:
            V_mat[idx][base_col + 6] += 1
        for idx in Z_idx['Z11']:
            W_mat[idx][base_col + 6] += 1
    
    # A_pad is 4x4 with 2x2 blocks:
    # X11 = A[:, 0:2]: U indices 0,1,4,5
    # X12 = A[:, 2:4]: U indices 2,3,6,7
    # X21 = 0: empty
    # X22 = 0: empty
    
    X_idx = {
        'X11': [0, 1, 4, 5],
        'X12': [2, 3, 6, 7],
        'X21': [],
        'X22': []
    }
    
    # First 4x4 multiplication: B_pad1 = B[:, 0:4]
    # Y11 = B[0:2, 0:2]: V indices 0,1,8,9
    # Y12 = B[0:2, 2:4]: V indices 2,3,10,11
    # Y21 = B[2:4, 0:2]: V indices 16,17,24,25
    # Y22 = B[2:4, 2:4]: V indices 18,19,26,27
    
    Y_idx1 = {
        'Y11': [0, 1, 8, 9],
        'Y12': [2, 3, 10, 11],
        'Y21': [16, 17, 24, 25],
        'Y22': [18, 19, 26, 27]
    }
    
    # Z11 = C[0:2, 0:2]: W indices 0,1,2,3 (k*2+i for k=0,1 and i=0,1)
    # Z12 = C[0:2, 2:4]: W indices 4,5,6,7
    # Z21 = 0: empty
    # Z22 = 0: empty
    
    Z_idx1 = {
        'Z11': [0, 1, 2, 3],
        'Z12': [4, 5, 6, 7],
        'Z21': [],
        'Z22': []
    }
    
    strassen_4x4_assign(U, V, W, 0, X_idx, Y_idx1, Z_idx1)
    
    # Second 4x4 multiplication: B_pad2 = B[:, 4:8]
    # Y11 = B[0:2, 4:6]: V indices 4,5,12,13
    # Y12 = B[0:2, 6:8]: V indices 6,7,14,15
    # Y21 = B[2:4, 4:6]: V indices 20,21,28,29
    # Y22 = B[2:4, 6:8]: V indices 22,23,30,31
    
    Y_idx2 = {
        'Y11': [4, 5, 12, 13],
        'Y12': [6, 7, 14, 15],
        'Y21': [20, 21, 28, 29],
        'Y22': [22, 23, 30, 31]
    }
    
    # Z11 = C[0:2, 4:6]: W indices 8,9,10,11
    # Z12 = C[0:2, 6:8]: W indices 12,13,14,15
    # Z21 = 0: empty
    # Z22 = 0: empty
    
    Z_idx2 = {
        'Z11': [8, 9, 10, 11],
        'Z12': [12, 13, 14, 15],
        'Z21': [],
        'Z22': []
    }
    
    strassen_4x4_assign(U, V, W, 20, X_idx, Y_idx2, Z_idx2)
    
    return {"U": U, "V": V, "W": W}
