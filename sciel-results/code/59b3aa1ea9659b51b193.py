import time
import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 66 construction
    rank = 66
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: Row 0 of C (1 x 4 x 6)
    # A[0, j] * B[j, k] -> C[0, k]
    for j in range(m):
        for k in range(p):
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1
            
    # Part 2: Rows 1,2 of C (2 x 4 x 6)
    # Split inner dimension m=4 into 2+2
    # A_L = A[1:3, 0:2], A_R = A[1:3, 2:4]
    # B_T = B[0:2, :], B_B = B[2:4, :]
    # C = A_L B_T + A_R B_B
    
    # Helper to add Strassen block
    def add_strassen_block(U, V, W, col, a_idx, b_idx, c_idx):
        # a_idx: [a00, a01, a10, a11] global U indices
        # b_idx: [b00, b01, b10, b11] global V indices
        # c_idx: [c00, c01, c10, c11] global W indices
        # Returns next col
        c = col
        # M1 = (a00 + a11)(b00 + b11)
        U[a_idx[0]][c] = 1
        U[a_idx[3]][c] = 1
        V[b_idx[0]][c] = 1
        V[b_idx[3]][c] = 1
        W[c_idx[0]][c] = 1
        W[c_idx[3]][c] = 1
        W[c_idx[1]][c] = -1 # C00 = M1 - M4 + M6, C11 = M1 - M3 + M7
        # Wait, C00 uses M1, C11 uses M1.
        # C00 = M1 - M4 + M6
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 + M7
        
        # Let's trace contributions:
        # M1 contributes to C00 (+1) and C11 (+1)
        W[c_idx[0]][c] = 1
        W[c_idx[3]][c] = 1
        c += 1
        
        # M2 = (a10 + a11)b00
        U[a_idx[2]][c] = 1
        U[a_idx[3]][c] = 1
        V[b_idx[0]][c] = 1
        W[c_idx[2]][c] = 1 # C10
        c += 1
        
        # M3 = a00(b01 - b11)
        U[a_idx[0]][c] = 1
        V[b_idx[1]][c] = 1
        V[b_idx[3]][c] = -1
        W[c_idx[1]][c] = 1 # C01
        W[c_idx[3]][c] = -1 # C11
        c += 1
        
        # M4 = a11(b10 - b00)
        U[a_idx[3]][c] = 1
        V[b_idx[2]][c] = 1
        V[b_idx[0]][c] = -1
        W[c_idx[0]][c] = -1 # C00
        W[c_idx[2]][c] = 1 # C10
        c += 1
        
        # M5 = (a00 + a01)b11
        U[a_idx[0]][c] = 1
        U[a_idx[1]][c] = 1
        V[b_idx[3]][c] = 1
        W[c_idx[1]][c] = 1 # C01
        c += 1
        
        # M6 = (a10 - a00)(b00 + b01)
        U[a_idx[2]][c] = 1
        U[a_idx[0]][c] = -1
        V[b_idx[0]][c] = 1
        V[b_idx[1]][c] = 1
        W[c_idx[0]][c] = 1 # C00
        c += 1
        
        # M7 = (a01 - a11)(b10 + b11)
        U[a_idx[1]][c] = 1
        U[a_idx[3]][c] = -1
        V[b_idx[2]][c] = 1
        V[b_idx[3]][c] = 1
        W[c_idx[3]][c] = 1 # C11
        c += 1
        
        return c

    # Process A_L B_T
    # A_L rows 1,2; cols 0,1
    a_L_idx = [1*m + 0, 1*m + 1, 2*m + 0, 2*m + 1]
    
    for j in range(3):
        # B_T cols 2j, 2j+1; rows 0,1
        b_T_idx = [0*p + 2*j, 0*p + 2*j + 1, 1*p + 2*j, 1*p + 2*j + 1]
        # C rows 1,2; cols 2j, 2j+1
        c_idx = [2*j*n + 1, (2*j+1)*n + 1, 2*j*n + 2, (2*j+1)*n + 2]
        col = add_strassen_block(U, V, W, col, a_L_idx, b_T_idx, c_idx)
        
    # Process A_R B_B
    # A_R rows 1,2; cols 2,3
    a_R_idx = [1*m + 2, 1*m + 3, 2*m + 2, 2*m + 3]
    
    for j in range(3):
        # B_B cols 2j, 2j+1; rows 2,3
        b_B_idx = [2*p + 2*j, 2*p + 2*j + 1, 3*p + 2*j, 3*p + 2*j + 1]
        # C rows 1,2; cols 2j, 2j+1 (same as before)
        c_idx = [2*j*n + 1, (2*j+1)*n + 1, 2*j*n + 2, (2*j+1)*n + 2]
        col = add_strassen_block(U, V, W, col, a_R_idx, b_B_idx, c_idx)

    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
