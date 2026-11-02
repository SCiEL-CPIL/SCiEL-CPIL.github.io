import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(u_idx, u_coeff, v_idx, v_coeff, w_idx, w_coeff, col):
        U[u_idx][col] += u_coeff
        V[v_idx][col] += v_coeff
        W[w_idx][col] += w_coeff

    # Indices for A (2x4)
    # A_L: 0,1,2,3
    # A_R: 4,5,6,7
    a_L = [0, 1, 2, 3]
    a_R = [4, 5, 6, 7]
    
    # Indices for B (4x5)
    # B_T1 (2x2): 0,1,5,6
    # B_T2 (2x2): 2,3,7,8
    # B_T3 (2x1): 4,9
    # B_B1 (2x2): 10,11,15,16
    # B_B2 (2x2): 12,13,17,18
    # B_B3 (2x1): 14,19
    b_T1 = [0, 1, 5, 6]
    b_T2 = [2, 3, 7, 8]
    b_T3 = [4, 9]
    b_B1 = [10, 11, 15, 16]
    b_B2 = [12, 13, 17, 18]
    b_B3 = [14, 19]
    
    # Indices for C (2x5) -> W index k*n + i
    # C[:, 0:2] -> k=0,1 -> W[0..3]
    c_T1 = [0, 2, 1, 3] # c11, c12, c21, c22
    # C[:, 2:4] -> k=2,3 -> W[4..7]
    c_T2 = [4, 6, 5, 7]
    # C[:, 4:5] -> k=4 -> W[8..9]
    c_T3 = [8, 9] # c1, c2
    
    def strassen_2x2x2(a_idx, b_idx, c_idx, col_offset):
        # a_idx: [a11, a12, a21, a22]
        # b_idx: [b11, b12, b21, b22]
        # c_idx: [c11, c12, c21, c22]
        
        # M1 = (A11+A22)(B11+B22)
        add_term(a_idx[0], 1, b_idx[0], 1, c_idx[0], 1, col_offset)
        add_term(a_idx[0], 1, b_idx[1], 1, c_idx[1], 0, col_offset) # Wait, M1 contributes to c11 and c22
        # Correction:
        # M1 contributes to c11 (+1) and c22 (+1)
        # My add_term adds to a specific w_idx.
        # So I need to call add_term for each w contribution.
        
        # Let's rewrite the helper to be clearer or just call add_term multiple times.
        pass

    # Re-implement Strassen correctly with multiple add_term calls
    
    def add_strassen(a_idx, b_idx, c_idx, col_offset):
        # M1
        add_term(a_idx[0], 1, b_idx[0], 1, c_idx[0], 1, col_offset)
        add_term(a_idx[0], 1, b_idx[0], 1, c_idx[3], 1, col_offset)
        add_term(a_idx[0], 1, b_idx[1], 1, c_idx[0], 0, col_offset) # No, U and V must match for a rank-1 term
        # Rank-1 term: u_i v_j w_k.
        # For M1: (a11+a22)(b11+b22).
        # U: a11=1, a22=1. V: b11=1, b22=1.
        # W: c11=1, c22=1.
        # This is ONE rank-1 term?
        # No, (a11+a22) is a vector in U space. (b11+b22) is a vector in V space.
        # The product is a scalar.
        # It contributes to c11 and c22.
        # So in the matrix W, row c11 gets +1, row c22 gets +1.
        # This is valid. One column in U, V, W.
        
        col = col_offset
        # M1
        U[a_idx[0]][col] += 1
        U[a_idx[3]][col] += 1
        V[b_idx[0]][col] += 1
        V[b_idx[3]][col] += 1
        W[c_idx[0]][col] += 1
        W[c_idx[3]][col] += 1
        
        # M2 = (A21+A22)B11
        U[a_idx[2]][col+1] += 1
        U[a_idx[3]][col+1] += 1
        V[b_idx[0]][col+1] += 1
        W[c_idx[2]][col+1] += 1
        W[c_idx[3]][col+1] -= 1
        
        # M3 = A11(B12-B22)
        U[a_idx[0]][col+2] += 1
        V[b_idx[1]][col+2] += 1
        V[b_idx[3]][col+2] -= 1
        W[c_idx[1]][col+2] += 1
        W[c_idx[3]][col+2] += 1
        
        # M4 = A22(B21-B11)
        U[a_idx[3]][col+3] += 1
        V[b_idx[2]][col+3] += 1
        V[b_idx[0]][col+3] -= 1
        W[c_idx[0]][col+3] += 1
        W[c_idx[2]][col+3] += 1
        
        # M5 = (A11+A12)B22
        U[a_idx[0]][col+4] += 1
        U[a_idx[1]][col+4] += 1
        V[b_idx[3]][col+4] += 1
        W[c_idx[0]][col+4] -= 1
        W[c_idx[1]][col+4] += 1
        
        # M6 = (A21-A11)(B11+B12)
        U[a_idx[2]][col+5] += 1
        U[a_idx[0]][col+5] -= 1
        V[b_idx[0]][col+5] += 1
        V[b_idx[1]][col+5] += 1
        W[c_idx[3]][col+5] += 1
        
        # M7 = (A12-A22)(B21+B22)
        U[a_idx[1]][col+6] += 1
        U[a_idx[3]][col+6] -= 1
        V[b_idx[2]][col+6] += 1
        V[b_idx[3]][col+6] += 1
        W[c_idx[0]][col+6] += 1

    def add_schoolbook_2x2x1(a_idx, b_idx, c_idx, col_offset):
        # C1 = A11 B1 + A12 B2
        # C2 = A21 B1 + A22 B2
        col = col_offset
        # Term 1: A11 B1 -> C1
        U[a_idx[0]][col] += 1
        V[b_idx[0]][col] += 1
        W[c_idx[0]][col] += 1
        
        # Term 2: A12 B2 -> C1
        U[a_idx[1]][col+1] += 1
        V[b_idx[1]][col+1] += 1
        W[c_idx[0]][col+1] += 1
        
        # Term 3: A21 B1 -> C2
        U[a_idx[2]][col+2] += 1
        V[b_idx[0]][col+2] += 1
        W[c_idx[1]][col+2] += 1
        
        # Term 4: A22 B2 -> C2
        U[a_idx[3]][col+3] += 1
        V[b_idx[1]][col+3] += 1
        W[c_idx[1]][col+3] += 1

    # Apply Strassen
    add_strassen(a_L, b_T1, c_T1, 0)
    add_strassen(a_R, b_B1, c_T1, 7)
    add_strassen(a_L, b_T2, c_T2, 14)
    add_strassen(a_R, b_B2, c_T2, 21)
    
    # Apply Schoolbook
    add_schoolbook_2x2x1(a_L, b_T3, c_T3, 28)
    add_schoolbook_2x2x1(a_R, b_B3, c_T3, 32)

    return {"U": U, "V": V, "W": W}
