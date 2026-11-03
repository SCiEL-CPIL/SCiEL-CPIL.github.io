import numpy as np
import time

def construct():
    n, m, p = 2, 4, 8
    
    rank = 56
    U = [[0]*rank for _ in range(8)]
    V = [[0]*rank for _ in range(32)]
    W = [[0]*rank for _ in range(16)]
    
    col = 0
    for a_block in range(2):
        for b_block in range(4):
            if a_block == 0:
                b_row_base = 0
            else:
                b_row_base = 2
            
            if a_block == 0:
                a00_u, a01_u, a10_u, a11_u = 0, 1, 4, 5
            else:
                a00_u, a01_u, a10_u, a11_u = 2, 3, 6, 7
            
            b_col_base = b_block * 2
            b00_v = (b_row_base + 0)*8 + (b_col_base + 0)
            b01_v = (b_row_base + 0)*8 + (b_col_base + 1)
            b10_v = (b_row_base + 1)*8 + (b_col_base + 0)
            b11_v = (b_row_base + 1)*8 + (b_col_base + 1)
            
            c00_w = (b_col_base + 0)*2 + 0
            c01_w = (b_col_base + 0)*2 + 1
            c10_w = (b_col_base + 1)*2 + 0
            c11_w = (b_col_base + 1)*2 + 1
            
            strassen_A = [
                [1, 1, 0, 0],
                [0, 0, 1, 1],
                [1, 0, 0, 0],
                [0, 0, 0, 1],
                [1, 1, 0, 0],
                [-1, 0, 1, 0],
                [0, 1, 0, -1],
            ]
            
            strassen_B = [
                [1, 0, 0, 1],
                [1, 0, 0, 0],
                [0, 1, 0, -1],
                [0, 0, 1, -1],
                [0, 0, 0, 1],
                [1, 1, 0, 0],
                [0, 0, 1, 1],
            ]
            
            M_to_C = [
                [(c00_w, 1), (c11_w, 1)],
                [(c10_w, 1), (c11_w, -1)],
                [(c01_w, 1), (c11_w, 1)],
                [(c00_w, 1), (c10_w, 1)],
                [(c00_w, -1), (c01_w, 1)],
                [(c11_w, 1)],
                [(c00_w, 1)],
            ]
            
            for s in range(7):
                a_coeffs = strassen_A[s]
                U[a00_u][col] += a_coeffs[0]
                U[a01_u][col] += a_coeffs[1]
                U[a10_u][col] += a_coeffs[2]
                U[a11_u][col] += a_coeffs[3]
                
                b_coeffs = strassen_B[s]
                V[b00_v][col] += b_coeffs[0]
                V[b01_v][col] += b_coeffs[1]
                V[b10_v][col] += b_coeffs[2]
                V[b11_v][col] += b_coeffs[3]
                
                for (w_idx, coeff) in M_to_C[s]:
                    W[w_idx][col] += coeff
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
