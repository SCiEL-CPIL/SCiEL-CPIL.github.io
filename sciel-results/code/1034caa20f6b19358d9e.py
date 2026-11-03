"""Tensor decomposition for 3x3x7 matrix multiplication."""
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 7
    # Rank 23 for 3x3x3 (Laderman) + 36 for 3x3x4 (Trivial) = 59
    rank = 59
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Laderman's Algorithm for 3x3x3 part (columns 0,1,2 of B)
    # U indices: i*3 + j
    # V indices: j*7 + k (k in 0,1,2)
    # W indices: k*3 + i
    
    # Define the 23 products
    # Each product is a tuple (U_coeffs, V_coeffs, W_coeffs)
    # But it's easier to define the linear forms directly.
    
    # Helper to add a term
    def add_term(col_idx, u_coeffs, v_coeffs, w_coeffs):
        for idx, val in u_coeffs.items():
            U[idx, col_idx] += val
        for idx, val in v_coeffs.items():
            V[idx, col_idx] += val
        for idx, val in w_coeffs.items():
            W[idx, col_idx] += val

    # We need to map (i,j) to index i*3+j
    # (j,k) to index j*7+k
    # (k,i) to index k*3+i
    
    # Laderman Multiplications
    # M1
    add_term(0, {0*3+0: 1}, {0*7+0: 1}, {0*3+0: 1})
    # M2
    add_term(1, {0*3+1: 1}, {1*7+0: 1}, {0*3+1: 1}) # Wait, M2 contributes to c10?
    # Let's use the reconstruction formulas to fill W.
    # It is safer to define the 23 M's and then sum them into W.
    
    # Reset W to 0, fill U and V for the 23 terms, then compute W contributions.
    # Actually, the decomposition is T = sum M_k.
    # So for each k, we have u_k, v_k, w_k.
    # T_{a,b,c} = sum u_k[a] v_k[b] w_k[c].
    
    # Let's define u_k, v_k, w_k for k=0..22.
    
    u_list = []
    v_list = []
    w_list = []
    
    # M1: a00 b00 -> c00
    u_list.append({0: 1})
    v_list.append({0: 1}) # j=0, k=0 -> 0*7+0
    w_list.append({0: 1}) # k=0, i=0 -> 0*3+0
    
    # M2: a01 b10 -> c10? No, a01 b10 is part of c01?
    # Let's stick to the reconstruction.
    # c00 = M1 + M10 - M11 - M12
    # This implies M1 has w_00 = 1, M10 has w_00 = 1, M11 has w_00 = -1, M12 has w_00 = -1.
    
    # I will generate the U and V coefficients for the 23 terms,
    # and then calculate the W coefficients by applying the reconstruction formulas.
    
    # Term 1: a00 b00
    u1 = {0: 1} # a00
    v1 = {0: 1} # b00 (j=0, k=0)
    
    # Term 2: a01 b10
    u2 = {1: 1} # a01 (i=0, j=1 -> 0*3+1=1)
    v2 = {7: 1} # b10 (j=1, k=0 -> 1*7+0=7)
    
    # Term 3: a02 b20
    u3 = {2: 1} # a02 (i=0, j=2 -> 2)
    v3 = {14: 1} # b20 (j=2, k=0 -> 14)
    
    # Term 4: a10 b01
    u4 = {3: 1} # a10 (i=1, j=0 -> 3)
    v4 = {1: 1} # b01 (j=0, k=1 -> 1)
    
    # Term 5: a11 b11
    u5 = {4: 1} # a11
    v5 = {8: 1} # b11 (j=1, k=1 -> 8)
    
    # Term 6: a12 b21
    u6 = {5: 1} # a12
    v6 = {15: 1} # b21 (j=2, k=1 -> 15)
    
    # Term 7: a20 b02
    u7 = {6: 1} # a20
    v7 = {2: 1} # b02 (j=0, k=2 -> 2)
    
    # Term 8: a21 b12
    u8 = {7: 1} # a21
    v8 = {9: 1} # b12 (j=1, k=2 -> 9)
    
    # Term 9: a22 b22
    u9 = {8: 1} # a22
    v9 = {16: 1} # b22 (j=2, k=2 -> 16)
    
    # Term 10: (a00+a11)(b01+b10)
    u10 = {0: 1, 4: 1}
    v10 = {1: 1, 7: 1}
    
    # Term 11: (a01+a11)(b10+b00)
    u11 = {1: 1, 4: 1}
    v11 = {7: 1, 0: 1}
    
    # Term 12: (a10+a00)(b01+b00)
    u12 = {3: 1, 0: 1}
    v12 = {1: 1, 0: 1}
    
    # Term 13: (a11+a10)(b00+b01)
    u13 = {4: 1, 3: 1}
    v13 = {0: 1, 1: 1}
    
    # Term 14: (a02+a12)(b20+b10)
    u14 = {2: 1, 5: 1}
    v14 = {14: 1, 7: 1}
    
    # Term 15: (a12+a22)(b02+b22)
    u15 = {5: 1, 8: 1}
    v15 = {2: 1, 16: 1}
    
    # Term 16: (a22+a20)(b20+b02)
    u16 = {8: 1, 6: 1}
    v16 = {14: 1, 2: 1}
    
    # Term 17: (a00+a22)(b20+b02)
    u17 = {0: 1, 8: 1}
    v17 = {14: 1, 2: 1}
    
    # Term 18: (a20+a00)(b02+b22)
    u18 = {6: 1, 0: 1}
    v18 = {2: 1, 16: 1}
    
    # Term 19: (a02+a22)(b00+b22)
    u19 = {2: 1, 8: 1}
    v19 = {0: 1, 16: 1}
    
    # Term 20: (a22+a02)(b20+b00)
    u20 = {8: 1, 2: 1}
    v20 = {14: 1, 0: 1}
    
    # Term 21: (a00+a22)(b00+b22)
    u21 = {0: 1, 8: 1}
    v21 = {0: 1, 16: 1}
    
    # Term 22: (a01+a21)(b10+b20)
    u22 = {1: 1, 7: 1}
    v22 = {7: 1, 14: 1}
    
    # Term 23: (a21+a01)(b20+b10)
    u23 = {7: 1, 1: 1}
    v23 = {14: 1, 7: 1}
    
    # Fill U and V for columns 0-22
    for k in range(23):
        for idx, val in u_list[k].items():
            U[idx, k] = val
        for idx, val in v_list[k].items():
            V[idx, k] = val
            
    # Calculate W coefficients for columns 0-22 based on reconstruction
    # c00 = M1 + M10 - M11 - M12
    # c01 = M4 + M13 - M10 - M12
    # c02 = M3 + M14 - M15 - M19
    # c10 = M2 + M11 - M10 - M13
    # c11 = M5 + M10 - M11 - M13
    # c12 = M6 + M15 - M14 - M19
    # c20 = M7 + M16 - M17 - M22
    # c21 = M8 + M23 - M16 - M22
    # c22 = M9 + M17 - M16 - M23
    
    # W index for c_{i,k} is k*3 + i
    # c00 -> 0
    # c01 -> 1
    # c02 -> 2
    # c10 -> 3
    # c11 -> 4
    # c12 -> 5
    # c20 -> 6
    # c21 -> 7
    # c22 -> 8
    
    # M1 (col 0)
    W[0, 0] += 1
    # M10 (col 9)
    W[0, 9] += 1
    # M11 (col 10)
    W[0, 10] += -1
    # M12 (col 11)
    W[0, 11] += -1
    
    # M4 (col 3)
    W[1, 3] += 1
    # M13 (col 12)
    W[1, 12] += 1
    # M10 (col 9)
    W[1, 9] += -1
    # M12 (col 11)
    W[1, 11] += -1
    
    # M3 (col 2)
    W[2, 2] += 1
    # M14 (col 13)
    W[2, 13] += 1
    # M15 (col 14)
    W[2, 14] += -1
    # M19 (col 18)
    W[2, 18] += -1
    
    # M2 (col 1)
    W[3, 1] += 1
    # M11 (col 10)
    W[3, 10] += 1
    # M10 (col 9)
    W[3, 9] += -1
    # M13 (col 12)
    W[3, 12] += -1
    
    # M5 (col 4)
    W[4, 4] += 1
    # M10 (col 9)
    W[4, 9] += 1
    # M11 (col 10)
    W[4, 10] += -1
    # M13 (col 12)
    W[4, 12] += -1
    
    # M6 (col 5)
    W[5, 5] += 1
    # M15 (col 14)
    W[5, 14] += 1
    # M14 (col 13)
    W[5, 13] += -1
    # M19 (col 18)
    W[5, 18] += -1
    
    # M7 (col 6)
    W[6, 6] += 1
    # M16 (col 15)
    W[6, 15] += 1
    # M17 (col 16)
    W[6, 16] += -1
    # M22 (col 21)
    W[6, 21] += -1
    
    # M8 (col 7)
    W[7, 7] += 1
    # M23 (col 22)
    W[7, 22] += 1
    # M16 (col 15)
    W[7, 15] += -1
    # M22 (col 21)
    W[7, 21] += -1
    
    # M9 (col 8)
    W[8, 8] += 1
    # M17 (col 16)
    W[8, 16] += 1
    # M16 (col 15)
    W[8, 15] += -1
    # M23 (col 22)
    W[8, 22] += -1
    
    # Trivial part for columns 3-6 of B (indices 3,4,5,6)
    col = 23
    for i in range(3):
        for j in range(3):
            for k in range(3, 7):
                U[i*3 + j, col] = 1
                V[j*7 + k, col] = 1
                W[k*3 + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
