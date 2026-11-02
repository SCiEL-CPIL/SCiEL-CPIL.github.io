import numpy as np
import time

DIMENSIONS = [2, 2, 15]

def get_strassen_2x2_2():
    # Strassen decomposition for 2x2 matrix multiplication
    # U: (4, 7), V: (4, 7), W: (4, 7)
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B10 - B11)
    # M4 = A11 * (B01 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    # M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = 1
    # C00 += M1, C01 -= M1, C10 += M1, C11 -= M1 ? 
    # C00 = A00B00 + A01B10
    # C01 = A00B01 + A01B11
    # C10 = A10B00 + A11B10
    # C11 = A10B01 + A11B11
    
    # Let's verify W coefficients for M1:
    # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
    # C00 needs A00B00. Coeff +1.
    # C01 needs A00B01 + A01B11. M1 has A00B11. This doesn't match directly.
    # Standard Strassen:
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 - M6
    
    # So W for M1:
    # W[0, 0] = 1 (C00)
    # W[1, 0] = 0 (C01)
    # W[2, 0] = 0 (C10)
    # W[3, 0] = 1 (C11)
    
    # Let's re-define W based on the linear combinations.
    
    # M1: U: A00, A11. V: B00, B11.
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: U: A10, A11. V: B00.
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1 # C10 += M2, C11 -= M2
    
    # M3: U: A00. V: B10, B11.
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1 # C01 += M3, C11 += M3
    
    # M4: U: A11. V: B01, B00.
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1; W[2, 3] = 1 # C00 -= M4, C10 += M4
    
    # M5: U: A00, A01. V: B11.
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[1, 4] = 1 # C00 += M5, C01 += M5
    
    # M6: U: A10, A00. V: B00, B01.
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = -1 # C11 -= M6
    
    # M7: U: A01, A11. V: B10, B11.
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1 # C00 -= M7
    
    return U, V, W

def get_schoolbook_2x2_1():
    # 2x2 * 2x1 -> 2x1
    # Rank 4
    U = np.zeros((4, 4), dtype=int)
    V = np.zeros((2, 4), dtype=int)
    W = np.zeros((2, 4), dtype=int)
    
    # C0 = A00*B0 + A01*B1
    # C1 = A10*B0 + A11*B1
    
    # Term 0: A00*B0 -> C0
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    # Term 1: A01*B1 -> C0
    U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
    # Term 2: A10*B0 -> C1
    U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1
    # Term 3: A11*B1 -> C1
    U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    
    # Try to find rank 42 decomposition (14 for 2x5, 28 for 2x10)
    # This is complex to search. We will fallback to Strassen blocks (rank 53)
    # which is a guaranteed improvement over 60.
    
    # Construct Strassen-based decomposition
    # 7 blocks of 2x2 (rank 7 each) + 1 block of 2x1 (rank 4)
    # Total rank = 7*7 + 4 = 53
    
    U_s, V_s, W_s = get_strassen_2x2_2()
    U_1, V_1, W_1 = get_schoolbook_2x2_1()
    
    rank = 53
    U_full = np.zeros((n*m, rank), dtype=int)
    V_full = np.zeros((m*p, rank), dtype=int)
    W_full = np.zeros((p*n, rank), dtype=int)
    
    col_offset = 0
    for b in range(7):
        # Copy U_s
        U_full[:, col_offset:col_offset+7] = U_s
        
        # Copy V_s into correct column slice
        # Block b covers cols 2b, 2b+1
        start_col = 2*b
        V_full[0:2, start_col:start_col+2, col_offset:col_offset+7] = V_s
        # Note: V_s is 2x2 (local rows 0,1 correspond to global rows 0,1)
        # But V_full is m x p = 2 x 15.
        # So V_full[0, start_col] corresponds to B[0, 2b]
        # V_s[0, r] is the coefficient for B[0, 0] in local block.
        # So we place V_s into V_full[0:2, start_col:start_col+2, :]
        
        # Copy W_s into correct row slice
        # Block b covers rows 2b, 2b+1 of C (which are cols of B? No, C is p x n)
        # W is p x n = 15 x 2.
        # Local W_s is 2 x 2.
        # Rows of W_full correspond to k (0..14).
        # Block b uses k = 2b, 2b+1.
        W_full[start_col:start_col+2, 0:2, col_offset:col_offset+7] = W_s
        
        col_offset += 7
    
    # Last block: 2x1 (col 14)
    U_full[:, col_offset:col_offset+4] = U_1
    
    # V_1 is 2x2? No, 2x1 multiplication.
    # V_1 shape (2, 4).
    # Place into V_full[0:2, 14, :]
    V_full[0:2, 14, col_offset:col_offset+4] = V_1
    
    # W_1 shape (2, 4).
    # Place into W_full[14, 0:2, :]
    W_full[14, 0:2, col_offset:col_offset+4] = W_1
    
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
