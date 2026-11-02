"""Strassen-based decomposition for 2x2 * 2x16 matrix multiplication."""
import time

DIMENSIONS = [2, 2, 16]

# Strassen 2x2x2 rank-7 core:
# M1 = (A00+A11)*(B00+B11)
# M2 = (A10+A11)*B00
# M3 = A00*(B01-B11)
# M4 = A11*(B10-B00)
# M5 = (A00+A01)*B11
# M6 = (A10-A00)*(B00+B01)
# M7 = (A01-A11)*(B10+B11)
#
# C00 = M1 + M4 - M5 + M7
# C01 = M3 + M5
# C10 = M2 + M4
# C11 = M1 - M2 + M3 + M6

# For each of the 7 Strassen factors, we need:
# U entries: for each (i,j) pair in A, the coefficient of A[i,j] in each M_k
# V entries: for each (j,k) pair in B, the coefficient of B[j,k] in each M_k
# W entries: for each (k,i) pair in C, the coefficient of M_k in C[k,i]

# A indices: i in {0,1}, j in {0,1} -> row = i*m + j = i*2 + j
# B indices: j in {0,1}, k in {0,15} -> row = j*p + k = j*16 + k
# C indices: k in {0,15}, i in {0,1} -> row = k*n + i = k*2 + i (trace convention W[k*n+i, r])

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 16
    rank = 7 * p  # 112
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 112
    V = [[0] * rank for _ in range(m * p)]  # 32 x 112
    W = [[0] * rank for _ in range(p * n)]  # 32 x 112
    
    # Strassen coefficients for A (i,j) in each of 7 multipliers
    # M1: A00=1, A11=1
    # M2: A10=1, A11=1
    # M3: A00=1
    # M4: A11=1
    # M5: A00=1, A01=1
    # M6: A10=1, A00=-1
    # M7: A01=1, A11=-1
    A_coeff = [
        {0: 1, 3: 1},       # M1: A[0,0]=A00=idx0, A[1,1]=A11=idx3
        {2: 1, 3: 1},       # M2: A[1,0]=A10=idx2, A[1,1]=A11=idx3
        {0: 1},             # M3: A00=idx0
        {3: 1},             # M4: A11=idx3
        {0: 1, 1: 1},       # M5: A00=idx0, A01=idx1
        {2: 1, 0: -1},      # M6: A10=idx2, A00=idx0 (negative)
        {1: 1, 3: -1},      # M7: A01=idx1, A11=idx3 (negative)
    ]
    
    # B coefficients: for each multiplier, which B[j,k] entries and their signs
    # B index: j in {0,1}, k in {0..15}, row = j*16 + k
    # M1: B00+B11 -> B[0,k] and B[1,k] for all k
    # M2: B00 -> B[0,k]
    # M3: B01-B11 -> B[0,k] - B[1,k]
    # M4: B10-B00 -> B[1,k] - B[0,k]
    # M5: B11 -> B[1,k]
    # M6: B00+B01 -> B[0,k] + B[0,k]... wait
    # Let me re-derive. B is 2x16, so B[j][k] where j in {0,1}, k in {0..15}
    # In Strassen for 2x2: B00, B01, B10, B11 are 1x1 blocks.
    # But here B is 2x16, so we split B into 8 blocks of 2x2.
    # For block b (columns 2b, 2b+1): B00_b = B[0,2b], B01_b = B[0,2b+1], B10_b = B[1,2b], B11_b = B[1,2b+1]
    
    # For each block b and each multiplier, the B coefficients:
    # M1: B00_b + B11_b -> B[0,2b] += 1, B[1,2b+1] += 1
    # M2: B00_b -> B[0,2b] += 1
    # M3: B01_b - B11_b -> B[0,2b+1] += 1, B[1,2b+1] -= 1
    # M4: B10_b - B00_b -> B[1,2b] += 1, B[0,2b] -= 1
    # M5: B11_b -> B[1,2b+1] += 1
    # M6: B00_b + B01_b -> B[0,2b] += 1, B[0,2b+1] += 1
    # M7: B10_b + B11_b -> B[1,2b] += 1, B[1,2b+1] += 1
    
    # C coefficients: for each multiplier, which C[k,i] entries
    # C is 2x16, C[i][k] where i in {0,1}, k in {0..15}
    # W index: k*n + i = k*2 + i
    # For block b (output columns 2b, 2b+1):
    # C00_b = C[0,2b] = M1_b + M4_b - M5_b + M7_b
    # C01_b = C[0,2b+1] = M3_b + M5_b
    # C10_b = C[1,2b] = M2_b + M4_b
    # C11_b = C[1,2b+1] = M1_b - M2_b + M3_b + M6_b
    
    # W[k*n + i, col] where col = b*7 + m_idx
    for b in range(8):  # 8 blocks
        for m_idx in range(7):  # 7 multipliers
            col = b * 7 + m_idx
            
            # Fill U: A coefficients (same for all blocks)
            for a_idx, coeff in A_coeff[m_idx].items():
                U[a_idx][col] = coeff
            
            # Fill V: B coefficients for this block
            if m_idx == 0:  # M1: B00 + B11
                V[0 * 16 + 2*b][col] = 1
                V[1 * 16 + (2*b+1)][col] = 1
            elif m_idx == 1:  # M2: B00
                V[0 * 16 + 2*b][col] = 1
            elif m_idx == 2:  # M3: B01 - B11
                V[0 * 16 + (2*b+1)][col] = 1
                V[1 * 16 + (2*b+1)][col] = -1
            elif m_idx == 3:  # M4: B10 - B00
                V[1 * 16 + 2*b][col] = 1
                V[0 * 16 + 2*b][col] = -1
            elif m_idx == 4:  # M5: B11
                V[1 * 16 + (2*b+1)][col] = 1
            elif m_idx == 5:  # M6: B00 + B01
                V[0 * 16 + 2*b][col] = 1
                V[0 * 16 + (2*b+1)][col] = 1
            elif m_idx == 6:  # M7: B10 + B11
                V[1 * 16 + 2*b][col] = 1
                V[1 * 16 + (2*b+1)][col] = 1
            
            # Fill W: C coefficients for this block
            # C00_b = C[0,2b], C01_b = C[0,2b+1], C10_b = C[1,2b], C11_b = C[1,2b+1]
            # W index: k*n + i = k*2 + i
            if m_idx == 0:  # M1: C00 += 1, C11 += 1
                W[(2*b)*2 + 0][col] = 1    # C[0, 2b]
                W[(2*b+1)*2 + 1][col] = 1  # C[1, 2b+1]
            elif m_idx == 1:  # M2: C10 += 1, C11 -= 1
                W[(2*b)*2 + 1][col] = 1    # C[1, 2b]
                W[(2*b+1)*2 + 1][col] = -1 # C[1, 2b+1]
            elif m_idx == 2:  # M3: C01 += 1, C11 += 1
                W[(2*b+1)*2 + 0][col] = 1  # C[0, 2b+1]
                W[(2*b+1)*2 + 1][col] = 1  # C[1, 2b+1]
            elif m_idx == 3:  # M4: C00 += 1, C10 += 1
                W[(2*b)*2 + 0][col] = 1    # C[0, 2b]
                W[(2*b)*2 + 1][col] = 1    # C[1, 2b]
            elif m_idx == 4:  # M5: C00 -= 1, C01 += 1
                W[(2*b)*2 + 0][col] = -1   # C[0, 2b]
                W[(2*b+1)*2 + 0][col] = 1  # C[0, 2b+1]
            elif m_idx == 5:  # M6: C11 += 1
                W[(2*b+1)*2 + 1][col] = 1  # C[1, 2b+1]
            elif m_idx == 6:  # M7: C00 += 1
                W[(2*b)*2 + 0][col] = 1    # C[0, 2b]
    
    return {"U": U, "V": V, "W": W}
