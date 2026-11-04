import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # We will use a Strassen-based decomposition.
    # Split the 2x4 * 4x8 into 4 sub-problems of size 2x2 * 2x4? No.
    # Let's split the middle dimension (4) into two halves of size 2.
    # A is 2x4. B is 4x8.
    # A = [A1, A2] where A1 is 2x2, A2 is 2x2.
    # B = [B1; B2] where B1 is 2x8, B2 is 2x8.
    # C = A1*B1 + A2*B2.
    
    # Now we have two terms: (2x2 * 2x8) and (2x2 * 2x8).
    # For each term, we can further decompose.
    # 2x2 * 2x8. Let's split the 8 columns of B into four blocks of 2x2?
    # Or use Strassen on the 2x2 part?
    # Standard Strassen is for 2x2 * 2x2.
    # If we have 2x2 * 2x8, we can view the 2x8 as four 2x2 blocks.
    # So we have 4 independent 2x2 * 2x2 multiplications.
    # Each 2x2 * 2x2 has rank 7 via Strassen.
    # Total rank = 2 (terms from first split) * 4 (blocks) * 7 (Strassen rank) = 56.
    
    # Let's verify the indices.
    # A is n x m = 2 x 4.
    # B is m x p = 4 x 8.
    # C is n x p = 2 x 8.
    
    # Split 1: m=4 into m1=2, m2=2.
    # C = A[:, 0:2] @ B[0:2, :] + A[:, 2:4] @ B[2:4, :]
    
    # Term 1: A1 (2x2) @ B1 (2x8)
    # B1 has 8 columns. Split B1 into 4 blocks of 2 columns: B1[:, 0:2], B1[:, 2:4], B1[:, 4:6], B1[:, 6:8].
    # Each is 2x2.
    # So Term 1 = sum_{t=0}^3 A1 @ B1[:, 2t:2t+2]
    
    # Term 2: A2 (2x2) @ B2 (2x8)
    # Similarly split B2 into 4 blocks.
    # Term 2 = sum_{t=0}^3 A2 @ B2[:, 2t:2t+2]
    
    # Total 8 blocks of 2x2 * 2x2.
    # Each block uses Strassen with 7 multiplications.
    # Total rank = 8 * 7 = 56.
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56
    
    # Strassen coefficients for 2x2 * 2x2:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Strassen U factors (A side):
    # M1: A00, A11 -> 1, 1
    # M2: A10, A11 -> 1, 1
    # M3: A00 -> 1
    # M4: A11 -> 1
    # M5: A00, A01 -> 1, 1
    # M6: A10, A00 -> 1, -1
    # M7: A01, A11 -> 1, -1
    
    # Strassen V factors (B side):
    # M1: B00, B11 -> 1, 1
    # M2: B00 -> 1
    # M3: B01, B11 -> 1, -1
    # M4: B10, B00 -> 1, -1
    # M5: B11 -> 1
    # M6: B00, B01 -> 1, 1
    # M7: B10, B11 -> 1, 1
    
    # Strassen W factors (C side):
    # M1: C00, C11 -> 1, 1
    # M2: C10, C11 -> 1, -1
    # M3: C01, C10 -> 1, 1
    # M4: C00, C10 -> 1, 1
    # M5: C00, C01 -> -1, 1
    # M6: C11 -> 1
    # M7: C00, C11 -> 1, 1
    
    # Let's map these to the global indices.
    # For a block (s, t):
    # s in {0, 1} corresponds to A[:, 2s:2s+2] and B[2s:2s+2, :]
    # t in {0, 1, 2, 3} corresponds to B[:, 2t:2t+2]
    
    # Local indices in 2x2 block:
    # i_loc in {0, 1}, j_loc in {0, 1}, k_loc in {0, 1}
    
    # Global indices:
    # i = i_loc
    # j = 2s + j_loc
    # k = 2t + k_loc
    
    # U index: i * m + j = i_loc * 4 + 2s + j_loc
    # V index: j * p + k = (2s + j_loc) * 8 + 2t + k_loc
    # W index: k * n + i = (2t + k_loc) * 2 + i_loc
    
    strassen_U = [
        [1, 1],  # M1: A00, A11
        [1, 1],  # M2: A10, A11
        [1, 0],  # M3: A00
        [0, 1],  # M4: A11
        [1, 1],  # M5: A00, A01
        [1, -1], # M6: A10, A00
        [1, -1]  # M7: A01, A11
    ]
    # Note: The U factor is a vector of coefficients for the A elements involved.
    # For M1, A00 and A11 are added. So U[i0, col] += 1, U[i1, col] += 1.
    # Wait, the standard Strassen U matrix for 2x2 is 4x7.
    # Row i corresponds to A element.
    # M1 uses A00, A11. So U[0, 0]=1, U[3, 0]=1. (Using 0,1,2,3 for 00,01,10,11)
    # Let's be precise.
    
    # Local A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # Local B indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # Local C indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # U coefficients (rows 0,1,2,3 for A00,A01,A10,A11):
    # M1: [1, 0, 0, 1]
    # M2: [0, 0, 1, 1]
    # M3: [1, 0, 0, 0]
    # M4: [0, 0, 0, 1]
    # M5: [1, 1, 0, 0]
    # M6: [1, 0, 1, 0] with signs? No, (A10-A00). So A10: 1, A00: -1. -> [-1, 0, 1, 0]
    # M7: (A01-A11). So A01: 1, A11: -1. -> [0, 1, 0, -1]
    
    U_strassen = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    # Transposed? No, U is (n*m) x rank. Here local A is 2x2.
    # We need to map local A index (0,1,2,3) to global U index.
    
    # V coefficients (rows 0,1,2,3 for B00,B01,B10,B11):
    # M1: (B00+B11) -> [1, 0, 0, 1]
    # M2: B00 -> [1, 0, 0, 0]
    # M3: (B01-B11) -> [0, 1, 0, -1]
    # M4: (B10-B00) -> [-1, 0, 1, 0]
    # M5: B11 -> [0, 0, 0, 1]
    # M6: (B00+B01) -> [1, 1, 0, 0]
    # M7: (B10+B11) -> [0, 0, 1, 1]
    
    V_strassen = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    # W coefficients (rows 0,1,2,3 for C00,C01,C10,C11):
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Columns are M1..M7. Rows are C00, C01, C10, C11.
    W_strassen = [
        [1, 0, 0, 1, -1, 0, 1],   # C00
        [0, 0, 1, 0, 1, 0, 0],   # C01
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [1, -1, 1, 0, 0, 1, 0]   # C11
    ]
    
    # Now construct the full matrices.
    col = 0
    for s in range(2):  # 0, 1
        for t in range(4):  # 0, 1, 2, 3
            for r in range(7):
                # U contributions
                for i_loc in range(2):
                    for j_loc in range(2):
                        # Local A index: i_loc * 2 + j_loc
                        idx_local_A = i_loc * 2 + j_loc
                        coeff = U_strassen[r][idx_local_A]
                        if coeff != 0:
                            idx_global_i = i_loc
                            idx_global_j = 2 * s + j_loc
                            idx_U = idx_global_i * 4 + idx_global_j
                            U[idx_U][col] += coeff
                
                # V contributions
                for j_loc in range(2):
                    for k_loc in range(2):
                        idx_local_B = j_loc * 2 + k_loc
                        coeff = V_strassen[r][idx_local_B]
                        if coeff != 0:
                            idx_global_j = 2 * s + j_loc
                            idx_global_k = 2 * t + k_loc
                            idx_V = idx_global_j * 8 + idx_global_k
                            V[idx_V][col] += coeff
                
                # W contributions
                for k_loc in range(2):
                    for i_loc in range(2):
                        idx_local_C = k_loc * 2 + i_loc
                        coeff = W_strassen[r][idx_local_C]
                        if coeff != 0:
                            idx_global_k = 2 * t + k_loc
                            idx_global_i = i_loc
                            idx_W = idx_global_k * 2 + idx_global_i
                            W[idx_W][col] += coeff
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
