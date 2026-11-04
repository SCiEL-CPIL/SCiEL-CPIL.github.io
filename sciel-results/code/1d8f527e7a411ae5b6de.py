import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 8]
DEADLINE = time.time() + 120

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    nm, mp, pn = n * m, m * p, p * n  # 12, 32, 24

    # Strassen 2x2x2 decomposition
    # M_k = sum_{i,j} U[i,j,k] A[i,j] * sum_{j',k'} V[j',k',k] B[j',k']
    # C[i,k] = sum_{j} A[i,j] B[j,k] = sum_k W[k,i,k'] M_{k'}
    
    # Indices for 2x2x2:
    # A is 2x2, B is 2x2, C is 2x2.
    # U shape (4, 7), V shape (4, 7), W shape (4, 7)
    
    # Strassen formulas:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    U_222 = np.zeros((4, 7), dtype=int)
    V_222 = np.zeros((4, 7), dtype=int)
    W_222 = np.zeros((4, 7), dtype=int)
    
    # Fill U (coefficients of A in M_k)
    # A indices: 00->0, 01->1, 10->2, 11->3
    # M1: A00(1), A11(1)
    U_222[0, 0] = 1; U_222[3, 0] = 1
    # M2: A10(1), A11(1)
    U_222[2, 1] = 1; U_222[3, 1] = 1
    # M3: A00(1)
    U_222[0, 2] = 1
    # M4: A11(1)
    U_222[3, 3] = 1
    # M5: A00(1), A01(1)
    U_222[0, 4] = 1; U_222[1, 4] = 1
    # M6: A10(1), A00(-1)
    U_222[2, 5] = 1; U_222[0, 5] = -1
    # M7: A01(1), A11(-1)
    U_222[1, 6] = 1; U_222[3, 6] = -1
    
    # Fill V (coefficients of B in M_k)
    # B indices: 00->0, 01->1, 10->2, 11->3
    # M1: B00(1), B11(1)
    V_222[0, 0] = 1; V_222[3, 0] = 1
    # M2: B00(1)
    V_222[0, 1] = 1
    # M3: B01(1), B11(-1)
    V_222[1, 2] = 1; V_222[3, 2] = -1
    # M4: B10(1), B00(-1)
    V_222[2, 3] = 1; V_222[0, 3] = -1
    # M5: B11(1)
    V_222[3, 4] = 1
    # M6: B00(1), B01(1)
    V_222[0, 5] = 1; V_222[1, 5] = 1
    # M7: B10(1), B11(1)
    V_222[2, 6] = 1; V_222[3, 6] = 1
    
    # Fill W (coefficients of M_k in C_ij)
    # C indices: 00->0, 01->1, 10->2, 11->3
    # C00: M1(1), M4(1), M5(-1), M7(1)
    W_222[0, 0] = 1; W_222[0, 3] = 1; W_222[0, 4] = -1; W_222[0, 6] = 1
    # C01: M3(1), M5(1)
    W_222[1, 2] = 1; W_222[1, 4] = 1
    # C10: M2(1), M4(1)
    W_222[2, 1] = 1; W_222[2, 3] = 1
    # C11: M1(1), M2(-1), M3(1), M6(1)
    W_222[3, 0] = 1; W_222[3, 1] = -1; W_222[3, 2] = 1; W_222[3, 5] = 1

    # Helper to build larger decompositions
    def build_2x2x4():
        # M_{2,2,4}: A(2x2), B(2x4), C(2x4)
        # B = [B1, B2] where B1, B2 are 2x2.
        # C = [A B1, A B2].
        # Rank 14.
        # U: (4, 14), V: (8, 14), W: (8, 14)
        # A indices: 0-3
        # B indices: 0-7 (0-3 for B1, 4-7 for B2)
        # C indices: 0-7 (0-3 for C1, 4-7 for C2)
        
        U_224 = np.zeros((4, 14), dtype=int)
        V_224 = np.zeros((8, 14), dtype=int)
        W_224 = np.zeros((8, 14), dtype=int)
        
        # First 7 cols for B1
        U_224[:, 0:7] = U_222
        V_224[0:4, 0:7] = V_222
        W_224[0:4, 0:7] = W_222
        
        # Next 7 cols for B2
        U_224[:, 7:14] = U_222
        V_224[4:8, 7:14] = V_222
        W_224[4:8, 7:14] = W_222
        
        return U_224, V_224, W_224

    def build_2x2x8():
        # M_{2,2,8}: A(2x2), B(2x8), C(2x8)
        # B = [B1, B2, B3, B4] 2x2 each.
        # Rank 28.
        U_228 = np.zeros((4, 28), dtype=int)
        V_228 = np.zeros((16, 28), dtype=int)
        W_228 = np.zeros((16, 28), dtype=int)
        
        for b_idx in range(4):
            U_228[:, b_idx*7:(b_idx+1)*7] = U_222
            V_228[b_idx*4:(b_idx+1)*4, b_idx*7:(b_idx+1)*7] = V_222
            W_228[b_idx*4:(b_idx+1)*4, b_idx*7:(b_idx+1)*7] = W_222
            
        return U_228, V_228, W_228

    def build_2x4x8():
        # M_{2,4,8}: A(2x4), B(4x8), C(2x8)
        # A = [A1, A2] 2x2 each.
        # B = [B1; B2] 2x8 each.
        # C = A1 B1 + A2 B2.
        # Rank 28 + 28 = 56.
        U_248 = np.zeros((8, 56), dtype=int)
        V_248 = np.zeros((32, 56), dtype=int)
        W_248 = np.zeros((16, 56), dtype=int)
        
        # Part 1: A1 B1 (A1 is 2x2, B1 is 2x8)
        U_228_1, V_228_1, W_228_1 = build_2x2x8()
        U_248[0:4, 0:28] = U_228_1
        V_248[0:16, 0:28] = V_228_1
        W_248[:, 0:28] = W_228_1
        
        # Part 2: A2 B2 (A2 is 2x2, B2 is 2x8)
        U_228_2, V_228_2, W_228_2 = build_2x2x8()
        U_248[4:8, 28:56] = U_228_2
        V_248[16:32, 28:56] = V_228_2
        W_248[:, 28:56] = W_228_2
        
        return U_248, V_248, W_248

    def build_1x4x8():
        # M_{1,4,8}: A(1x4), B(4x8), C(1x8)
        # Rank 4.
        # C_k = sum_j A_j B_{j,k}
        # 4 products.
        U_148 = np.zeros((4, 4), dtype=int)
        V_148 = np.zeros((32, 4), dtype=int)
        W_148 = np.zeros((8, 4), dtype=int)
        
        # A indices: 0-3
        # B indices: 0-31 (4x8)
        # C indices: 0-7
        
        for j in range(4):
            # Product j: A_j * B_{j,:}
            U_148[j, j] = 1
            # V: row j of B is indices j*8 to j*8+7
            V_148[j*8:(j+1)*8, j] = 1
            # W: all C_k get this product
            W_148[:, j] = 1
            
        return U_148, V_148, W_148

    # Assemble 3x4x8
    # A(3x4) = [A_top (2x4); A_bottom (1x4)]
    # B(4x8)
    # C(3x8) = [C_top (2x8); C_bottom (1x8)]
    # C_top = A_top B (rank 56)
    # C_bottom = A_bottom B (rank 4)
    # Total rank 60.
    
    U_348 = np.zeros((12, 60), dtype=int)
    V_348 = np.zeros((32, 60), dtype=int)
    W_348 = np.zeros((24, 60), dtype=int)
    
    # Top part: 2x4x8
    U_248, V_248, W_248 = build_2x4x8()
    U_348[0:8, 0:56] = U_248
    V_348[:, 0:56] = V_248
    W_348[0:16, 0:56] = W_248
    
    # Bottom part: 1x4x8
    U_148, V_148, W_148 = build_1x4x8()
    U_348[8:12, 56:60] = U_148
    V_348[:, 56:60] = V_148
    W_348[16:24, 56:60] = W_148
    
    # Convert to lists of floats (or ints)
    # The task asks for integer/half-integer entries.
    # Our entries are integers.
    
    U_out = [[int(x) for x in row] for row in U_348]
    V_out = [[int(x) for x in row] for row in V_348]
    W_out = [[int(x) for x in row] for row in W_348]
    
    return {"U": U_out, "V": V_out, "W": W_out}

