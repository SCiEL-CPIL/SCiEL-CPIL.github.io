import time
import sys

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 42
    rank = 42
    
    U = [[0] * rank for _ in range(n * m)] # 8 x 42
    V = [[0] * rank for _ in range(m * p)] # 24 x 42
    W = [[0] * rank for _ in range(p * n)] # 12 x 42
    
    # Indices for A (2x4): A[i, j] -> U[i*4+j, col]
    # Indices for B (4x6): B[j, k] -> V[j*6+k, col]
    # Indices for C (2x6): C[i, k] -> W[k*2+i, col] (Trace convention W[k*n+i, r])
    
    # Block definitions
    # A = [A1 A2]
    # A1 is A[0:2, 0:2], A2 is A[0:2, 2:4]
    # B = [B1; B2]
    # B1 is B[0:2, 0:6], B2 is B[2:4, 0:6]
    # B1 = [B11 B12 B13], B2 = [B21 B22 B23]
    # B11 is B[0:2, 0:2], B12 is B[0:2, 2:4], B13 is B[0:2, 4:6]
    # B21 is B[2:4, 0:2], B22 is B[2:4, 2:4], B23 is B[2:4, 4:6]
    
    # C = [C1 C2 C3]
    # C1 is C[0:2, 0:2], C2 is C[0:2, 2:4], C3 is C[0:2, 4:6]
    
    # Strassen coefficients for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = A10(B00+B01)
    # M3 = A00 B10
    # M4 = A11 B11
    # M5 = (A00+A01) B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Output reconstruction
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    def add_strassen(A_offset, B_offset, C_offset, base_col):
        # A_offset: (i, j) offset in A
        # B_offset: (i, j) offset in B
        # C_offset: (i, j) offset in C
        # base_col: starting column index
        
        # A indices: (A_offset[0]+r, A_offset[1]+c)
        # B indices: (B_offset[0]+r, B_offset[1]+c)
        # C indices: (C_offset[0]+r, C_offset[1]+c)
        
        # M1
        col = base_col
        # A00+A11
        U[A_offset[0]*4 + A_offset[1]][col] += 1
        U[(A_offset[0]+1)*4 + (A_offset[1]+1)][col] += 1
        # B00+B11
        V[B_offset[0]*6 + B_offset[1]][col] += 1
        V[(B_offset[0]+1)*6 + (B_offset[1]+1)][col] += 1
        # C00, C11
        W[C_offset[1]*2 + C_offset[0]][col] += 1 # C00
        W[(C_offset[1]+1)*2 + (C_offset[0]+1)][col] += 1 # C11
        
        # M2
        col = base_col + 1
        # A10
        U[(A_offset[0]+1)*4 + A_offset[1]][col] += 1
        # B00+B01
        V[B_offset[0]*6 + B_offset[1]][col] += 1
        V[B_offset[0]*6 + (B_offset[1]+1)][col] += 1
        # C10
        W[C_offset[1]*2 + (C_offset[0]+1)][col] += 1
        
        # M3
        col = base_col + 2
        # A00
        U[A_offset[0]*4 + A_offset[1]][col] += 1
        # B10
        V[(B_offset[0]+1)*6 + B_offset[1]][col] += 1
        # C01, C11
        W[(C_offset[1]+1)*2 + C_offset[0]][col] += 1 # C01
        W[(C_offset[1]+1)*2 + (C_offset[0]+1)][col] += 1 # C11
        
        # M4
        col = base_col + 3
        # A11
        U[(A_offset[0]+1)*4 + (A_offset[1]+1)][col] += 1
        # B11
        V[(B_offset[0]+1)*6 + (B_offset[1]+1)][col] += 1
        # C00, C10
        W[C_offset[1]*2 + C_offset[0]][col] += 1 # C00
        W[C_offset[1]*2 + (C_offset[0]+1)][col] += 1 # C10
        
        # M5
        col = base_col + 4
        # A00+A01
        U[A_offset[0]*4 + A_offset[1]][col] += 1
        U[A_offset[0]*4 + (A_offset[1]+1)][col] += 1
        # B11
        V[(B_offset[0]+1)*6 + (B_offset[1]+1)][col] += 1
        # C00 (-1), C01 (+1)
        W[C_offset[1]*2 + C_offset[0]][col] -= 1
        W[(C_offset[1]+1)*2 + C_offset[0]][col] += 1
        
        # M6
        col = base_col + 5
        # A10-A00
        U[(A_offset[0]+1)*4 + A_offset[1]][col] += 1
        U[A_offset[0]*4 + A_offset[1]][col] -= 1
        # B00+B01
        V[B_offset[0]*6 + B_offset[1]][col] += 1
        V[B_offset[0]*6 + (B_offset[1]+1)][col] += 1
        # C11 (+1)
        W[(C_offset[1]+1)*2 + (C_offset[0]+1)][col] += 1
        
        # M7
        col = base_col + 6
        # A01-A11
        U[A_offset[0]*4 + (A_offset[1]+1)][col] += 1
        U[(A_offset[0]+1)*4 + (A_offset[1]+1)][col] -= 1
        # B10+B11
        V[(B_offset[0]+1)*6 + B_offset[1]][col] += 1
        V[(B_offset[0]+1)*6 + (B_offset[1]+1)][col] += 1
        # C00 (+1)
        W[C_offset[1]*2 + C_offset[0]][col] += 1

    # 3 blocks of output
    # Block 0: C1 (cols 0-1 of C)
    # A1 (cols 0-1 of A), B11 (rows 0-1, cols 0-1 of B)
    add_strassen((0, 0), (0, 0), (0, 0), 0)
    # A2 (cols 2-3 of A), B21 (rows 2-3, cols 0-1 of B)
    add_strassen((0, 2), (2, 0), (0, 0), 7)
    
    # Block 1: C2 (cols 2-3 of C)
    # A1 (cols 0-1 of A), B12 (rows 0-1, cols 2-3 of B)
    add_strassen((0, 0), (0, 2), (0, 2), 14)
    # A2 (cols 2-3 of A), B22 (rows 2-3, cols 2-3 of B)
    add_strassen((0, 2), (2, 2), (0, 2), 21)
    
    # Block 2: C3 (cols 4-5 of C)
    # A1 (cols 0-1 of A), B13 (rows 0-1, cols 4-5 of B)
    add_strassen((0, 0), (0, 4), (0, 4), 28)
    # A2 (cols 2-3 of A), B23 (rows 2-3, cols 4-5 of B)
    add_strassen((0, 2), (2, 4), (0, 4), 35)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
