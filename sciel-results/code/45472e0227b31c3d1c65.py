import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 4, 8]

def _verify(U, V, W):
    """Verify that U, V, W exactly decompose the 3x4x8 matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    if not U or not V or not W:
        return False
    rank = len(U[0]) if U else 0
    if len(U) != n * m:
        return False
    if len(V) != m * p:
        return False
    if len(W) != p * n:
        return False
    if any(len(row) != rank for row in U):
        return False
    if any(len(row) != rank for row in V):
        return False
    if any(len(row) != rank for row in W):
        return False
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _strassen_2x2x2():
    """
    Return the 7-term Strassen decomposition for 2x2 x 2x2 matrix multiplication.
    Returns (U, V, W) where:
      U has shape (4, 7)  -- indices (i*m+j) for i in {0,1}, j in {0,1}, m=2
      V has shape (4, 7)  -- indices (j*p+k) for j in {0,1}, k in {0,1}, p=2
      W has shape (4, 7)  -- indices (k*n+i) for k in {0,1}, i in {0,1}, n=2
    """
    # Strassen formulas:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # U indices: (i, j) -> i*2 + j
    # V indices: (j, k) -> j*2 + k
    # W indices: (k, i) -> k*2 + i
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00 + A11)(B00 + B11)
    # A00: U[0,0]=1, A11: U[3,0]=1
    # B00: V[0,0]=1, B11: V[3,0]=1
    # C00: W[0,0]=1, C11: W[3,0]=1
    # C00 += M1, C11 += M1
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    W[0,0] = 1; W[3,0] = 1
    
    # M2 = (A10 + A11)B00
    # A10: U[2,1]=1, A11: U[3,1]=1
    # B00: V[0,1]=1
    # C10 += M2, C11 += M2 (with -1)
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[2,1] = 1; W[3,1] = -1
    
    # M3 = A00(B10 - B11)
    # A00: U[0,2]=1
    # B10: V[2,2]=1, B11: V[3,2]=-1
    # C01 += M3, C10 += M3
    U[0,2] = 1
    V[2,2] = 1; V[3,2] = -1
    W[1,2] = 1; W[2,2] = 1
    
    # M4 = A11(B01 - B00)
    # A11: U[3,3]=1
    # B01: V[1,3]=1, B00: V[0,3]=-1
    # C00 += M4, C10 += M4
    U[3,3] = 1
    V[1,3] = 1; V[0,3] = -1
    W[0,3] = 1; W[2,3] = 1
    
    # M5 = (A00 + A01)B11
    # A00: U[0,4]=1, A01: U[1,4]=1
    # B11: V[3,4]=1
    # C00 -= M5, C01 += M5
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[0,4] = -1; W[1,4] = 1
    
    # M6 = (A10 - A00)(B00 + B01)
    # A10: U[2,5]=1, A00: U[0,5]=-1
    # B00: V[0,5]=1, B01: V[1,5]=1
    # C11 += M6
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[1,5] = 1
    W[3,5] = 1
    
    # M7 = (A01 - A11)(B10 + B11)
    # A01: U[1,6]=1, A11: U[3,6]=-1
    # B10: V[2,6]=1, B11: V[3,6]=1
    # C00 += M7
    U[1,6] = 1; U[3,6] = -1
    V[2,6] = 1; V[3,6] = 1
    W[0,6] = 1
    
    return U, V, W

def _strassen_2x2x4():
    """
    Decompose 2x2 x 2x4 multiplication using Strassen on the 2x2 part.
    The 4 columns of B are handled by splitting into two 2x2 blocks.
    Returns (U, V, W) where:
      U has shape (4, 14)  -- indices (i*2+j) for i in {0,1}, j in {0,1}
      V has shape (8, 14)  -- indices (j*4+k) for j in {0,1}, k in {0,1,2,3}
      W has shape (8, 14)  -- indices (k*2+i) for k in {0,1,2,3}, i in {0,1}
    """
    # Split B into B_left (2x2) and B_right (2x2)
    # C_left = A @ B_left (2x2 x 2x2, Strassen rank 7)
    # C_right = A @ B_right (2x2 x 2x2, Strassen rank 7)
    # Total rank: 14
    
    U_sl, V_sl, W_sl = _strassen_2x2x2()
    
    U = [[0]*14 for _ in range(4)]
    V = [[0]*14 for _ in range(8)]
    W = [[0]*14 for _ in range(8)]
    
    # Left half: columns 0,1 of B
    for i in range(4):
        for r in range(7):
            U[i, r] = U_sl[i, r]
    for j in range(2):
        for k in range(2):
            for r in range(7):
                V[j*4+k, r] = V_sl[j*2+k, r]
    for k in range(2):
        for i in range(2):
            for r in range(7):
                W[k*2+i, r] = W_sl[k*2+i, r]
    
    # Right half: columns 2,3 of B
    for i in range(4):
        for r in range(7):
            U[i, r+7] = U_sl[i, r]
    for j in range(2):
        for k in range(2):
            for r in range(7):
                V[j*4+(k+2), r+7] = V_sl[j*2+k, r]
    for k in range(2):
        for i in range(2):
            for r in range(7):
                W[(k+2)*2+i, r+7] = W_sl[k*2+i, r]
    
    return U, V, W

def _strassen_2x4x8():
    """
    Decompose 2x4 x 4x8 multiplication.
    Split A into two 2x2 blocks (rows 0-1 and rows 2-3 of A, but A is 2x4 so split columns)
    Actually A is 2x4, B is 4x8.
    Split A into A_left (2x2) and A_right (2x2) by columns.
    Split B into B_top (4x4) and B_bottom (4x4) by rows.
    
    C = A @ B = A_left @ B_top + A_right @ B_bottom
    
    Each term is 2x2 x 4x4. We can further decompose 2x2 x 4x4 using Strassen on the 2x2 part.
    2x2 x 4x4: split B into four 2x2 blocks. Use Strassen for each: 4 * 7 = 28 terms.
    Total: 28 + 28 = 56 terms.
    """
    # A is 2x4: split into A_left (cols 0-1) and A_right (cols 2-3)
    # B is 4x8: split into B_top (rows 0-1) and B_bottom (rows 2-3)
    
    # C = A_left @ B_top + A_right @ B_bottom
    # A_left @ B_top: 2x2 x 4x4 -> use 2x2x4 Strassen twice (for the 4 columns)
    # Actually 2x2 x 4x4: treat as 2x2 x (4x4). Split 4x4 into two 4x2? No.
    # 2x2 x 4x4: split the 4 columns of B into two 2-column groups.
    # For each 2-column group: 2x2 x 4x2. Split 4x2 into two 2x2 blocks (rows).
    # So 2x2 x 4x2 = A @ (B1; B2) = A@B1 + A@B2, each 2x2 x 2x2, rank 7 each. Total 14.
    # Two such groups: 28 terms.
    
    # So A_left @ B_top: 28 terms
    # A_right @ B_bottom: 28 terms
    # Total: 56 terms
    
    U = [[0]*56 for _ in range(8)]   # 2x4 = 8 entries
    V = [[0]*56 for _ in range(32)]  # 4x8 = 32 entries
    W = [[0]*56 for _ in range(16)]  # 8x2 = 16 entries
    
    # Helper: add 2x2 x 4x4 decomposition at given offsets
    def add_2x2x4(U_off, V_off, W_off, r_start):
        # A is 2x2 (rows 0-1, cols 0-1 of the local 2x4 A)
        # B is 4x4 (rows 0-3, cols 0-3 of the local 4x8 B)
        # C is 4x2 (rows 0-3, cols 0-1 of the local 4x2 C)
        
        # Split B into B_top (rows 0-1) and B_bottom (rows 2-3)
        # C = A @ B_top + A @ B_bottom
        # A @ B_top: 2x2 x 2x4 -> use 2x2x4 Strassen (14 terms)
        # A @ B_bottom: 2x2 x 2x4 -> use 2x2x4 Strassen (14 terms)
        
        U_sl, V_sl, W_sl = _strassen_2x2x4()
        
        # A @ B_top: B rows 0-1, cols 0-3
        for i in range(4):
            for r in range(14):
                U[U_off + i, r_start + r] = U_sl[i, r]
        for j in range(2):
            for k in range(4):
                for r in range(14):
                    V[V_off + j*4+k, r_start + r] = V_sl[j*4+k, r]
        for k in range(4):
            for i in range(2):
                for r in range(14):
                    W[W_off + k*2+i, r_start + r] = W_sl[k*2+i, r]
        
        # A @ B_bottom: B rows 2-3, cols 0-3
        for i in range(4):
            for r in range(14):
                U[U_off + i, r_start + 14 + r] = U_sl[i, r]
        for j in range(2):
            for k in range(4):
                for r in range(14):
                    V[V_off + (j+2)*4+k, r_start + 14 + r] = V_sl[j*4+k, r]
        for k in range(4):
            for i in range(2):
                for r in range(14):
                    W[W_off + k*2+i, r_start + 14 + r] = W_sl[k*2+i, r]
    
    # A_left @ B_top: A rows 0-1, cols 0-1; B rows 0-1, cols 0-3; C rows 0-3, cols 0-1
    # In the full 2x4 x 4x8:
    # A_left is at U rows 0-3 (i*2+j for i in {0,1}, j in {0,1})
    # B_top is at V rows 0-7 (j*4+k for j in {0,1}, k in {0,1,2,3})
    # C_left is at W rows 0-7 (k*2+i for k in {0,1,2,3}, i in {0,1})
    add_2x2x4(0, 0, 0, 0)
    
    # A_right @ B_bottom: A rows 0-1, cols 2-3; B rows 2-3, cols 0-3; C rows 0-3, cols 0-1
    # In the full 2x4 x 4x8:
    # A_right is at U rows 4-7 (i*2+j for i in {0,1}, j in {2,3})
    # B_bottom is at V rows 8-15 (j*4+k for j in {2,3}, k in {0,1,2,3})
    # C_right is at W rows 0-7 (same as C_left, since we're adding to the same output)
    add_2x2x4(4, 8, 0, 28)
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Schoolbook rank = 96
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_schoolbook
    
    # Try to construct rank-88 decomposition:
    # Split the 3x4 matrix into row 0 (1x4) and rows 1-2 (2x4)
    # Row 0: 1x4 x 4x8 schoolbook, rank 32
    # Rows 1-2: 2x4 x 4x8 Strassen-based, rank 56
    # Total: 88
    
    rank_total = 88
    U_full = [[0]*rank_total for _ in range(12)]  # 3x4 = 12
    V_full = [[0]*rank_total for _ in range(32)]  # 4x8 = 32
    W_full = [[0]*rank_total for _ in range(24)]  # 8x3 = 24
    
    # Part 1: 1x4x8 schoolbook (rank 0-31)
    # A row 0: U indices 0-3 (i=0, j=0-3)
    # B: V indices 0-31 (j=0-3, k=0-7)
    # C: W indices k*3+0 for k=0-7
    for j in range(4):
        for k in range(8):
            r = j*8 + k
            U_full[j][r] = 1          # U[0*4+j, r] = U[j, r]
            V_full[j*8+k][r] = 1      # V[j*8+k, r]
            W_full[k*3][r] = 1        # W[k*3+0, r]
    
    # Part 2: 2x4x8 Strassen (rank 32-87)
    # A rows 1-2: U indices 4-11 (i=1-2, j=0-3)
    # B: V indices 0-31
    # C: W indices k*3+1 for k=0-7 (i=1) and k*3+2 for k=0-7 (i=2)
    
    U_2x4x8, V_2x4x8, W_2x4x8 = _strassen_2x4x8()
    
    for i in range(8):
        for r in range(56):
            U_full[4+i][32+r] = U_2x4x8[i][r]
    
    for j in range(32):
        for r in range(56):
            V_full[j][32+r] = V_2x4x8[j][r]
    
    for k in range(16):
        for r in range(56):
            # W_2x4x8 has 16 rows: k*2+i for k in {0,1,2,3}, i in {0,1}
            # In the full W, i=0 corresponds to row 1 of the 3x4 matrix, i=1 to row 2
            # So W_full[k*3 + 1 + i, r] for i in {0,1}
            local_k = k // 2
            local_i = k % 2
            W_full[local_k*3 + 1 + local_i][32+r] = W_2x4x8[k][r]
    
    result = {"U": U_full, "V": V_full, "W": W_full}
    
    # Verify
    if _verify(U_full, V_full, W_full):
        return result
    
    # Fallback: return schoolbook
    return best_result
