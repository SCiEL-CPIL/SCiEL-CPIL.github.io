import sys
import time

def construct():
    """
    Rank-50 tensor decomposition for 2x4 by 4x7 matrix multiplication.
    
    Uses Strassen (rank 7) for 2x2x2 blocks and Schoolbook (rank 4) for 2x2x1 blocks.
    Trace convention: W[k*n+i, r]
    """
    n, m, p = 2, 4, 7
    rank = 50
    
    U = [[0] * rank for _ in range(n * m)]  # shape (8, 50)
    V = [[0] * rank for _ in range(m * p)]  # shape (28, 50)
    W = [[0] * rank for _ in range(p * n)]  # shape (14, 50)
    
    col = 0
    
    def add_strassen_block(A_idx, B_idx, C_idx):
        """Add a Strassen 2x2x2 decomposition (rank 7)."""
        nonlocal col
        # Strassen's 7 multiplications for 2x2 * 2x2 -> 2x2
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b01-b11)
        # M4 = a11(b10-b00)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        def proc(A_c, B_c, C_c):
            nonlocal col
            for la, c in A_c.items():
                ga = A_idx[la]
                U[ga[0] * m + ga[1]][col] += c
            for lb, c in B_c.items():
                gb = B_idx[lb]
                V[gb[0] * p + gb[1]][col] += c
            for lc, c in C_c.items():
                gc = C_idx[lc]
                W[gc[1] * n + gc[0]][col] += c
            col += 1

        # M1
        proc({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1})
        # M2
        proc({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1})
        # M3
        proc({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1})
        # M4
        proc({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1})
        # M5
        proc({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1})
        # M6
        proc({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1})
        # M7
        proc({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})

    def add_schoolbook_221(A_idx, B_idx, C_idx):
        """Add a schoolbook 2x2 * 2x1 -> 2x1 decomposition (rank 4)."""
        nonlocal col
        # c0 = a00*b0 + a01*b1
        # c1 = a10*b0 + a11*b1
        def pm(ia, ib, ic, c=1):
            nonlocal col
            ga = A_idx[ia]
            gb = B_idx[ib]
            gc = C_idx[ic]
            
            U[ga[0] * m + ga[1]][col] += c
            V[gb[0] * p + gb[1]][col] += c
            W[gc[1] * n + gc[0]][col] += c
            col += 1

        pm(0, 0, 0, 1)
        pm(1, 1, 0, 1)
        pm(2, 0, 1, 1)
        pm(3, 1, 1, 1)

    # A split: A_L (cols 0,1), A_R (cols 2,3)
    A_L = [(0, 0), (0, 1), (1, 0), (1, 1)]
    A_R = [(0, 2), (0, 3), (1, 2), (1, 3)]
    
    # Block 1: B cols 0,1 (4x2)
    B_1_L = [(0, 0), (0, 1), (1, 0), (1, 1)]
    B_1_R = [(2, 0), (2, 1), (3, 0), (3, 1)]
    C_1 = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    add_strassen_block(A_L, B_1_L, C_1)
    add_strassen_block(A_R, B_1_R, C_1)
    
    # Block 2: B cols 2,3 (4x2)
    B_2_L = [(0, 2), (0, 3), (1, 2), (1, 3)]
    B_2_R = [(2, 2), (2, 3), (3, 2), (3, 3)]
    C_2 = [(0, 2), (0, 3), (1, 2), (1, 3)]
    
    add_strassen_block(A_L, B_2_L, C_2)
    add_strassen_block(A_R, B_2_R, C_2)
    
    # Block 3: B cols 4,5,6 (4x3)
    # First 2 cols (4x2): Strassen
    B_3_L_L = [(0, 4), (0, 5), (1, 4), (1, 5)]
    B_3_L_R = [(2, 4), (2, 5), (3, 4), (3, 5)]
    C_3_L = [(0, 4), (0, 5), (1, 4), (1, 5)]
    
    add_strassen_block(A_L, B_3_L_L, C_3_L)
    add_strassen_block(A_R, B_3_L_R, C_3_L)
    
    # Last col (4x1): Schoolbook
    B_4_L = [(0, 6), (1, 6)]
    B_4_R = [(2, 6), (3, 6)]
    C_4 = [(0, 6), (1, 6)]
    
    add_schoolbook_221(A_L, B_4_L, C_4)
    add_schoolbook_221(A_R, B_4_R, C_4)
    
    return {"U": U, "V": V, "W": W}
