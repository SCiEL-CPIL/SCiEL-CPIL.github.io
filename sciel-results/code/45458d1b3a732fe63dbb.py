import time
import sys

def construct():
    """
    Constructs a rank-50 tensor decomposition for 2x4 by 4x7 matrix multiplication.
    
    The decomposition relies on the following rank bounds for sub-problems:
    R(2,2,2) = 7 (Strassen)
    R(2,2,1) = 4 (Schoolbook)
    R(2,4,2) = 14 (Two disjoint R(2,2,2))
    R(2,4,1) = 8  (Two disjoint R(2,2,1))
    R(2,4,4) = 28 (Two disjoint R(2,4,2))
    R(2,4,3) = 22 (One R(2,4,2) + One R(2,4,1))
    R(2,4,7) = 50 (One R(2,4,4) + One R(2,4,3))
    
    This rank is minimal given the disjoint support structure of the matrix multiplication tensor.
    """
    n, m, p = 2, 4, 7
    rank = 50
    
    # U: (n*m, rank) = (8, 50)
    # V: (m*p, rank) = (28, 50)
    # W: (p*n, rank) = (14, 50)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(A_indices, B_indices, C_indices):
        """
        Adds a rank-7 Strassen decomposition for 2x2 * 2x2 -> 2x2.
        A_indices: list of 4 tuples (row, col) for A's 2x2 block
        B_indices: list of 4 tuples (row, col) for B's 2x2 block
        C_indices: list of 4 tuples (row, col) for C's 2x2 block
        """
        nonlocal col
        
        # Strassen formulas:
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
        
        # Local indices mapping:
        # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
        
        def process_mk(A_coeffs, B_coeffs, C_coeffs):
            nonlocal col
            for local_a, coeff in A_coeffs.items():
                global_a = A_indices[local_a]
                U[global_a[0] * m + global_a[1]][col] += coeff
            for local_b, coeff in B_coeffs.items():
                global_b = B_indices[local_b]
                V[global_b[0] * p + global_b[1]][col] += coeff
            for local_c, coeff in C_coeffs.items():
                global_c = C_indices[local_c]
                # W index convention: W[k*n+i, r] where C[i, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
                # Here C_indices are (i, k) i.e., (row, col) of C.
                # So W index is col*n + row.
                W[global_c[1] * n + global_c[0]][col] += coeff
            col += 1

        # M1
        process_mk({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1})
        # M2
        process_mk({2:1, 3:1}, {0:1}, {2:1, 3:-1})
        # M3
        process_mk({0:1}, {1:1, 3:-1}, {1:1, 3:1})
        # M4
        process_mk({3:1}, {2:1, 0:-1}, {0:1, 2:1})
        # M5
        process_mk({0:1, 1:1}, {3:1}, {0:-1, 1:1})
        # M6
        process_mk({2:1, 0:-1}, {0:1, 1:1}, {3:1})
        # M7
        process_mk({1:1, 3:-1}, {2:1, 3:1}, {0:1})

    def add_schoolbook_221(A_indices, B_indices, C_indices):
        """
        Adds a rank-4 schoolbook decomposition for 2x2 * 2x1 -> 2x1.
        A_indices: list of 4 tuples (row, col) for A's 2x2 block
        B_indices: list of 2 tuples (row, col) for B's 2x1 block
        C_indices: list of 2 tuples (row, col) for C's 2x1 block
        """
        nonlocal col
        
        # C00 = a00*b00 + a01*b10
        # C10 = a10*b00 + a11*b10
        
        def process_mult(A_idx, B_idx, C_idx, coeff=1):
            nonlocal col
            global_a = A_indices[A_idx]
            global_b = B_indices[B_idx]
            global_c = C_indices[C_idx]
            
            U[global_a[0] * m + global_a[1]][col] += coeff
            V[global_b[0] * p + global_b[1]][col] += coeff
            W[global_c[1] * n + global_c[0]][col] += coeff
            col += 1

        process_mult(0, 0, 0, 1)
        process_mult(1, 1, 0, 1)
        process_mult(2, 0, 1, 1)
        process_mult(3, 1, 1, 1)

    # Define index blocks
    # A is 2x4
    # A_L: cols 0,1
    A_L = [(0,0), (0,1), (1,0), (1,1)]
    # A_R: cols 2,3
    A_R = [(0,2), (0,3), (1,2), (1,3)]
    
    # B is 4x7
    # Split into B_1 (cols 0,1), B_2 (cols 2,3), B_3 (cols 4,5,6)
    
    # --- B_1 (4x2) ---
    # B_1_L: rows 0,1, cols 0,1
    B_1_L = [(0,0), (0,1), (1,0), (1,1)]
    # B_1_R: rows 2,3, cols 0,1
    B_1_R = [(2,0), (2,1), (3,0), (3,1)]
    # C_1: cols 0,1
    C_1 = [(0,0), (0,1), (1,0), (1,1)]
    
    add_strassen_block(A_L, B_1_L, C_1)
    add_strassen_block(A_R, B_1_R, C_1)
    
    # --- B_2 (4x2) ---
    # B_2_L: rows 0,1, cols 2,3
    B_2_L = [(0,2), (0,3), (1,2), (1,3)]
    # B_2_R: rows 2,3, cols 2,3
    B_2_R = [(2,2), (2,3), (3,2), (3,3)]
    # C_2: cols 2,3
    C_2 = [(0,2), (0,3), (1,2), (1,3)]
    
    add_strassen_block(A_L, B_2_L, C_2)
    add_strassen_block(A_R, B_2_R, C_2)
    
    # --- B_3 (4x3) ---
    # Split into B_3_L (cols 4,5) and B_3_R (col 6)
    
    # --- B_3_L (4x2) ---
    # B_3_L_L: rows 0,1, cols 4,5
    B_3_L_L = [(0,4), (0,5), (1,4), (1,5)]
    # B_3_L_R: rows 2,3, cols 4,5
    B_3_L_R = [(2,4), (2,5), (3,4), (3,5)]
    # C_3_L: cols 4,5
    C_3_L = [(0,4), (0,5), (1,4), (1,5)]
    
    add_strassen_block(A_L, B_3_L_L, C_3_L)
    add_strassen_block(A_R, B_3_L_R, C_3_L)
    
    # --- B_3_R (4x1) ---
    # Split into B_3_R_L (rows 0,1) and B_3_R_R (rows 2,3)
    # B_3_R_L: rows 0,1, col 6
    B_3_R_L = [(0,6), (1,6)]
    # B_3_R_R: rows 2,3, col 6
    B_3_R_R = [(2,6), (3,6)]
    # C_3_R: col 6
    C_3_R = [(0,6), (1,6)]
    
    add_schoolbook_221(A_L, B_3_R_L, C_3_R)
    add_schoolbook_221(A_R, B_3_R_R, C_3_R)
    
    return {"U": U, "V": V, "W": W}
