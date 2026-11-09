"""Matrix multiplication 4x4x5 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def _strassen_2x2():
    """Return U (4x7), V (4x7), W (4x7) for 2x2x2 Strassen decomposition.
    U rows: A00, A01, A10, A11
    V rows: B00, B01, B10, B11
    W rows: C00, C01, C10, C11
    """
    U = np.array([
        [1, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 1],
    ], dtype=np.int64)
    V = np.array([
        [1, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 1],
        [0, 0, 0, 0, 0, 1, 1],
    ], dtype=np.int64)
    W = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0],
    ], dtype=np.int64)
    return U, V, W


def _build_4x4x4_strassen():
    """Build 4x4x4 decomposition using block Strassen (rank 49).
    
    U: (16, 49) - rows indexed by i*4+j for A[i,j]
    V: (16, 49) - rows indexed by j*4+k for B[j,k]
    W: (16, 49) - rows indexed by k*4+i for C[i,k] (trace convention)
    """
    n = 4
    u2, v2, w2 = _strassen_2x2()
    
    U = np.zeros((16, 49), dtype=np.int64)
    V = np.zeros((16, 49), dtype=np.int64)
    W = np.zeros((16, 49), dtype=np.int64)
    
    # Block structure:
    # A = [[A00, A01], [A10, A11]] where each block is 2x2
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    #
    # A00 = A[0:2, 0:2], A01 = A[0:2, 2:4]
    # A10 = A[2:4, 0:2], A11 = A[2:4, 2:4]
    # Similarly for B and C.
    #
    # Strassen formulas:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each Mi, it's a 2x2x2 multiplication.
    # The 2x2 A_sub has entries that are linear combinations of 4x4 A entries.
    # Similarly for B_sub and C_sub.
    
    # Define helper functions to get row indices
    def a_row(i, j):
        """Row index in U for A[i,j]"""
        return i * 4 + j
    
    def b_row(j, k):
        """Row index in V for B[j,k]"""
        return j * 4 + k
    
    def w_row(k, i):
        """Row index in W for C[i,k] (trace convention: W[k*n+i])"""
        return k * 4 + i
    
    # For each of the 7 Strassen multiplications, define:
    # - a_entries: dict mapping (r,c) in 2x2 to list of (U_row, coeff)
    # - b_entries: dict mapping (r,c) in 2x2 to list of (V_row, coeff)
    # - c_entries: dict mapping (r,c) in 2x2 to list of (W_row, coeff)
    # - c_offset: (row_offset, col_offset) for where C_sub maps to in 4x4 C
    
    # M1: (A00+A11)(B00+B11) -> contributes to C00 and C11
    # A_sub[0,0] = A[0,0]+A[2,2]
    # A_sub[0,1] = A[0,1]+A[2,3]
    # A_sub[1,0] = A[1,0]+A[3,2]
    # A_sub[1,1] = A[1,1]+A[3,3]
    # B_sub[0,0] = B[0,0]+B[2,2]
    # B_sub[0,1] = B[0,1]+B[2,3]
    # B_sub[1,0] = B[1,0]+B[3,2]
    # B_sub[1,1] = B[1,1]+B[3,3]
    # C_sub maps to C[0:2, 0:2] (C00) with coeff +1 AND C[2:4, 2:4] (C11) with coeff +1
    
    # Actually, let me reconsider. In the standard Strassen:
    # C00 = M1 + M4 - M5 + M7
    # C11 = M1 - M2 + M3 + M6
    #
    # M1 is a 2x2 matrix. When we say C00 = M1 + ..., it means the 2x2 block C00
    # gets M1 added to it. Similarly C11 gets M1 added.
    #
    # So M1's (0,0) element contributes to C[0,0] (from C00) and C[2,2] (from C11).
    # M1's (0,1) element contributes to C[0,1] (from C00) and C[2,3] (from C11).
    # M1's (1,0) element contributes to C[1,0] (from C00) and C[3,2] (from C11).
    # M1's (1,1) element contributes to C[1,1] (from C00) and C[3,3] (from C11).
    
    # Let me define the 7 multiplications with their A, B, C mappings.
    
    def add_multiplication(U_mat, V_mat, W_mat, a_entries, b_entries, c_entries, col_start):
        """Add one 2x2x2 Strassen multiplication (7 terms) to the decomposition.
        
        a_entries: list of 4 dicts, each mapping position (r,c) to list of (U_row, coeff)
                   Actually, let's use a different structure.
                   a_entries[i] for i in 0..3 represents the i-th position in the 2x2 A_sub.
                   Position 0 = (0,0), 1 = (0,1), 2 = (1,0), 3 = (1,1)
                   Each a_entries[i] is a list of (U_row, coeff) pairs.
        """
        u2_local, v2_local, w2_local = _strassen_2x2()
        
        for term in range(7):
            col = col_start + term
            # For each of the 4 positions in A_sub
            for pos in range(4):
                for (row, coeff) in a_entries[pos]:
                    U_mat[row, col] += u2_local[pos, term] * coeff
            # For each of the 4 positions in B_sub
            for pos in range(4):
                for (row, coeff) in b_entries[pos]:
                    V_mat[row, col] += v2_local[pos, term] * coeff
            # For each of the 4 positions in C_sub
            for pos in range(4):
                for (row, coeff) in c_entries[pos]:
                    W_mat[row, col] += w2_local[pos, term] * coeff
    
    # Define the 7 multiplications
    
    # M1: (A00+A11)(B00+B11)
    # A_sub positions:
    #   (0,0): A[0,0]+A[2,2] -> [(a_row(0,0), 1), (a_row(2,2), 1)]
    #   (0,1): A[0,1]+A[2,3] -> [(a_row(0,1), 1), (a_row(2,3), 1)]
    #   (1,0): A[1,0]+A[3,2] -> [(a_row(1,0), 1), (a_row(3,2), 1)]
    #   (1,1): A[1,1]+A[3,3] -> [(a_row(1,1), 1), (a_row(3,3), 1)]
    a1 = [
        [(a_row(0,0), 1), (a_row(2,2), 1)],
        [(a_row(0,1), 1), (a_row(2,3), 1)],
        [(a_row(1,0), 1), (a_row(3,2), 1)],
        [(a_row(1,1), 1), (a_row(3,3), 1)],
    ]
    b1 = [
        [(b_row(0,0), 1), (b_row(2,2), 1)],
        [(b_row(0,1), 1), (b_row(2,3), 1)],
        [(b_row(1,0), 1), (b_row(3,2), 1)],
        [(b_row(1,1), 1), (b_row(3,3), 1)],
    ]
    # C_sub contributes to C00 (top-left 2x2) with +1 and C11 (bottom-right 2x2) with +1
    # C_sub[0,0] -> C[0,0] (w_row(0,0)) and C[2,2] (w_row(2,2))
    # C_sub[0,1] -> C[0,1] (w_row(1,0)) and C[2,3] (w_row(3,2))
    # C_sub[1,0] -> C[1,0] (w_row(0,1)) and C[3,2] (w_row(2,3))
    # C_sub[1,1] -> C[1,1] (w_row(1,1)) and C[3,3] (w_row(3,3))
    c1 = [
        [(w_row(0,0), 1), (w_row(2,2), 1)],
        [(w_row(1,0), 1), (w_row(3,2), 1)],
        [(w_row(0,1), 1), (w_row(2,3), 1)],
        [(w_row(1,1), 1), (w_row(3,3), 1)],
    ]
    
    # M2: (A10+A11)B00
    # A_sub[0,0] = A[2,0]+A[4,0]... wait, A is 4x4, so indices are 0-3.
    # A10 = A[2:4, 0:2], A11 = A[2:4, 2:4]
    # A_sub[0,0] = A[2,0]+A[2,2]
    # A_sub[0,1] = A[2,1]+A[2,3]
    # A_sub[1,0] = A[3,0]+A[3,2]
    # A_sub[1,1] = A[3,1]+A[3,3]
    a2 = [
        [(a_row(2,0), 1), (a_row(2,2), 1)],
        [(a_row(2,1), 1), (a_row(2,3), 1)],
        [(a_row(3,0), 1), (a_row(3,2), 1)],
        [(a_row(3,1), 1), (a_row(3,3), 1)],
    ]
    b2 = [
        [(b_row(0,0), 1)],
        [(b_row(0,1), 1)],
        [(b_row(1,0), 1)],
        [(b_row(1,1), 1)],
    ]
    # C_sub contributes to C10 (bottom-left 2x2) with +1
    # C_sub[0,0] -> C[2,0] (w_row(0,2))
    # C_sub[0,1] -> C[2,1] (w_row(1,2))
    # C_sub[1,0] -> C[3,0] (w_row(0,3))
    # C_sub[1,1] -> C[3,1] (w_row(1,3))
    c2 = [
        [(w_row(0,2), 1)],
        [(w_row(1,2), 1)],
        [(w_row(0,3), 1)],
        [(w_row(1,3), 1)],
    ]
    
    # M3: A00(B01-B11)
    a3 = [
        [(a_row(0,0), 1)],
        [(a_row(0,1), 1)],
        [(a_row(1,0), 1)],
        [(a_row(1,1), 1)],
    ]
    b3 = [
        [(b_row(0,2), 1), (b_row(2,2), -1)],
        [(b_row(0,3), 1), (b_row(2,3), -1)],
        [(b_row(1,2), 1), (b_row(3,2), -1)],
        [(b_row(1,3), 1), (b_row(3,3), -1)],
    ]
    # C_sub contributes to C01 (top-right 2x2) with +1
    # C_sub[0,0] -> C[0,2] (w_row(2,0))
    # C_sub[0,1] -> C[0,3] (w_row(3,0))
    # C_sub[1,0] -> C[1,2] (w_row(2,1))
    # C_sub[1,1] -> C[1,3] (w_row(3,1))
    c3 = [
        [(w_row(2,0), 1)],
        [(w_row(3,0), 1)],
        [(w_row(2,1), 1)],
        [(w_row(3,1), 1)],
    ]
    
    # M4: A11(B10-B00)
    a4 = [
        [(a_row(2,2), 1)],
        [(a_row(2,3), 1)],
        [(a_row(3,2), 1)],
        [(a_row(3,3), 1)],
    ]
    b4 = [
        [(b_row(1,0), 1), (b_row(0,0), -1)],
        [(b_row(1,1), 1), (b_row(0,1), -1)],
        [(b_row(2,0), 1), (b_row(1,0), -1)],
        [(b_row(2,1), 1), (b_row(1,1), -1)],
    ]
    # C_sub contributes to C00 (top-left 2x2) with +1 and C11 (bottom-right 2x2) with +1
    c4 = [
        [(w_row(0,0), 1), (w_row(2,2), 1)],
        [(w_row(1,0), 1), (w_row(3,2), 1)],
        [(w_row(0,1), 1), (w_row(2,3), 1)],
        [(w_row(1,1), 1), (w_row(3,3), 1)],
    ]
    
    # M5: (A00+A01)B11
    a5 = [
        [(a_row(0,0), 1), (a_row(0,2), 1)],
        [(a_row(0,1), 1), (a_row(0,3), 1)],
        [(a_row(1,0), 1), (a_row(1,2), 1)],
        [(a_row(1,1), 1), (a_row(1,3), 1)],
    ]
    b5 = [
        [(b_row(2,2), 1)],
        [(b_row(2,3), 1)],
        [(b_row(3,2), 1)],
        [(b_row(3,3), 1)],
    ]
    # C_sub contributes to C00 (top-left 2x2) with -1 and C01 (top-right 2x2) with +1
    c5 = [
        [(w_row(0,0), -1), (w_row(2,0), 1)],
        [(w_row(1,0), -1), (w_row(3,0), 1)],
        [(w_row(0,1), -1), (w_row(2,1), 1)],
        [(w_row(1,1), -1), (w_row(3,1), 1)],
    ]
    
    # M6: (A10-A00)(B00+B01)
    a6 = [
        [(a_row(2,0), 1), (a_row(0,0), -1)],
        [(a_row(2,1), 1), (a_row(0,1), -1)],
        [(a_row(3,0), 1), (a_row(1,0), -1)],
        [(a_row(3,1), 1), (a_row(1,1), -1)],
    ]
    b6 = [
        [(b_row(0,0), 1), (b_row(0,2), 1)],
        [(b_row(0,1), 1), (b_row(0,3), 1)],
        [(b_row(1,0), 1), (b_row(1,2), 1)],
        [(b_row(1,1), 1), (b_row(1,3), 1)],
    ]
    # C_sub contributes to C11 (bottom-right 2x2) with +1
    c6 = [
        [(w_row(2,2), 1)],
        [(w_row(3,2), 1)],
        [(w_row(2,3), 1)],
        [(w_row(3,3), 1)],
    ]
    
    # M7: (A01-A11)(B10+B11)
    a7 = [
        [(a_row(0,2), 1), (a_row(2,2), -1)],
        [(a_row(0,3), 1), (a_row(2,3), -1)],
        [(a_row(1,2), 1), (a_row(3,2), -1)],
        [(a_row(1,3), 1), (a_row(3,3), -1)],
    ]
    b7 = [
        [(b_row(1,0), 1), (b_row(2,0), 1)],
        [(b_row(1,1), 1), (b_row(2,1), 1)],
        [(b_row(2,0), 1), (b_row(3,0), 1)],
        [(b_row(2,1), 1), (b_row(3,1), 1)],
    ]
    # C_sub contributes to C00 (top-left 2x2) with +1
    c7 = [
        [(w_row(0,0), 1)],
        [(w_row(1,0), 1)],
        [(w_row(0,1), 1)],
        [(w_row(1,1), 1)],
    ]
    
    # Add all 7 multiplications
    col_start = 0
    add_multiplication(U, V, W, a1, b1, c1, col_start)
    col_start = 7
    add_multiplication(U, V, W, a2, b2, c2, col_start)
    col_start = 14
    add_multiplication(U, V, W, a3, b3, c3, col_start)
    col_start = 21
    add_multiplication(U, V, W, a4, b4, c4, col_start)
    col_start = 28
    add_multiplication(U, V, W, a5, b5, c5, col_start)
    col_start = 35
    add_multiplication(U, V, W, a6, b6, c6, col_start)
    col_start = 42
    add_multiplication(U, V, W, a7, b7, c7, col_start)
    
    return U, V, W


def _build_4x4x5_decomposition():
    """Build 4x4x5 decomposition.
    
    Split B into B_left (4x4) and B_right (4x1).
    C_left = A * B_left (4x4), rank 49 via Strassen
    C_right = A * B_right (4x1), rank 16 via schoolbook
    Total rank: 49 + 16 = 65
    
    U: (16, 65) - rows for A[i,j]
    V: (20, 65) - rows for B[j,k] where k=0..4
    W: (20, 65) - rows for C[i,k] where k=0..4, using W[k*n+i] convention
    """
    n, m, p = 4, 4, 5
    
    # Build 4x4x4 part (rank 49)
    U4, V4, W4 = _build_4x4x4_strassen()
    
    # Build 4x4x1 part (rank 16)
    # For k=4 (the 5th column of B), we need schoolbook:
    # C[i,4] = sum_j A[i,j] * B[j,4]
    U1 = np.zeros((16, 16), dtype=np.int64)
    V1 = np.zeros((20, 16), dtype=np.int64)
    W1 = np.zeros((20, 16), dtype=np.int64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            U1[i * m + j, col] = 1
            V1[j * p + 4, col] = 1  # B[j,4]
            W1[4 * n + i, col] = 1  # C[i,4] -> W[4*4+i]
            col += 1
    
    # Combine: 4x4x4 part uses V rows for B[j,0..3] and W rows for C[i,0..3]
    # 4x4x1 part uses V row for B[j,4] and W row for C[i,4]
    
    U = np.zeros((16, 65), dtype=np.int64)
    V = np.zeros((20, 65), dtype=np.int64)
    W = np.zeros((20, 65), dtype=np.int64)
    
    # Copy 4x4x4 part (first 49 columns)
    U[:, :49] = U4
    V[:16, :49] = V4  # V4 is 16x49, corresponds to B[j,0..3]
    W[:16, :49] = W4  # W4 is 16x49, corresponds to C[i,0..3]
    
    # Copy 4x4x1 part (last 16 columns)
    U[:, 49:] = U1
    V[:, 49:] = V1
    W[:, 49:] = W1
    
    return U, V, W


def construct():
    """Construct a decomposition for 4x4x5 matrix multiplication.
    Returns {'U': (16, rank), 'V': (20, rank), 'W': (20, rank)}
    """
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    U, V, W = _build_4x4x5_decomposition()
    
    # Convert to lists for the return interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
