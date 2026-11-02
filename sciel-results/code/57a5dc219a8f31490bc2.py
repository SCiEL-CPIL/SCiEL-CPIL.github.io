import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 130.0

DIMENSIONS = [2, 4, 5]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x5 matrix.
    
    Strategy: Optimized block decomposition.
    Split output C into C[:, 0:2] and C[:, 2:5].
    
    1. C[:, 0:2] = A[:, 0:4] * B[0:4, 0:2]
       Split A into A_L (cols 0:3) and A_R (col 3).
       Split B into B_L (rows 0:3) and B_R (row 3).
       C[:, 0:2] = A_L * B_L + A_R * B_R
       
       A_L * B_L is 2x3 * 3x2.
       Split A_L into A_LL (cols 0:2) and A_LR (col 2).
       Split B_L into B_LL (rows 0:2) and B_LR (row 2).
       A_L * B_L = A_LL * B_LL + A_LR * B_LR
       A_LL * B_LL is 2x2 * 2x2 -> Rank 7 (Strassen)
       A_LR * B_LR is 2x1 * 1x2 -> Rank 2 (Special Strassen)
       
       A_R * B_R is 2x1 * 1x2 -> Rank 2 (Special Strassen)
       
       Total Rank for C[:, 0:2] = 7 + 2 + 2 = 11.
       
    2. C[:, 2:5] = A[:, 0:4] * B[0:4, 2:5]
       Split A into A_L (cols 0:2) and A_R (cols 2:4).
       C[:, 2:5] = A_L * B[0:2, 2:5] + A_R * B[2:4, 2:5]
       
       A_L * B[0:2, 2:5] is 2x2 * 2x3.
       Split B part into B_LL (cols 2:4) and B_LR (col 4).
       A_L * B_LL is 2x2 * 2x2 -> Rank 7 (Strassen)
       A_L * B_LR is 2x2 * 2x1 -> Rank 4 (Schoolbook)
       Total 11.
       
       A_R * B[2:4, 2:5] is 2x2 * 2x3.
       Split B part into B_RL (cols 2:4) and B_RR (col 4).
       A_R * B_RL is 2x2 * 2x2 -> Rank 7 (Strassen)
       A_R * B_RR is 2x2 * 2x1 -> Rank 4 (Schoolbook)
       Total 11.
       
       Total Rank for C[:, 2:5] = 11 + 11 = 22.
       
    Total Rank = 11 + 22 = 33.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    
    rank = 33
    
    # Initialize factor matrices
    U = [[0] * rank for _ in range(n * m)] # (8, 33)
    V = [[0] * rank for _ in range(m * p)] # (20, 33)
    W = [[0] * rank for _ in range(p * n)] # (10, 33)
    
    col = 0
    
    def add_strassen_222(i0, j0, k0, col_start):
        """
        Add a Strassen 2x2x2 decomposition for:
        C[i0:i0+2, k0:k0+2] = A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2]
        """
        nonlocal col
        c = col_start
        
        # M1 = (A00 + A11) * (B00 + B11)
        U[i0 * m + j0][c] += 1
        U[(i0+1) * m + (j0+1)][c] += 1
        V[j0 * p + k0][c] += 1
        V[(j0+1) * p + (k0+1)][c] += 1
        W[k0 * n + i0][c] += 1
        W[(k0+1) * n + (i0+1)][c] += 1
        c += 1
        
        # M2 = (A10 + A11) * B00
        U[(i0+1) * m + j0][c] += 1
        U[(i0+1) * m + (j0+1)][c] += 1
        V[j0 * p + k0][c] += 1
        W[k0 * n + (i0+1)][c] += 1
        W[(k0+1) * n + (i0+1)][c] -= 1
        c += 1
        
        # M3 = A00 * (B01 - B11)
        U[i0 * m + j0][c] += 1
        V[j0 * p + (k0+1)][c] += 1
        V[(j0+1) * p + (k0+1)][c] -= 1
        W[(k0+1) * n + i0][c] += 1
        W[(k0+1) * n + (i0+1)][c] += 1
        c += 1
        
        # M4 = A11 * (B10 - B00)
        U[(i0+1) * m + (j0+1)][c] += 1
        V[(j0+1) * p + k0][c] += 1
        V[j0 * p + k0][c] -= 1
        W[k0 * n + i0][c] += 1
        W[k0 * n + (i0+1)][c] += 1
        c += 1
        
        # M5 = (A00 + A01) * B11
        U[i0 * m + j0][c] += 1
        U[i0 * m + (j0+1)][c] += 1
        V[(j0+1) * p + (k0+1)][c] += 1
        W[k0 * n + i0][c] -= 1
        W[(k0+1) * n + i0][c] += 1
        c += 1
        
        # M6 = (A10 - A00) * (B00 + B01)
        U[(i0+1) * m + j0][c] += 1
        U[i0 * m + j0][c] -= 1
        V[j0 * p + k0][c] += 1
        V[j0 * p + (k0+1)][c] += 1
        W[(k0+1) * n + (i0+1)][c] += 1
        c += 1
        
        # M7 = (A01 - A11) * (B10 + B11)
        U[i0 * m + (j0+1)][c] += 1
        U[(i0+1) * m + (j0+1)][c] -= 1
        V[(j0+1) * p + k0][c] += 1
        V[(j0+1) * p + (k0+1)][c] += 1
        W[k0 * n + i0][c] += 1
        c += 1
        
        col = c
        return c

    def add_strassen_212(i0, j0, k0, col_start):
        """
        Add a 2x1x2 decomposition (Rank 2) for:
        C[i0:i0+2, k0:k0+2] = A[i0:i0+2, j0:j0+1] * B[j0:j0+1, k0:k0+2]
        A is 2x1, B is 1x2.
        M1 = (A[i0,j0] + A[i0+1,j0]) * B[j0,k0]
        M2 = (A[i0,j0] + A[i0+1,j0]) * B[j0,k0+1]
        C[i0,k0] += M1
        C[i0+1,k0] += M1
        C[i0,k0+1] += M2
        C[i0+1,k0+1] += M2
        """
        nonlocal col
        c = col_start
        
        # M1
        U[i0 * m + j0][c] += 1
        U[(i0+1) * m + j0][c] += 1
        V[j0 * p + k0][c] += 1
        W[k0 * n + i0][c] += 1
        W[k0 * n + (i0+1)][c] += 1
        c += 1
        
        # M2
        U[i0 * m + j0][c] += 1
        U[(i0+1) * m + j0][c] += 1
        V[j0 * p + (k0+1)][c] += 1
        W[(k0+1) * n + i0][c] += 1
        W[(k0+1) * n + (i0+1)][c] += 1
        c += 1
        
        col = c
        return c

    def add_schoolbook_221(i0, j0, k0, col_start):
        """
        Add schoolbook decomposition for:
        C[i0:i0+2, k0:k0+1] = A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+1]
        Rank 4.
        """
        nonlocal col
        c = col_start
        for di in range(2):
            for dj in range(2):
                U[(i0 + di) * m + (j0 + dj)][c] = 1
                V[(j0 + dj) * p + k0][c] = 1
                W[k0 * n + (i0 + di)][c] = 1
                c += 1
        col = c
        return c

    # Block 1: C[:, 0:2]
    # 1.1: A[:, 0:2] * B[0:2, 0:2] (2x2 * 2x2) -> Rank 7
    col = add_strassen_222(0, 0, 0, col)
    
    # 1.2: A[:, 2:3] * B[2:3, 0:2] (2x1 * 1x2) -> Rank 2
    # A cols 2, B rows 2. Output C[:, 0:2]
    col = add_strassen_212(0, 2, 0, col)
    
    # 1.3: A[:, 3:4] * B[3:4, 0:2] (2x1 * 1x2) -> Rank 2
    # A cols 3, B rows 3. Output C[:, 0:2]
    col = add_strassen_212(0, 3, 0, col)
    
    # Block 2: C[:, 2:5]
    # 2.1: A[:, 0:2] * B[0:2, 2:4] (2x2 * 2x2) -> Rank 7
    col = add_strassen_222(0, 0, 2, col)
    
    # 2.2: A[:, 0:2] * B[0:2, 4:5] (2x2 * 2x1) -> Rank 4
    col = add_schoolbook_221(0, 0, 4, col)
    
    # 2.3: A[:, 2:4] * B[2:4, 2:4] (2x2 * 2x2) -> Rank 7
    col = add_strassen_222(0, 2, 2, col)
    
    # 2.4: A[:, 2:4] * B[2:4, 4:5] (2x2 * 2x1) -> Rank 4
    col = add_schoolbook_221(0, 2, 4, col)
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
