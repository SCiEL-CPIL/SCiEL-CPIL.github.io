"""Matrix multiplication 3x4x4 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]

def construct():
    """Find a rank-minimal exact tensor decomposition for 3x4x4 matrix multiplication.
    
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    Score is the rank (lower is better).
    """
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 4, 4
    
    # Strategy: Use structured Strassen block decomposition for rank 44.
    # Partition A (3x4) into top 2 rows and bottom 1 row.
    # Partition B (4x4) into four 2x2 blocks.
    # Top 2x4 * 4x4: four 2x2x2 subproblems, each done via Strassen (rank 7). Total: 4*7 = 28.
    # Bottom 1x4 * 4x4: four 1x2x2 subproblems, each done via schoolbook (rank 4). Total: 4*4 = 16.
    # Grand total: 28 + 16 = 44.
    
    rank = 44
    U = np.zeros((n * m, rank), dtype=np.int8)
    V = np.zeros((m * p, rank), dtype=np.int8)
    W = np.zeros((p * n, rank), dtype=np.int8)
    
    col = 0
    
    # Helper to add Strassen 2x2x2 block
    # A block: rows [rA, rA+1], cols [cA, cA+1]
    # B block: rows [rB, rB+1], cols [cB, cB+1]
    # C block: rows [rA, rA+1], cols [cB, cB+1]
    def add_strassen(rA, cA, rB, cB):
        nonlocal col
        # Strassen 7 terms for 2x2 * 2x2 -> 2x2
        # M1 = (X11+X22)(Y11+Y22)
        U[(rA+0)*m + (cA+0), col] = 1
        U[(rA+1)*m + (cA+1), col] = 1
        V[(rB+0)*p + (cB+0), col] = 1
        V[(rB+1)*p + (cB+1), col] = 1
        W[(cB+0)*n + (rA+0), col] = 1  # Z11
        W[(cB+1)*n + (rA+1), col] = 1  # Z22
        col += 1
        
        # M2 = (X21+X22)Y11
        U[(rA+1)*m + (cA+0), col] = 1
        U[(rA+1)*m + (cA+1), col] = 1
        V[(rB+0)*p + (cB+0), col] = 1
        W[(cB+0)*n + (rA+1), col] = 1  # Z21
        W[(cB+1)*n + (rA+1), col] = -1 # Z22
        col += 1
        
        # M3 = X11(Y12-Y22)
        U[(rA+0)*m + (cA+0), col] = 1
        V[(rB+0)*p + (cB+1), col] = 1
        V[(rB+1)*p + (cB+1), col] = -1
        W[(cB+1)*n + (rA+0), col] = 1  # Z12
        W[(cB+1)*n + (rA+1), col] = 1  # Z22
        col += 1
        
        # M4 = X22(Y21-Y11)
        U[(rA+1)*m + (cA+1), col] = 1
        V[(rB+1)*p + (cB+0), col] = 1
        V[(rB+0)*p + (cB+0), col] = -1
        W[(cB+0)*n + (rA+0), col] = 1  # Z11
        W[(cB+0)*n + (rA+1), col] = 1  # Z21
        col += 1
        
        # M5 = (X11+X12)Y22
        U[(rA+0)*m + (cA+0), col] = 1
        U[(rA+0)*m + (cA+1), col] = 1
        V[(rB+1)*p + (cB+1), col] = 1
        W[(cB+1)*n + (rA+0), col] = 1  # Z12
        W[(cB+0)*n + (rA+0), col] = -1 # Z11
        col += 1
        
        # M6 = (X21-X11)(Y11+Y12)
        U[(rA+1)*m + (cA+0), col] = 1
        U[(rA+0)*m + (cA+0), col] = -1
        V[(rB+0)*p + (cB+0), col] = 1
        V[(rB+0)*p + (cB+1), col] = 1
        W[(cB+1)*n + (rA+1), col] = 1  # Z22
        col += 1
        
        # M7 = (X12-X22)(Y21+Y22)
        U[(rA+0)*m + (cA+1), col] = 1
        U[(rA+1)*m + (cA+1), col] = -1
        V[(rB+1)*p + (cB+0), col] = 1
        V[(rB+1)*p + (cB+1), col] = 1
        W[(cB+0)*n + (rA+0), col] = 1  # Z11
        col += 1

    # Helper to add schoolbook 1x2x2 block
    # A block: row rA, cols [cA, cA+1]
    # B block: rows [rB, rB+1], cols [cB, cB+1]
    # C block: row rA, cols [cB, cB+1]
    def add_schoolbook_1x2x2(rA, cA, rB, cB):
        nonlocal col
        # C0 = A0 B00 + A1 B10
        U[(rA)*m + (cA+0), col] = 1
        V[(rB+0)*p + (cB+0), col] = 1
        W[(cB+0)*n + (rA), col] = 1
        col += 1
        
        U[(rA)*m + (cA+1), col] = 1
        V[(rB+1)*p + (cB+0), col] = 1
        W[(cB+0)*n + (rA), col] = 1
        col += 1
        
        # C1 = A0 B01 + A1 B11
        U[(rA)*m + (cA+0), col] = 1
        V[(rB+0)*p + (cB+1), col] = 1
        W[(cB+1)*n + (rA), col] = 1
        col += 1
        
        U[(rA)*m + (cA+1), col] = 1
        V[(rB+1)*p + (cB+1), col] = 1
        W[(cB+1)*n + (rA), col] = 1
        col += 1

    # Apply blocks for top 2 rows (Strassen)
    # A rows 0-1, B rows 0-1, C cols 0-1
    add_strassen(0, 0, 0, 0)
    # A rows 0-1, B rows 2-3, C cols 0-1
    add_strassen(0, 2, 2, 0)
    # A rows 0-1, B rows 0-1, C cols 2-3
    add_strassen(0, 0, 0, 2)
    # A rows 0-1, B rows 2-3, C cols 2-3
    add_strassen(0, 2, 2, 2)
    
    # Apply blocks for bottom 1 row (Schoolbook)
    # A row 2, B rows 0-1, C cols 0-1
    add_schoolbook_1x2x2(2, 0, 0, 0)
    # A row 2, B rows 2-3, C cols 0-1
    add_schoolbook_1x2x2(2, 2, 2, 0)
    # A row 2, B rows 0-1, C cols 2-3
    add_schoolbook_1x2x2(2, 0, 0, 2)
    # A row 2, B rows 2-3, C cols 2-3
    add_schoolbook_1x2x2(2, 2, 2, 2)
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Verify the decomposition
    def verify(U, V, W):
        rank = len(U[0])
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
            return False
        for mat in [U, V, W]:
            for row in mat:
                for val in row:
                    if val * 2 != int(val * 2):
                        return False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if s != 1:
                        return False
        return True
    
    if verify(U_list, V_list, W_list):
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook (rank 48)
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
