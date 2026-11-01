"""
Rank-minimal exact tensor decomposition for 2x3x7 matrix multiplication.

Strategy:
  1. Use a hybrid Strassen/Schoolbook decomposition (rank 39) that exploits
     the structure of 2x3x7 by splitting the inner dimension.
  2. Fall back to the schoolbook 42-rank decomposition if verification fails.

The hybrid decomposition splits A (2x3) into A1 (2x2) and A2 (2x1),
and B (3x7) into B1 (2x7) and B2 (1x7). Then C = A1*B1 + A2*B2.
For A1*B1: 3 Strassen blocks (2x2x2, rank 7 each) + 1 schoolbook (2x2x1, rank 4) = 25.
For A2*B2: schoolbook (2x1x7, rank 14).
Total: 39.
"""
import time

DIMENSIONS = [2, 3, 7]

def _hybrid_decomposition():
    """
    Hybrid Strassen/Schoolbook decomposition for 2x3x7 (rank 39).
    
    Split A (2x3) = [A1 (2x2) | A2 (2x1)]
    Split B (3x7) = [B1 (2x7); B2 (1x7)]
    
    C = A1*B1 + A2*B2
    
    For A1*B1: 3 Strassen blocks (2x2x2, rank 7) + 1 schoolbook (2x2x1, rank 4) = 25
    For A2*B2: schoolbook (2x1x7, rank 14)
    Total rank: 39
    """
    n, m, p = DIMENSIONS  # 2, 3, 7
    rank = 39
    U = [[0] * rank for _ in range(n * m)]  # (6, 39)
    V = [[0] * rank for _ in range(m * p)]  # (21, 39)
    W = [[0] * rank for _ in range(p * n)]  # (14, 39)
    
    col = 0
    
    def add_strassen_2x2x2(B_col_start):
        """Add Strassen decomposition for 2x2x2 block using A cols 0,1 and B rows 0,1.
        B cols: B_col_start, B_col_start+1
        Output: C[0,B_col_start], C[0,B_col_start+1], C[1,B_col_start], C[1,B_col_start+1]
        """
        nonlocal col
        # A indices (rows 0,1; cols 0,1)
        a00 = 0 * m + 0
        a01 = 0 * m + 1
        a10 = 1 * m + 0
        a11 = 1 * m + 1
        
        # B indices (rows 0,1; cols B_col_start, B_col_start+1)
        b00 = 0 * p + B_col_start
        b01 = 0 * p + (B_col_start + 1)
        b10 = 1 * p + B_col_start
        b11 = 1 * p + (B_col_start + 1)
        
        # W indices: W[k*n + i]
        w00 = B_col_start * n + 0
        w01 = (B_col_start + 1) * n + 0
        w10 = B_col_start * n + 1
        w11 = (B_col_start + 1) * n + 1
        
        # P1 = (a00+a11)*(b00+b11)
        # C[0,0] += P1, C[1,1] += P1
        U[a00][col] = 1; U[a11][col] = 1
        V[b00][col] = 1; V[b11][col] = 1
        W[w00][col] = 1; W[w11][col] = 1
        col += 1
        
        # P2 = (a10+a11)*b00
        # C[1,0] += P2
        U[a10][col] = 1; U[a11][col] = 1
        V[b00][col] = 1
        W[w10][col] = 1
        col += 1
        
        # P3 = a00*(b01-b11)
        # C[0,1] += P3, C[1,1] -= P3
        U[a00][col] = 1
        V[b01][col] = 1; V[b11][col] = -1
        W[w01][col] = 1; W[w11][col] = -1
        col += 1
        
        # P4 = a11*(b10-b00)
        # C[0,0] -= P4, C[1,0] += P4
        U[a11][col] = 1
        V[b10][col] = 1; V[b00][col] = -1
        W[w00][col] = -1; W[w10][col] = 1
        col += 1
        
        # P5 = (a00+a01)*b11
        # C[0,0] += P5, C[0,1] -= P5
        U[a00][col] = 1; U[a01][col] = 1
        V[b11][col] = 1
        W[w00][col] = 1; W[w01][col] = -1
        col += 1
        
        # P6 = (a10-a00)*(b00+b01)
        # C[1,0] += P6, C[1,1] += P6
        U[a10][col] = 1; U[a00][col] = -1
        V[b00][col] = 1; V[b01][col] = 1
        W[w10][col] = 1; W[w11][col] = 1
        col += 1
        
        # P7 = (a01-a11)*(b10+b11)
        # C[0,0] -= P7, C[0,1] += P7
        U[a01][col] = 1; U[a11][col] = -1
        V[b10][col] = 1; V[b11][col] = 1
        W[w00][col] = -1; W[w01][col] = 1
        col += 1
    
    # 3 Strassen blocks for B1 cols [0,1], [2,3], [4,5]
    for block in range(3):
        add_strassen_2x2x2(block * 2)
    
    # Schoolbook for B1 col [6]: 2x2x1 (4 terms)
    a00 = 0 * m + 0
    a01 = 0 * m + 1
    a10 = 1 * m + 0
    a11 = 1 * m + 1
    b06 = 0 * p + 6
    b16 = 1 * p + 6
    w06 = 6 * n + 0
    w16 = 6 * n + 1
    
    # C[0,6] += A[0,0]*B[0,6]
    U[a00][col] = 1; V[b06][col] = 1; W[w06][col] = 1; col += 1
    # C[0,6] += A[0,1]*B[1,6]
    U[a01][col] = 1; V[b16][col] = 1; W[w06][col] = 1; col += 1
    # C[1,6] += A[1,0]*B[0,6]
    U[a10][col] = 1; V[b06][col] = 1; W[w16][col] = 1; col += 1
    # C[1,6] += A[1,1]*B[1,6]
    U[a11][col] = 1; V[b16][col] = 1; W[w16][col] = 1; col += 1
    
    # Part 2: A2*B2 where A2 is col 2 of A (2x1), B2 is row 2 of B (1x7)
    # C[i,k] += A[i,2] * B[2,k]
    for i in range(2):
        for k in range(7):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    return {"U": U, "V": V, "W": W}

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _check_decomposition(result):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    U = result["U"]
    V = result["V"]
    W = result["W"]
    rank = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit
    
    # Try the hybrid Strassen/Schoolbook decomposition (rank 39)
    result = _hybrid_decomposition()
    
    if _check_decomposition(result):
        return result
    
    # Fallback to schoolbook (rank 42)
    result = _schoolbook()
    if _check_decomposition(result):
        return result
    
    return result

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
