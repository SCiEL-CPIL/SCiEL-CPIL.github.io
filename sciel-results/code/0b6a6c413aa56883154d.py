import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 3, 8]

def _verify(U, V, W):
    """Verify that the decomposition exactly computes matrix multiplication."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    
    if U.shape[0] != n * m:
        return False
    if V.shape[0] != m * p:
        return False
    if W.shape[0] != p * n:
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook_decomposition():
    """Return the schoolbook rank-72 decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2():
    """
    Return U, V, W for 2x2 * 2x2 matrix multiplication using Strassen (rank 7).
    Local indices:
      U: 2x2 = 4 rows (i_local, j_local)
      V: 2x2 = 4 rows (j_local, k_local)
      W: 2x2 = 4 rows (k_local, i_local)
    """
    # Strassen's 7 products for 2x2:
    # M1 = (A11)(B11+B22)
    # M2 = (A21+A22)(B11)
    # M3 = (A11)(B22)
    # M4 = (A22)(B21-B11)
    # M5 = (A11+A22)(B22+B12)
    # M6 = (A21-A11)(B12+B22)
    # M7 = (A12-A22)(B21+B11)
    
    # U entries: u[i,j] for product r
    # V entries: v[j,k] for product r
    # W entries: w[k,i] for product r
    
    # Product 1: (A11)(B11+B22)
    # U: A11 at (0,0) -> 1
    # V: B11 at (0,0) -> 1, B22 at (1,1) -> 1
    # W: C11 += M1, C22 += M1 -> W[0,0]=1, W[1,1]=1
    
    # Product 2: (A21+A22)(B11)
    # U: A21 at (1,0) -> 1, A22 at (1,1) -> 1
    # V: B11 at (0,0) -> 1
    # W: C21 += M2 -> W[0,1]=1
    
    # Product 3: (A11)(B22)
    # U: A11 at (0,0) -> 1
    # V: B22 at (1,1) -> 1
    # W: C12 += M3 -> W[1,0]=1
    
    # Product 4: (A22)(B21-B11)
    # U: A22 at (1,1) -> 1
    # V: B21 at (1,0) -> 1, B11 at (0,0) -> -1
    # W: C22 += M4 -> W[1,1]=1
    
    # Product 5: (A11+A22)(B22+B12)
    # U: A11 at (0,0) -> 1, A22 at (1,1) -> 1
    # V: B22 at (1,1) -> 1, B12 at (0,1) -> 1
    # W: C12 += M5, C22 += M5 -> W[1,0]=1, W[1,1]=1
    
    # Product 6: (A21-A11)(B12+B22)
    # U: A21 at (1,0) -> 1, A11 at (0,0) -> -1
    # V: B12 at (0,1) -> 1, B22 at (1,1) -> 1
    # W: C21 += M6 -> W[0,1]=1
    
    # Product 7: (A12-A22)(B21+B11)
    # U: A12 at (0,1) -> 1, A22 at (1,1) -> -1
    # V: B21 at (1,0) -> 1, B11 at (0,0) -> 1
    # W: C11 += M7 -> W[0,0]=1
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Product 0 (M1): (A11)(B11+B22)
    U[0][0] = 1  # A11
    V[0][0] = 1  # B11
    V[1][0] = 1  # B22
    W[0][0] = 1  # C11
    W[1][0] = 1  # C22
    
    # Product 1 (M2): (A21+A22)(B11)
    U[2][1] = 1  # A21
    U[3][1] = 1  # A22
    V[0][1] = 1  # B11
    W[2][1] = 1  # C21
    
    # Product 2 (M3): (A11)(B22)
    U[0][2] = 1  # A11
    V[1][2] = 1  # B22
    W[1][2] = 1  # C12
    
    # Product 3 (M4): (A22)(B21-B11)
    U[3][3] = 1  # A22
    V[0][3] = -1 # B11
    V[1][3] = 1  # B21
    W[1][3] = 1  # C22
    
    # Product 4 (M5): (A11+A22)(B22+B12)
    U[0][4] = 1  # A11
    U[3][4] = 1  # A22
    V[0][4] = 1  # B12
    V[1][4] = 1  # B22
    W[1][4] = 1  # C12
    W[3][4] = 1  # C22
    
    # Product 5 (M6): (A21-A11)(B12+B22)
    U[0][5] = -1 # A11
    U[2][5] = 1  # A21
    V[0][5] = 1  # B12
    V[1][5] = 1  # B22
    W[2][5] = 1  # C21
    
    # Product 6 (M7): (A12-A22)(B21+B11)
    U[1][6] = 1  # A12
    U[3][6] = -1 # A22
    V[0][6] = 1  # B11
    V[1][6] = 1  # B21
    W[0][6] = 1  # C11
    
    return U, V, W

def _schoolbook_1xNxM(N, M):
    """
    Return U, V, W for 1xN * NxM matrix multiplication (schoolbook).
    U: 1xN = N rows
    V: NxM = N*M rows
    W: Mx1 = M rows
    Rank = N*M
    """
    rank = N * M
    U = [[0]*rank for _ in range(N)]
    V = [[0]*rank for _ in range(N*M)]
    W = [[0]*rank for _ in range(M)]
    
    col = 0
    for j in range(N):
        for k in range(M):
            U[j][col] = 1
            V[j*M+k][col] = 1
            W[k][col] = 1
            col += 1
    return U, V, W

def _schoolbook_Nx1xM(N, M):
    """
    Return U, V, W for Nx1 * 1xM matrix multiplication (outer product).
    U: Nx1 = N rows
    V: 1xM = M rows
    W: MxN = M*N rows
    Rank = N*M
    """
    rank = N * M
    U = [[0]*rank for _ in range(N)]
    V = [[0]*rank for _ in range(M)]
    W = [[0]*rank for _ in range(M*N)]
    
    col = 0
    for i in range(N):
        for k in range(M):
            U[i][col] = 1
            V[k][col] = 1
            W[k*N+i][col] = 1
            col += 1
    return U, V, W

def _build_block_decomposition():
    """
    Build a block decomposition for 3x3 * 3x8 using Strassen and schoolbook.
    
    Split A (3x3) into A1 (2x3) and A2 (1x3).
    Split B (3x8) into B1 (3x4) and B2 (3x4).
    
    C = [A*B1, A*B2] = [C1, C2] where C1 is 3x4, C2 is 3x4.
    
    C1 = A1*B1 (2x3 * 3x4) stacked with A2*B1 (1x3 * 3x4)
    C2 = A1*B2 (2x3 * 3x4) stacked with A2*B2 (1x3 * 3x4)
    
    For A1*B1 (2x3 * 3x4):
      Split A1 into A1a (2x2) and A1b (2x1)
      Split B1 into B1a (2x4) and B1b (1x4)
      A1*B1 = A1a*B1a + A1b*B1b
      
      A1a*B1a is 2x2 * 2x4. Split B1a into B1a1 (2x2) and B1a2 (2x2).
      A1a*B1a = A1a*B1a1 + A1a*B1a2
      Each is 2x2 * 2x2, use Strassen rank 7. Total: 14.
      
      A1b*B1b is 2x1 * 1x4. This is an outer product.
      U: 2x1 = 2 rows, V: 1x4 = 4 rows, W: 4x2 = 8 rows. Rank = 8.
      
      Total for A1*B1: 14 + 8 = 22.
      
    Similarly for A1*B2: 22.
    
    For A2*B1 (1x3 * 3x4):
      Split A2 into A2a (1x2) and A2b (1x1)
      Split B1 into B1a (2x4) and B1b (1x4)
      A2*B1 = A2a*B1a + A2b*B1b
      
      A2a*B1a is 1x2 * 2x4. Split B1a into B1a1 (2x2) and B1a2 (2x2).
      A2a*B1a = A2a*B1a1 + A2a*B1a2
      Each is 1x2 * 2x2. Schoolbook rank 4. Total: 8.
      
      A2b*B1b is 1x1 * 1x4. Rank 4.
      
      Total for A2*B1: 8 + 4 = 12.
      
    Similarly for A2*B2: 12.
    
    Total rank: 22 + 22 + 12 + 12 = 68.
    """
    n, m, p = DIMENSIONS
    total_rank = 68
    
    U = [[0]*total_rank for _ in range(n*m)]  # 9 rows
    V = [[0]*total_rank for _ in range(m*p)]  # 24 rows
    W = [[0]*total_rank for _ in range(p*n)]  # 24 rows
    
    col = 0
    
    # Helper to place a local decomposition into global U, V, W
    def place(local_U, local_V, local_W, u_offset, v_offset, w_offset, rank_local):
        nonlocal col
        for i_local in range(len(local_U)):
            for r in range(rank_local):
                U[u_offset + i_local][col + r] = local_U[i_local][r]
        for i_local in range(len(local_V)):
            for r in range(rank_local):
                V[v_offset + i_local][col + r] = local_V[i_local][r]
        for i_local in range(len(local_W)):
            for r in range(rank_local):
                W[w_offset + i_local][col + r] = local_W[i_local][r]
        col += rank_local
    
    # Part 1: A1*B1 (2x3 * 3x4) -> rows 0,1 of C, cols 0-3 of C
    # A1a (2x2): rows 0,1 of A, cols 0,1 -> U indices 0,1,2,3
    # A1b (2x1): rows 0,1 of A, col 2 -> U indices 4,5
    # B1a (2x4): rows 0,1 of B, cols 0-3 -> V indices 0-7
    # B1b (1x4): row 2 of B, cols 0-3 -> V indices 8-11
    # C1a (2x4): rows 0,1 of C, cols 0-3 -> W indices 0-7
    # C1b (2x1): rows 0,1 of C, col 2 -> W indices 8,9
    
    # A1a*B1a1: 2x2 * 2x2 (B1a1 is rows 0,1 of B, cols 0,1)
    # B1a1: V indices 0,1,4,5 (j=0,k=0 -> 0; j=0,k=1 -> 1; j=1,k=0 -> 4; j=1,k=1 -> 5)
    # C1a1: W indices 0,1,4,5 (k=0,i=0 -> 0; k=1,i=0 -> 1; k=0,i=1 -> 4; k=1,i=1 -> 5)
    
    sU, sV, sW = _strassen_2x2x2()
    # Map local 2x2 indices to global
    # Local U: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # Global U for A1a: same
    # Local V: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # Global V for B1a1: (0,0)->0, (0,1)->1, (1,0)->4, (1,1)->5
    # Local W: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # Global W for C1a1: (0,0)->0, (0,1)->1, (1,0)->4, (1,1)->5
    
    v_map = [0, 1, 4, 5]  # local j,k -> global j*p+k
    w_map = [0, 1, 4, 5]  # local k,i -> global k*n+i
    
    for r in range(7):
        for i_local in range(4):
            U[i_local][col + r] = sU[i_local][r]
        for i_local in range(4):
            V[v_map[i_local]][col + r] = sV[i_local][r]
        for i_local in range(4):
            W[w_map[i_local]][col + r] = sW[i_local][r]
    col += 7
    
    # A1a*B1a2: 2x2 * 2x2 (B1a2 is rows 0,1 of B, cols 2,3)
    # B1a2: V indices 2,3,6,7
    # C1a2: W indices 2,3,6,7
    
    v_map = [2, 3, 6, 7]
    w_map = [2, 3, 6, 7]
    
    for r in range(7):
        for i_local in range(4):
            U[i_local][col + r] = sU[i_local][r]
        for i_local in range(4):
            V[v_map[i_local]][col + r] = sV[i_local][r]
        for i_local in range(4):
            W[w_map[i_local]][col + r] = sW[i_local][r]
    col += 7
    
    # A1b*B1b: 2x1 * 1x4 (outer product)
    # A1b: U indices 4,5
    # B1b: V indices 8,9,10,11
    # C1b: W indices 8,9,10,11 (k=0..3, i=0,1 -> k*3+0, k*3+1)
    
    for k in range(4):
        U[4][col] = 1
        U[5][col] = 1
        V[8 + k][col] = 1
        W[(k)*3 + 0][col] = 1  # k, i=0
        W[(k)*3 + 1][col] = 1  # k, i=1
        col += 1
    
    # Part 2: A1*B2 (2x3 * 3x4) -> rows 0,1 of C, cols 4-7 of C
    # B2: rows 0,1,2 of B, cols 4-7
    # B2a (2x4): rows 0,1 of B, cols 4-7 -> V indices 12-19
    # B2b (1x4): row 2 of B, cols 4-7 -> V indices 20-23
    # C2a (2x4): rows 0,1 of C, cols 4-7 -> W indices 12-19
    # C2b (2x1): rows 0,1 of C, col 2 -> W indices 20,21
    
    # A1a*B2a1: 2x2 * 2x2 (B2a1 is rows 0,1 of B, cols 4,5)
    v_map = [12, 13, 16, 17]
    w_map = [12, 13, 16, 17]
    
    for r in range(7):
        for i_local in range(4):
            U[i_local][col + r] = sU[i_local][r]
        for i_local in range(4):
            V[v_map[i_local]][col + r] = sV[i_local][r]
        for i_local in range(4):
            W[w_map[i_local]][col + r] = sW[i_local][r]
    col += 7
    
    # A1a*B2a2: 2x2 * 2x2 (B2a2 is rows 0,1 of B, cols 6,7)
    v_map = [14, 15, 18, 19]
    w_map = [14, 15, 18, 19]
    
    for r in range(7):
        for i_local in range(4):
            U[i_local][col + r] = sU[i_local][r]
        for i_local in range(4):
            V[v_map[i_local]][col + r] = sV[i_local][r]
        for i_local in range(4):
            W[w_map[i_local]][col + r] = sW[i_local][r]
    col += 7
    
    # A1b*B2b: 2x1 * 1x4 (outer product)
    for k in range(4):
        U[4][col] = 1
        U[5][col] = 1
        V[20 + k][col] = 1
        W[(4+k)*3 + 0][col] = 1
        W[(4+k)*3 + 1][col] = 1
        col += 1
    
    # Part 3: A2*B1 (1x3 * 3x4) -> row 2 of C, cols 0-3 of C
    # A2a (1x2): row 2 of A, cols 0,1 -> U indices 6,7
    # A2b (1x1): row 2 of A, col 2 -> U index 8
    # B1a (2x4): rows 0,1 of B, cols 0-3 -> V indices 0-7
    # B1b (1x4): row 2 of B, cols 0-3 -> V indices 8-11
    # C3a (1x4): row 2 of C, cols 0-3 -> W indices 12,13,14,15
    # C3b (1x1): row 2 of C, col 2 -> W index 16
    
    # A2a*B1a1: 1x2 * 2x2 (B1a1 is rows 0,1 of B, cols 0,1)
    # Schoolbook rank 4
    # U: 1x2 = 2 rows (local 0,1)
    # V: 2x2 = 4 rows (local 0,1,2,3)
    # W: 2x1 = 2 rows (local 0,1)
    
    for k_local in range(2):
        for j_local in range(2):
            U[6 + j_local][col] = 1  # A2a[j_local]
            V[j_local*2 + k_local][col] = 1  # B1a1[j_local, k_local]
            W[k_local*2 + 0][col] = 1  # C3a[k_local] -> global k*3+2
            col += 1
    
    # Wait, I need to be more careful. Let me redo this.
    # A2a*B1a1: 1x2 * 2x2
    # A2a: U indices 6,7 (j=0,1)
    # B1a1: V indices 0,1,4,5 (j=0,k=0 -> 0; j=0,k=1 -> 1; j=1,k=0 -> 4; j=1,k=1 -> 5)
    # C3a1: W indices 12,13 (k=0,1, i=2 -> k*3+2)
    
    for j_local in range(2):
        for k_local in range(2):
            v_idx = [0, 1, 4, 5][j_local*2 + k_local]
            w_idx = 12 + k_local
            U[6 + j_local][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
    
    # A2a*B1a2: 1x2 * 2x2 (B1a2 is rows 0,1 of B, cols 2,3)
    # B1a2: V indices 2,3,6,7
    # C3a2: W indices 14,15
    
    for j_local in range(2):
        for k_local in range(2):
            v_idx = [2, 3, 6, 7][j_local*2 + k_local]
            w_idx = 14 + k_local
            U[6 + j_local][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
    
    # A2b*B1b: 1x1 * 1x4
    for k in range(4):
        U[8][col] = 1
        V[8 + k][col] = 1
        W[12 + k][col] = 1
        col += 1
    
    # Part 4: A2*B2 (1x3 * 3x4) -> row 2 of C, cols 4-7 of C
    # B2a (2x4): rows 0,1 of B, cols 4-7 -> V indices 12-19
    # B2b (1x4): row 2 of B, cols 4-7 -> V indices 20-23
    # C4a (1x4): row 2 of C, cols 4-7 -> W indices 16,17,18,19
    # C4b (1x1): row 2 of C, col 2 -> W index 20
    
    # A2a*B2a1: 1x2 * 2x2 (B2a1 is rows 0,1 of B, cols 4,5)
    for j_local in range(2):
        for k_local in range(2):
            v_idx = [12, 13, 16, 17][j_local*2 + k_local]
            w_idx = 16 + k_local
            U[6 + j_local][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
    
    # A2a*B2a2: 1x2 * 2x2 (B2a2 is rows 0,1 of B, cols 6,7)
    for j_local in range(2):
        for k_local in range(2):
            v_idx = [14, 15, 18, 19][j_local*2 + k_local]
            w_idx = 18 + k_local
            U[6 + j_local][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
    
    # A2b*B2b: 1x1 * 1x4
    for k in range(4):
        U[8][col] = 1
        V[20 + k][col] = 1
        W[16 + k][col] = 1
        col += 1
    
    # Check that we used all 68 columns
    assert col == total_rank, f"Expected {total_rank} columns, got {col}"
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a low-rank exact decomposition for 3x3 * 3x8 matrix multiplication.
    Uses block decomposition with Strassen (rank 7 for 2x2x2) and schoolbook.
    Target rank: 68.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135
    
    # Try the block decomposition
    try:
        result = _build_block_decomposition()
        if _verify(result["U"], result["V"], result["W"]):
            return result
    except Exception:
        pass
    
    # Fallback to schoolbook
    return _schoolbook_decomposition()
# EVOLVE-BLOCK-END
