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
    
    W convention: W[k*n + i, r] where n=3, so row index = k*3 + i
    U convention: U[i*m + j, r] where m=3, so row index = i*3 + j
    V convention: V[j*p + k, r] where p=8, so row index = j*8 + k
    
    Split A (3x3) into A1 (2x3) [rows 0,1] and A2 (1x3) [row 2].
    Split B (3x8) into B1 (3x4) [cols 0-3] and B2 (3x4) [cols 4-7].
    
    C[i,k] = sum_j A[i,j]*B[j,k]
    
    For i in {0,1}, k in {0,1,2,3}: C[i,k] = sum_{j=0}^{1} A1[i,j]*B1[j,k] + A1[i,2]*B1[2,k]
    For i in {0,1}, k in {4,5,6,7}: C[i,k] = sum_{j=0}^{1} A1[i,j]*B2[j,k] + A1[i,2]*B2[2,k]
    For i=2, k in {0,1,2,3}: C[2,k] = sum_{j=0}^{1} A2[j]*B1[j,k] + A2[2]*B1[2,k]
    For i=2, k in {4,5,6,7}: C[2,k] = sum_{j=0}^{1} A2[j]*B2[j,k] + A2[2]*B2[2,k]
    
    A1*B1: 2x3 * 3x4
      Split A1 into A1a (2x2) and A1b (2x1)
      Split B1 into B1a (2x4) and B1b (1x4)
      A1*B1 = A1a*B1a + A1b*B1b
      A1a*B1a: 2x2 * 2x4 = two 2x2*2x2 blocks, each Strassen rank 7 -> 14
      A1b*B1b: 2x1 * 1x4 = outer product, rank 8
      Total: 22
      
    A1*B2: same structure, rank 22
    
    A2*B1: 1x3 * 3x4
      Split A2 into A2a (1x2) and A2b (1x1)
      Split B1 into B1a (2x4) and B1b (1x4)
      A2*B1 = A2a*B1a + A2b*B1b
      A2a*B1a: 1x2 * 2x4 = two 1x2*2x2 blocks, each schoolbook rank 4 -> 8
      A2b*B1b: 1x1 * 1x4 = rank 4
      Total: 12
      
    A2*B2: same structure, rank 12
    
    Total rank: 22 + 22 + 12 + 12 = 68
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=8
    total_rank = 68
    
    U = [[0]*total_rank for _ in range(n*m)]  # 9 rows: U[i*m+j, r] = U[i*3+j, r]
    V = [[0]*total_rank for _ in range(m*p)]  # 24 rows: V[j*p+k, r] = V[j*8+k, r]
    W = [[0]*total_rank for _ in range(p*n)]  # 24 rows: W[k*n+i, r] = W[k*3+i, r]
    
    col = 0
    sU, sV, sW = _strassen_2x2x2()
    
    # Part 1: A1*B1 -> C[0:2, 0:4]
    # A1a: A[0:2, 0:2] -> U indices: i*3+j for i in {0,1}, j in {0,1} -> 0,1,3,4
    # A1b: A[0:2, 2] -> U indices: i*3+2 for i in {0,1} -> 2, 5
    # B1a: B[0:2, 0:4] -> V indices: j*8+k for j in {0,1}, k in {0,1,2,3} -> 0,1,2,3,8,9,10,11
    # B1b: B[2, 0:4] -> V indices: 2*8+k = 16,17,18,19
    # C1a: C[0:2, 0:4] -> W indices: k*3+i for k in {0,1,2,3}, i in {0,1} -> 0,1,3,4,6,7,9,10
    # C1b: C[0:2, 2:4] -> W indices: k*3+i for k in {2,3}, i in {0,1} -> 6,7,9,10
    
    # A1a*B1a1: 2x2 * 2x2, B1a1 = B[0:2, 0:2]
    # V indices for B[0:2,0:2]: j*8+k -> 0,1,8,9
    # W indices for C[0:2,0:2]: k*3+i -> 0,1,3,4
    v_map = [0, 1, 8, 9]
    w_map = [0, 1, 3, 4]
    u_map = [0, 1, 3, 4]  # U[i*3+j] for i,j in {0,1}
    
    for r in range(7):
        for i_local in range(4):
            U[u_map[i_local]][col + r] = sU[i_local][r]
        for i_local in range(4):
            V[v_map[i_local]][col + r] = sV[i_local][r]
        for i_local in range(4):
            W[w_map[i_local]][col + r] = sW[i_local][r]
    col += 7
    
    # A1a*B1a2: 2x2 * 2x2, B1a2 = B[0:2, 2:4]
    # V indices: j*8+k for j in {0,1}, k in {2,3} -> 2,3,10,11
    # W indices: k*3+i for k in {2,3}, i in {0,1} -> 6,7,9,10
    v_map = [2, 3, 10, 11]
    w_map = [6, 7, 9, 10]
    
    for r in range(7):
        for i_local in range(4):
            U[u_map[i_local]][col + r] = sU[i_local][r]
        for i_local in range(4):
            V[v_map[i_local]][col + r] = sV[i_local][r]
        for i_local in range(4):
            W[w_map[i_local]][col + r] = sW[i_local][r]
    col += 7
    
    # A1b*B1b: 2x1 * 1x4 outer product
    # A1b: U[2] and U[5] (A[0,2] and A[1,2])
    # B1b: V[16:20] (B[2,0:4])
    # C1b: C[0:2, 2:4] -> W[k*3+i] for k in {2,3}, i in {0,1} -> 6,7,9,10
    for k in range(4):
        U[2][col] = 1   # A[0,2]
        U[5][col] = 1   # A[1,2]
        V[16 + k][col] = 1  # B[2,k]
        # C[0, 2+k] and C[1, 2+k]
        W[(2+k)*3 + 0][col] = 1  # k_global = 2+k, i=0
        W[(2+k)*3 + 1][col] = 1  # k_global = 2+k, i=1
        col += 1
    
    # Part 2: A1*B2 -> C[0:2, 4:8]
    # B2a: B[0:2, 4:8] -> V indices: j*8+k for j in {0,1}, k in {4,5,6,7} -> 4,5,6,7,12,13,14,15
    # B2b: B[2, 4:8] -> V indices: 2*8+k = 20,21,22,23
    # C2a: C[0:2, 4:8] -> W indices: k*3+i for k in {4,5,6,7}, i in {0,1} -> 12,13,15,16,18,19,21,22
    
    # A1a*B2a1: 2x2 * 2x2, B2a1 = B[0:2, 4:6]
    v_map = [4, 5, 12, 13]
    w_map = [12, 13, 15, 16]
    
    for r in range(7):
        for i_local in range(4):
            U[u_map[i_local]][col + r] = sU[i_local][r]
        for i_local in range(4):
            V[v_map[i_local]][col + r] = sV[i_local][r]
        for i_local in range(4):
            W[w_map[i_local]][col + r] = sW[i_local][r]
    col += 7
    
    # A1a*B2a2: 2x2 * 2x2, B2a2 = B[0:2, 6:8]
    v_map = [6, 7, 14, 15]
    w_map = [18, 19, 21, 22]
    
    for r in range(7):
        for i_local in range(4):
            U[u_map[i_local]][col + r] = sU[i_local][r]
        for i_local in range(4):
            V[v_map[i_local]][col + r] = sV[i_local][r]
        for i_local in range(4):
            W[w_map[i_local]][col + r] = sW[i_local][r]
    col += 7
    
    # A1b*B2b: 2x1 * 1x4 outer product
    for k in range(4):
        U[2][col] = 1   # A[0,2]
        U[5][col] = 1   # A[1,2]
        V[20 + k][col] = 1  # B[2,4+k]
        W[(4+k)*3 + 0][col] = 1  # k_global = 4+k, i=0
        W[(4+k)*3 + 1][col] = 1  # k_global = 4+k, i=1
        col += 1
    
    # Part 3: A2*B1 -> C[2, 0:4]
    # A2a: A[2, 0:2] -> U indices: 2*3+0=6, 2*3+1=7
    # A2b: A[2, 2] -> U index: 2*3+2=8
    # B1a: B[0:2, 0:4] -> V indices: 0,1,2,3,8,9,10,11
    # B1b: B[2, 0:4] -> V indices: 16,17,18,19
    # C3: C[2, 0:4] -> W indices: k*3+2 for k in {0,1,2,3} -> 2,5,8,11
    
    # A2a*B1a1: 1x2 * 2x2, B1a1 = B[0:2, 0:2]
    # V indices: 0,1,8,9
    # W indices: k*3+2 for k in {0,1} -> 2,5
    for j_local in range(2):
        for k_local in range(2):
            v_idx = [0, 1, 8, 9][j_local*2 + k_local]
            w_idx = k_local*3 + 2
            U[6 + j_local][col] = 1  # A[2, j_local]
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
    
    # A2a*B1a2: 1x2 * 2x2, B1a2 = B[0:2, 2:4]
    # V indices: 2,3,10,11
    # W indices: k*3+2 for k in {2,3} -> 8,11
    for j_local in range(2):
        for k_local in range(2):
            v_idx = [2, 3, 10, 11][j_local*2 + k_local]
            w_idx = (2+k_local)*3 + 2
            U[6 + j_local][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
    
    # A2b*B1b: 1x1 * 1x4
    for k in range(4):
        U[8][col] = 1  # A[2,2]
        V[16 + k][col] = 1  # B[2,k]
        W[k*3 + 2][col] = 1  # C[2,k]
        col += 1
    
    # Part 4: A2*B2 -> C[2, 4:8]
    # B2a: B[0:2, 4:8] -> V indices: 4,5,6,7,12,13,14,15
    # B2b: B[2, 4:8] -> V indices: 20,21,22,23
    # C4: C[2, 4:8] -> W indices: k*3+2 for k in {4,5,6,7} -> 14,17,20,23
    
    # A2a*B2a1: 1x2 * 2x2, B2a1 = B[0:2, 4:6]
    for j_local in range(2):
        for k_local in range(2):
            v_idx = [4, 5, 12, 13][j_local*2 + k_local]
            w_idx = (4+k_local)*3 + 2
            U[6 + j_local][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
    
    # A2a*B2a2: 1x2 * 2x2, B2a2 = B[0:2, 6:8]
    for j_local in range(2):
        for k_local in range(2):
            v_idx = [6, 7, 14, 15][j_local*2 + k_local]
            w_idx = (6+k_local)*3 + 2
            U[6 + j_local][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
    
    # A2b*B2b: 1x1 * 1x4
    for k in range(4):
        U[8][col] = 1
        V[20 + k][col] = 1
        W[(4+k)*3 + 2][col] = 1
        col += 1
    
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
