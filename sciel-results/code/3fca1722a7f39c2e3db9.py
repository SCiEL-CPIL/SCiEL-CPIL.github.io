"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2():
    """
    Return Strassen decomposition for 2x2 * 2x2 -> 2x2.
    U shape (4, 7), V shape (4, 7), W shape (4, 7).
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c].
    
    Strassen's 7 multiplications:
    M1 = (A11+A22)*(B11+B22)
    M2 = (A21+A22)*B11
    M3 = A11*(B12-B22)
    M4 = A22*(B21-B11)
    M5 = (A11+A12)*B22
    M6 = (A21-A11)*(B11+B12)
    M7 = (A12-A22)*(B21+B22)
    
    Output:
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    rank = 7
    U = np.zeros((4, rank), dtype=float)
    V = np.zeros((4, rank), dtype=float)
    W = np.zeros((4, rank), dtype=float)
    
    # A indices: A11=0, A12=1, A21=2, A22=3
    # B indices: B11=0, B12=1, B21=2, B22=3
    # C indices: C11=0, C12=1, C21=2, C22=3
    
    # M1 = (A11+A22)*(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1  # A11, A22
    V[0, 0] = 1; V[3, 0] = 1  # B11, B22
    W[0, 0] = 1; W[3, 0] = 1  # C11, C22
    
    # M2 = (A21+A22)*B11
    U[2, 1] = 1; U[3, 1] = 1  # A21, A22
    V[0, 1] = 1  # B11
    W[2, 1] = 1; W[3, 1] = -1  # C21, -C22
    
    # M3 = A11*(B12-B22)
    U[0, 2] = 1  # A11
    V[1, 2] = 1; V[3, 2] = -1  # B12, -B22
    W[1, 2] = 1; W[3, 2] = 1  # C12, C22
    
    # M4 = A22*(B21-B11)
    U[3, 3] = 1  # A22
    V[2, 3] = 1; V[0, 3] = -1  # B21, -B11
    W[0, 3] = 1; W[2, 3] = 1  # C11, C21
    
    # M5 = (A11+A12)*B22
    U[0, 4] = 1; U[1, 4] = 1  # A11, A12
    V[3, 4] = 1  # B22
    W[0, 4] = -1; W[1, 4] = 1  # -C11, C12
    
    # M6 = (A21-A11)*(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1  # A21, -A11
    V[0, 5] = 1; V[1, 5] = 1  # B11, B12
    W[3, 5] = 1  # C22
    
    # M7 = (A12-A22)*(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1  # A12, -A22
    V[2, 6] = 1; V[3, 6] = 1  # B21, B22
    W[0, 6] = 1  # C11
    
    return U, V, W

def construct():
    """
    Decompose 2x2 * 2x12 multiplication using Strassen on each 2x2x2 block.
    
    The 2x12 matrix B can be viewed as 6 independent 2x2 blocks:
    B[:, 0:2], B[:, 2:4], ..., B[:, 10:12]
    
    For each block b (0..5), we compute A * B[:, 2b:2b+2] using Strassen.
    
    Total rank = 6 * 7 = 42.
    
    Indexing:
    U: (n*m, rank) = (4, 42), U[i*m+j, c]
    V: (m*p, rank) = (24, 42), V[j*p+k, c]
    W: (p*n, rank) = (24, 42), W[k*n+i, c]
    
    For block b, the k indices are 2b and 2b+1.
    Strassen W indices: C11=k=0, C12=k=1, C21=k=2, C22=k=3 (within block)
    Global W index: k*n + i = (2b + local_k)*2 + i
    """
    n, m, p = DIMENSIONS
    num_blocks = p // 2  # 6 blocks
    block_rank = 7
    total_rank = num_blocks * block_rank  # 42
    
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    U_blk, V_blk, W_blk = _strassen_2x2()
    
    for b in range(num_blocks):
        base = b * block_rank
        # U is the same for all blocks (A is shared)
        U[:, base:base+block_rank] = U_blk
        
        # V: for block b, B indices are 2b (B11), 2b+1 (B12), 2b+2 (B21), 2b+3 (B22)
        # V index: j*p + k where j is row of B (0 or 1), k is column (0..11)
        # V_blk[j, c] corresponds to B[j, local_k]
        # local_k=0 -> global k=2b, local_k=1 -> global k=2b+1
        for j in range(m):  # j=0,1
            for c in range(block_rank):
                for local_k in range(2):
                    if V_blk[j * 2 + local_k, c] != 0:
                        V[j * p + (2 * b + local_k), base + c] = V_blk[j * 2 + local_k, c]
        
        # W: W[k*n + i, c] where k is global column, i is row of A
        # W_blk[local_k * 2 + i, c] where local_k=0,1, i=0,1
        # global k = 2*b + local_k
        for local_k in range(2):
            for i in range(n):
                for c in range(block_rank):
                    W[(2 * b + local_k) * n + i, base + c] = W_blk[local_k * 2 + i, c]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
