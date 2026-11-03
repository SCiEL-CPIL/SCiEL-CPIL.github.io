"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                if abs(actual - expected) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2():
    """
    Returns U, V, W for the rank-7 Strassen decomposition of 2x2x2.
    U is 4x7, V is 4x7, W is 4x7.
    Indices: U[row, col], V[row, col], W[col, row] in standard tensor notation?
    Here: U[i,j], V[j,k], W[k,i].
    """
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b01-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b10)
    # M7 = (a01-a11)(b01+b11)
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1
    
    # M7
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _construct_rank44():
    """
    Constructs a rank-44 decomposition for 3x4x4.
    Strategy:
    1. Split n=3 into 2+1.
    2. The 2x4x4 part is decomposed into four 2x2x2 blocks using Strassen (rank 7 each).
       Total rank for 2x4x4 part = 4 * 7 = 28.
    3. The 1x4x4 part is decomposed using schoolbook (rank 16).
    Total rank = 28 + 16 = 44.
    """
    n, m, p = 3, 4, 4
    rank_total = 44
    U = np.zeros((n * m, rank_total))
    V = np.zeros((m * p, rank_total))
    W = np.zeros((p * n, rank_total))
    
    # Part 1: 2x4x4 (rows 0,1 of A)
    # Split m=4 into 2+2 and p=4 into 2+2.
    # This gives 4 blocks of 2x2x2.
    sU, sV, sW = _strassen_2x2x2()
    
    col = 0
    for j_block in range(2): # j in {0,1} or {2,3}
        j0 = 2 * j_block
        j1 = j0 + 1
        for k_block in range(2): # k in {0,1} or {2,3}
            k0 = 2 * k_block
            k1 = k0 + 1
            
            # Block is A[0:2, j0:j1] * B[j0:j1, k0:k1] -> C[0:2, k0:k1]
            for s in range(7):
                c = col + s
                
                # U indices: (i, j) where i in {0,1}, j in {j0, j1}
                # sU indices: (i_sub, s) where i_sub in {0,1} corresponding to i=0,1
                U[0 * m + j0, c] = sU[0, s]
                U[0 * m + j1, c] = sU[1, s]
                U[1 * m + j0, c] = sU[2, s]
                U[1 * m + j1, c] = sU[3, s]
                
                # V indices: (j, k) where j in {j0, j1}, k in {k0, k1}
                V[j0 * p + k0, c] = sV[0, s]
                V[j0 * p + k1, c] = sV[1, s]
                V[j1 * p + k0, c] = sV[2, s]
                V[j1 * p + k1, c] = sV[3, s]
                
                # W indices: (k, i) where k in {k0, k1}, i in {0,1}
                W[k0 * n + 0, c] = sW[0, s]
                W[k0 * n + 1, c] = sW[1, s]
                W[k1 * n + 0, c] = sW[2, s]
                W[k1 * n + 1, c] = sW[3, s]
                
                col += 1
    
    # Part 2: 1x4x4 (row 2 of A)
    # Schoolbook: 16 terms
    for j in range(4):
        for k in range(4):
            c = col
            U[2 * m + j, c] = 1
            V[j * p + k, c] = 1
            W[k * n + 2, c] = 1
            col += 1
            
    assert col == rank_total, f"Rank mismatch: {col} vs {rank_total}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    best = _schoolbook(n, m, p)
    best_rank = len(best["U"][0])
    
    try:
        candidate = _construct_rank44()
        if candidate and _verify(candidate["U"], candidate["V"], candidate["W"], n, m, p):
            rank = len(candidate["U"][0])
            if rank < best_rank:
                best = candidate
                best_rank = rank
    except Exception:
        pass
    
    return best
# EVOLVE-BLOCK-END
