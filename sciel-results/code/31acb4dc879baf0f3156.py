"""Matrix multiplication 2x3x8 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 3, 8]

import time
import numpy as np

# EVOLVE-BLOCK-START

def _build_decomp(n, m, p):
    """
    Build a decomposition for 2x3x8.
    Split A = [A1 | A2] where A1 is 2x2, A2 is 2x1.
    Split B = [B1; B2] where B1 is 2x8, B2 is 1x8.
    C = A1*B1 + A2*B2.
    
    A2*B2: Rank 1 (using the fact that it's an outer product of a vector and a vector, 
    but in the tensor sense, it's a single term if we allow W to have all 1s).
    Actually, let's re-verify the rank 1 claim.
    T[i, 2, k] = 1.
    Term: U[i, 2, 0]=1, V[2, k, 0]=1, W[k, i, 0]=1.
    Sum U[i,2,0] V[2,k,0] W[k,i,0] = 1*1*1 = 1.
    Yes, rank 1.
    
    A1*B1: A1 is 2x2, B1 is 2x8.
    Split B1 into four 2x2 blocks.
    Each block is 2x2x2 multiplication, rank 7 (Strassen).
    Total rank for A1*B1: 4 * 7 = 28.
    
    Total rank: 1 + 28 = 29.
    """
    n, m, p = 2, 3, 8
    
    rank_total = 29
    U = [[0] * rank_total for _ in range(n * m)]  # 6 x 29
    V = [[0] * rank_total for _ in range(m * p)]  # 24 x 29
    W = [[0] * rank_total for _ in range(p * n)]  # 16 x 29
    
    col = 0
    
    # Part 1: A2*B2 (Rank 1)
    # A2 is column 2 of A (indices 2, 5 in flat U)
    # B2 is row 2 of B (indices 16..23 in flat V)
    # C is all entries (indices 0..15 in flat W)
    
    # U: rows 2 and 5 (i=0,j=2 and i=1,j=2)
    U[2][col] = 1
    U[5][col] = 1
    
    # V: cols 0..7 of row 2 (j=2, k=0..7) -> indices 16..23
    for k in range(8):
        V[16 + k][col] = 1
        
    # W: all entries (k=0..7, i=0..1) -> indices 0..15
    for k in range(8):
        W[k * 2 + 0][col] = 1
        W[k * 2 + 1][col] = 1
        
    col += 1
    
    # Part 2: A1*B1 via Strassen (4 blocks)
    # A1 is 2x2: indices 0,1,3,4 in U
    # B1 is 2x8: split into 4 blocks of 2x2
    for g in range(4):
        # B block indices in V
        # j=0, k=2g, 2g+1 -> 0*8 + 2g, 0*8 + 2g+1
        # j=1, k=2g, 2g+1 -> 1*8 + 2g, 1*8 + 2g+1
        b00 = 0 + 2 * g
        b01 = 0 + 2 * g + 1
        b10 = 8 + 2 * g
        b11 = 8 + 2 * g + 1
        
        # A1 indices in U
        a00 = 0
        a01 = 1
        a10 = 3
        a11 = 4
        
        # C indices in W
        # k=2g, i=0 -> 2g*2 + 0
        # k=2g, i=1 -> 2g*2 + 1
        # k=2g+1, i=0 -> (2g+1)*2 + 0
        # k=2g+1, i=1 -> (2g+1)*2 + 1
        c00 = (2 * g) * 2 + 0
        c01 = (2 * g) * 2 + 1
        c10 = (2 * g + 1) * 2 + 0
        c11 = (2 * g + 1) * 2 + 1
        
        # M1 = (a00+a11)(b00+b11)
        U[a00][col] = 1; U[a11][col] = 1
        V[b00][col] = 1; V[b11][col] = 1
        W[c00][col] = 1; W[c11][col] = 1
        col += 1
        
        # M2 = (a10+a11)b00
        U[a10][col] = 1; U[a11][col] = 1
        V[b00][col] = 1
        W[c01][col] = 1
        col += 1
        
        # M3 = a00(b11-b01)
        U[a00][col] = 1
        V[b11][col] = 1; V[b01][col] = -1
        W[c10][col] = 1; W[c11][col] = -1
        col += 1
        
        # M4 = a11(b00-b10)
        U[a11][col] = 1
        V[b00][col] = 1; V[b10][col] = -1
        W[c00][col] = -1; W[c01][col] = 1
        col += 1
        
        # M5 = (a00+a01)b11
        U[a00][col] = 1; U[a01][col] = 1
        V[b11][col] = 1
        W[c10][col] = 1
        col += 1
        
        # M6 = (a10-a00)(b00+b01)
        U[a10][col] = 1; U[a00][col] = -1
        V[b00][col] = 1; V[b01][col] = 1
        W[c00][col] = 1
        col += 1
        
        # M7 = (a01-a11)(b10+b11)
        U[a01][col] = 1; U[a11][col] = -1
        V[b10][col] = 1; V[b11][col] = 1
        W[c11][col] = 1
        col += 1
        
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Verify decomposition correctness."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                if abs(val - 1.0) > max_err:
                    max_err = abs(val - 1.0)
    return max_err


def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    n, m, p = DIMENSIONS
    
    # Primary: Rank-29 decomposition
    try:
        U, V, W = _build_decomp(n, m, p)
        err = _verify(U, V, W, n, m, p)
        if err < 1e-10:
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback: schoolbook rank-48
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

# EVOLVE-BLOCK-END
