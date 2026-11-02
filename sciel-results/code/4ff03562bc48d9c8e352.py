"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces the tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Check valid terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U[a] * V[b] * W[c])
                if abs(val - 1.0) > 1e-9:
                    return False
    
    # Check spurious terms
    valid = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid.add((i * m + j, j * p + k, k * n + i))
    
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                if (a, b, c) not in valid:
                    val = np.sum(U[a] * V[b] * W[c])
                    if abs(val) > 1e-9:
                        return False
    return True

def _rank32_decomposition(n, m, p):
    """
    Rank-32 decomposition: 2 terms per column k of B.
    For each k: one term handles A[:,0]*B[0,k], another handles A[:,1]*B[1,k].
    """
    rank = 2 * p  # 32
    U = [[0] * rank for _ in range(n * m)]  # 4 x 32
    V = [[0] * rank for _ in range(m * p)]  # 32 x 32
    W = [[0] * rank for _ in range(p * n)]  # 32 x 32
    
    for k in range(p):
        col1 = 2 * k
        U[0][col1] = 1
        U[2][col1] = 1
        V[k][col1] = 1
        W[2 * k][col1] = 1
        W[2 * k + 1][col1] = 1
        
        col2 = 2 * k + 1
        U[1][col2] = 1
        U[3][col2] = 1
        V[p + k][col2] = 1
        W[2 * k][col2] = 1
        W[2 * k + 1][col2] = 1
    
    return U, V, W

def _strassen_shared(n, m, p):
    """
    Attempt to use Strassen's 7-multiplication scheme but exploit that A is shared
    across all 16 columns. Instead of 8 independent 2x2x2 blocks (rank 56),
    try to find a decomposition where the A-side coefficients are reused.
    
    Key idea: For each pair of columns (2t, 2t+1), we have a 2x2x2 sub-tensor.
    Strassen uses 7 scalar multiplications per block. But since A is the same
    for all blocks, we can precompute the 7 Strassen combinations of A once,
    and then for each block we only need to multiply by the corresponding
    combinations of B.
    
    The 7 Strassen products for a 2x2x2 block are:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B01-B11)
    M4 = A11(B10-B00)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B01)
    M7 = (A01-A11)(B10+B11)
    
    The A-side combinations are:
    S1 = A00+A11
    S2 = A10+A11
    S3 = A00
    S4 = A11
    S5 = A00+A01
    S6 = A10-A00
    S7 = A01-A11
    
    These are fixed linear combinations of A entries. For each block t,
    we need 7 multiplications: S_i * (B-combination for block t, i).
    
    In a tensor decomposition, each rank-1 term corresponds to one multiplication.
    The A-side coefficient vector is fixed (same S_i for all blocks), but the
    B-side and C-side vary per block.
    
    So we can have 7 "A-patterns" and for each block, 7 terms with the same
    A-pattern but different B and C entries. This gives 7*8 = 56 terms.
    
    UNLESS we can combine blocks! If two blocks share the same B-combination
    pattern (which they don't since B entries are different), we could merge.
    
    However, there's another approach: use the fact that we only need C[0,k] and C[1,k]
    for each k (not C[0,k], C[1,k], C[0,k+1], C[1,k+1] as a 2x2 output).
    
    Wait, let me re-examine. For block t (columns 2t, 2t+1 of B):
    Output C[0,2t], C[1,2t], C[0,2t+1], C[1,2t+1]
    
    In the trace convention: c = k*n + i = k*2 + i
    C[0,2t] -> c = 4t, C[1,2t] -> c = 4t+1
    C[0,2t+1] -> c = 4t+2, C[1,2t+1] -> c = 4t+3
    
    So for block t, the 4 output entries are at positions 4t, 4t+1, 4t+2, 4t+3.
    This IS a full 2x2 output, so Strassen applies directly.
    
    The rank is 7 per block * 8 blocks = 56. To improve, we need to find
    a decomposition of the full 2x2x16 tensor with rank < 56.
    
    Lower bound considerations: The tensor has 4*32*32 = 4096 entries, 64 of which
    are 1 and the rest 0. The border rank and other invariants might give lower bounds.
    
    For now, let's stick with rank 56 via Strassen and see if we can find rank < 56
    by a different construction.
    """
    num_blocks = p // 2  # 8
    rank = num_blocks * 7  # 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for t in range(num_blocks):
        b00 = 2 * t
        b01 = 2 * t + 1
        b10 = 16 + 2 * t
        b11 = 16 + 2 * t + 1
        c00 = 4 * t
        c01 = 4 * t + 2
        c10 = 4 * t + 1
        c11 = 4 * t + 3
        
        # M1 = (A00+A11)(B00+B11)
        U[0][col] += 1; U[3][col] += 1
        V[b00][col] += 1; V[b11][col] += 1
        W[c00][col] += 1; W[c11][col] += 1
        col += 1
        # M2 = (A10+A11)B00
        U[2][col] += 1; U[3][col] += 1
        V[b00][col] += 1
        W[c10][col] += 1; W[c11][col] -= 1
        col += 1
        # M3 = A00(B01-B11)
        U[0][col] += 1
        V[b01][col] += 1; V[b11][col] -= 1
        W[c01][col] += 1; W[c11][col] += 1
        col += 1
        # M4 = A11(B10-B00)
        U[3][col] += 1
        V[b10][col] += 1; V[b00][col] -= 1
        W[c00][col] += 1; W[c10][col] += 1
        col += 1
        # M5 = (A00+A01)B11
        U[0][col] += 1; U[1][col] += 1
        V[b11][col] += 1
        W[c00][col] -= 1; W[c01][col] += 1
        col += 1
        # M6 = (A10-A00)(B00+B01)
        U[2][col] += 1; U[0][col] -= 1
        V[b00][col] += 1; V[b01][col] += 1
        W[c11][col] += 1
        col += 1
        # M7 = (A01-A11)(B10+B11)
        U[1][col] += 1; U[3][col] -= 1
        V[b10][col] += 1; V[b11][col] += 1
        W[c00][col] += 1
        col += 1
    
    return U, V, W

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115
    
    # Try rank-32 decomposition first
    U, V, W = _rank32_decomposition(n, m, p)
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fall back to Strassen (rank 56)
    if time.time() < deadline:
        U, V, W = _strassen_shared(n, m, p)
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    # Final fallback: schoolbook (rank 64)
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
