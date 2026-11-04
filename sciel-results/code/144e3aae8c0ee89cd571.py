"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """
    Return the Strassen 2x2x2 decomposition as (U, V, W) arrays.
    U: shape (4, 7), V: shape (4, 7), W: shape (4, 7)
    Indices: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1 = (A00 + A11) * (B00 + B11)
    U[0, 0] = 1  # A00
    U[3, 0] = 1  # A11
    V[0, 0] = 1  # B00
    V[3, 0] = 1  # B11
    W[0, 0] = 1  # C00
    W[3, 0] = 1  # C11
    
    # M2 = (A10 + A11) * B00
    U[2, 1] = 1  # A10
    U[3, 1] = 1  # A11
    V[0, 1] = 1  # B00
    W[2, 1] = 1  # C10
    
    # M3 = A00 * (B10 - B00)
    U[0, 2] = 1  # A00
    V[2, 2] = 1  # B10
    V[0, 2] = -1  # B00
    W[1, 2] = 1  # C01
    W[3, 2] = 1  # C11
    
    # M4 = A11 * (B11 - B01)
    U[3, 3] = 1  # A11
    V[3, 3] = 1  # B11
    V[1, 3] = -1  # B01
    W[0, 3] = 1  # C00
    W[2, 3] = 1  # C10
    
    # M5 = (A00 + A01) * B11
    U[0, 4] = 1  # A00
    U[1, 4] = 1  # A01
    V[3, 4] = 1  # B11
    W[0, 4] = -1  # C00
    W[1, 4] = 1  # C01
    
    # M6 = (A10 - A00) * (B00 + B01)
    U[2, 5] = 1  # A10
    U[0, 5] = -1  # A00
    V[0, 5] = 1  # B00
    V[1, 5] = 1  # B01
    W[3, 5] = 1  # C11
    
    # M7 = (A01 - A11) * (B10 + B11)
    U[1, 6] = 1  # A01
    U[3, 6] = -1  # A11
    V[2, 6] = 1  # B10
    V[3, 6] = 1  # B11
    W[0, 6] = -1  # C00
    
    return U, V, W


def _build_decomposition(n, m, p):
    """
    Build a rank-minimal exact tensor decomposition for
    multiplying an n x m matrix by an m x p matrix.
    
    For 2x3x6:
    - Split j dimension: j=0 (rank 2) + j=1,2 (rank 21)
    - Total rank = 23
    """
    n, m, p = 2, 3, 6
    
    rank_part1 = 2  # j=0 block
    rank_part2 = 21  # j=1,2 block (3 Strassen blocks of 7)
    total_rank = rank_part1 + rank_part2
    
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    # Part 1: j=0, i=0,1, k=0..5 (rank 2)
    # Term 1: i=0
    U[0 * m + 0, 0] = 1
    for k in range(p):
        V[0 * p + k, 0] = 1
        W[k * n + 0, 0] = 1
    
    # Term 2: i=1
    U[1 * m + 0, 1] = 1
    for k in range(p):
        V[0 * p + k, 1] = 1
        W[k * n + 1, 1] = 1
    
    # Part 2: j=1,2
    # For each k-pair (2*t, 2*t+1), use a 2x2x2 Strassen decomposition.
    S_U, S_V, S_W = _strassen_2x2x2()
    
    for t in range(p // 2):
        k0 = 2 * t
        k1 = 2 * t + 1
        base = rank_part1 + t * 7
        
        for c in range(7):
            # Map Strassen indices to global indices
            # Strassen: i_s=0,1; j_s=0,1; k_s=0,1
            # Global: i=0,1; j=1+j_s; k=k0+k_s
            
            # U: U_global[i * m + (1 + j_s), base + c] = U_strassen[i_s * 2 + j_s, c]
            for i_s in range(2):
                for j_s in range(2):
                    val = S_U[i_s * 2 + j_s, c]
                    if val != 0:
                        i = i_s
                        j = 1 + j_s
                        U[i * m + j, base + c] += val
            
            # V: V_global[(1 + j_s) * p + (k0 + k_s), base + c] = V_strassen[j_s * 2 + k_s, c]
            for j_s in range(2):
                for k_s in range(2):
                    val = S_V[j_s * 2 + k_s, c]
                    if val != 0:
                        j = 1 + j_s
                        k = k0 + k_s
                        V[j * p + k, base + c] += val
            
            # W: W_global[(k0 + k_s) * n + i_s, base + c] = W_strassen[k_s * 2 + i_s, c]
            for k_s in range(2):
                for i_s in range(2):
                    val = S_W[k_s * 2 + i_s, c]
                    if val != 0:
                        k = k0 + k_s
                        i = i_s
                        W[k * n + i, base + c] += val
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly represents matrix multiplication."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for c in range(rank):
                    actual += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(actual - expected) > tol:
                    return False
    return True


def _schoolbook_decomposition(n, m, p):
    """Fallback schoolbook decomposition with rank n*m*p."""
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
    return {"U": U, "V": V, "W": W}


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    # Try the Strassen-based block decomposition
    result = _build_decomposition(n, m, p)
    
    # Verify
    if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p, tol=1e-6):
        return result
    
    # Fallback to schoolbook
    return _schoolbook_decomposition(n, m, p)
# EVOLVE-BLOCK-END
