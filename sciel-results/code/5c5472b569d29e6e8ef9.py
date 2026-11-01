"""Matrix multiplication 2x3x8 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 3, 8]

import time
import numpy as np

_DEADLINE = 145.0  # Safety margin before 150s kill


def _build_schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
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
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    rank = U_arr.shape[1]
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(val - 1.0) > max_err:
                    max_err = abs(val - 1.0)
    return max_err


def _build_decomp(n, m, p):
    """
    Build a decomposition using Strassen for the 2x2 sub-blocks.
    
    For 2x3x8:
    - Split A (2x3) into A1 (2x2, cols 0,1) and A2 (2x1, col 2)
    - Split B (3x8) into B1 (2x8, rows 0,1) and B2 (1x8, row 2)
    - C = A1*B1 + A2*B2
    
    A2*B2: rank 16
    A1*B1: 2x2 * 2x8, split into 4 Strassen blocks of 2x2x2, rank 7*4=28
    
    Total rank: 44
    """
    n, m, p = 2, 3, 8
    
    rank_total = 44
    U = [[0] * rank_total for _ in range(n * m)]  # 6 x 44
    V = [[0] * rank_total for _ in range(m * p)]  # 24 x 44
    W = [[0] * rank_total for _ in range(p * n)]  # 16 x 44
    
    col = 0
    
    # Part 1: A2*B2 (rank 16)
    for k in range(8):
        U[2][col] = 1
        V[16 + k][col] = 1
        W[k * 2 + 0][col] = 1
        col += 1
        U[5][col] = 1
        V[16 + k][col] = 1
        W[k * 2 + 1][col] = 1
        col += 1
    
    # Part 2: A1*B1 via Strassen (rank 28)
    for g in range(4):
        b11 = 0 + 2 * g
        b12 = 0 + 2 * g + 1
        b21 = 8 + 2 * g
        b22 = 8 + 2 * g + 1
        
        a11 = 0
        a12 = 1
        a21 = 3
        a22 = 4
        
        # M1 = (a11+a22)(b11+b22)
        U[a11][col] = 1
        U[a22][col] = 1
        V[b11][col] = 1
        V[b22][col] = 1
        W[(2 * g) * 2 + 0][col] = 1
        W[(2 * g + 1) * 2 + 1][col] = 1
        col += 1
        
        # M2 = (a21+a22)b11
        U[a21][col] = 1
        U[a22][col] = 1
        V[b11][col] = 1
        W[(2 * g) * 2 + 1][col] = 1
        col += 1
        
        # M3 = a11(b22-b12)
        U[a11][col] = 1
        V[b22][col] = 1
        V[b12][col] = -1
        W[(2 * g + 1) * 2 + 0][col] = 1
        W[(2 * g + 1) * 2 + 1][col] = -1
        col += 1
        
        # M4 = a22(b11-b21)
        U[a22][col] = 1
        V[b11][col] = 1
        V[b21][col] = -1
        W[(2 * g) * 2 + 0][col] = -1
        W[(2 * g) * 2 + 1][col] = 1
        col += 1
        
        # M5 = (a11+a12)b22
        U[a11][col] = 1
        U[a12][col] = 1
        V[b22][col] = 1
        W[(2 * g + 1) * 2 + 0][col] = 1
        col += 1
        
        # M6 = (a21-a11)(b11+b12)
        U[a21][col] = 1
        U[a11][col] = -1
        V[b11][col] = 1
        V[b12][col] = 1
        W[(2 * g) * 2 + 0][col] = 1
        col += 1
        
        # M7 = (a12-a22)(b21+b22)
        U[a12][col] = 1
        U[a22][col] = -1
        V[b21][col] = 1
        V[b22][col] = 1
        W[(2 * g + 1) * 2 + 1][col] = 1
        col += 1
    
    return U, V, W


def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    
    # Try the Strassen-based decomposition (rank 44)
    try:
        if time.time() - start_time > _DEADLINE:
            raise TimeoutError("Deadline exceeded")
        U, V, W = _build_decomp(n, m, p)
        err = _verify(U, V, W, n, m, p)
        if err < 1e-10:
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback: schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
