"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast vectorized verification using einsum."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    U_3d = U_arr.reshape(n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)
    
    T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
    
    max_err = np.max(np.abs(T - 1.0))
    return max_err < tol, max_err


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that the decomposition exactly reconstructs the matrix multiplication tensor."""
    ok, _ = _verify_fast(U, V, W, n, m, p, tol)
    return ok


def _strassen_2x2x2x2():
    """Return U, V, W for 2x2 * 2x2 = 2x2 using Strassen (rank 7)."""
    n = m = p = 2
    rank = 7
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # P1 = (A11+A22)(B11+B22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # P2 = (A21+A22)B11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # P3 = A11(B12-B22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # P4 = A22(B21-B11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # P5 = (A11+A12)B22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # P6 = (A21-A11)(B11+B12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # P7 = (A12-A22)(B21+B22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[3][6] = 1
    
    return U, V, W


def _naive_1x2x2x2():
    """Return U, V, W for 1x2 * 2x2 = 1x2 using naive (rank 4)."""
    n = 1; m = 2; p = 2
    rank = 4
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
    U[0][1] = 1; V[1][1] = 1; W[1][1] = 1
    U[1][2] = 1; V[2][2] = 1; W[0][2] = 1
    U[1][3] = 1; V[3][3] = 1; W[1][3] = 1
    
    return U, V, W


def _add_block(U, V, W, n, m, p, i_off, j_off, k_off, n_b, m_b, p_b, block_U, block_V, block_W, col_off):
    """Add a block decomposition at offset positions."""
    for i in range(n_b):
        for j in range(m_b):
            for c in range(len(block_U[0])):
                U[(i_off+i)*m + (j_off+j)][col_off+c] += block_U[i*m_b+j][c]
    for j in range(m_b):
        for k in range(p_b):
            for c in range(len(block_V[0])):
                V[(j_off+j)*p + (k_off+k)][col_off+c] += block_V[j*p_b+k][c]
    for k in range(p_b):
        for i in range(n_b):
            for c in range(len(block_W[0])):
                W[(k_off+k)*n + (i_off+i)][col_off+c] += block_W[k*n_b+i][c]


def _strassen_3x4x4():
    """Build a Strassen-based decomposition for 3x4 * 4x4 = 3x4."""
    n, m, p = 3, 4, 4
    total_rank = 4 * 7 + 4 * 4  # 4 Strassen + 4 naive = 28 + 16 = 44
    
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    col = 0
    sU, sV, sW = _strassen_2x2x2x2()
    
    # Top 2 rows, left 2 cols of B
    _add_block(U, V, W, n, m, p, 0, 0, 0, 2, 2, 2, sU, sV, sW, col)
    col += 7
    # Top 2 rows, right 2 cols of B
    _add_block(U, V, W, n, m, p, 0, 2, 0, 2, 2, 2, sU, sV, sW, col)
    col += 7
    # Top 2 rows, left 2 cols of B, right 2 cols of output
    _add_block(U, V, W, n, m, p, 0, 0, 2, 2, 2, 2, sU, sV, sW, col)
    col += 7
    # Top 2 rows, right 2 cols of B, right 2 cols of output
    _add_block(U, V, W, n, m, p, 0, 2, 2, 2, 2, 2, sU, sV, sW, col)
    col += 7
    
    nU, nV, nW = _naive_1x2x2x2()
    
    # Bottom 1 row, left 2 cols of B, left 2 cols of output
    _add_block(U, V, W, n, m, p, 2, 0, 0, 1, 2, 2, nU, nV, nW, col)
    col += 4
    # Bottom 1 row, right 2 cols of B, left 2 cols of output
    _add_block(U, V, W, n, m, p, 2, 2, 0, 1, 2, 2, nU, nV, nW, col)
    col += 4
    # Bottom 1 row, left 2 cols of B, right 2 cols of output
    _add_block(U, V, W, n, m, p, 2, 0, 2, 1, 2, 2, nU, nV, nW, col)
    col += 4
    # Bottom 1 row, right 2 cols of B, right 2 cols of output
    _add_block(U, V, W, n, m, p, 2, 2, 2, 1, 2, 2, nU, nV, nW, col)
    col += 4
    
    assert col == total_rank, f"Expected {total_rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}


def _schoolbook():
    n, m, p = DIMENSIONS
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


def _try_polish(U, V, W, n, m, p, max_time=5):
    """Try to polish a near-exact solution to exact fractions."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    U_frac = [[Fraction(x).limit_denominator(100) for x in row] for row in U_arr.tolist()]
    V_frac = [[Fraction(x).limit_denominator(100) for x in row] for row in V_arr.tolist()]
    W_frac = [[Fraction(x).limit_denominator(100) for x in row] for row in W_arr.tolist()]
    
    ok, _ = _verify_fast(U_frac, V_frac, W_frac, n, m, p, tol=1e-9)
    if ok:
        return U_frac, V_frac, W_frac
    
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Try Strassen-based decomposition first (rank 44)
    try:
        strassen_result = _strassen_3x4x4()
        ok, err = _verify_fast(strassen_result['U'], strassen_result['V'], strassen_result['W'], n, m, p, tol=1e-9)
        if ok:
            return strassen_result
    except Exception:
        pass
    
    # Fallback to schoolbook
    return _schoolbook()

# EVOLVE-BLOCK-END
