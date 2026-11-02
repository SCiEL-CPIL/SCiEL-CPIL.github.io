"""
Rank-minimal tensor decomposition for 2x2 * 2x12 matrix multiplication.

Strategy:
  - Try rank-22 via two 2x2x6 blocks (7 Strassen + 4 combined products).
  - Fallback rank-42 via six 2x2x2 Strassen blocks.
"""

import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify tensor decomposition: U[a]·(V[b]*W[c]) == delta."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                val = float(np.dot(U[a], V[b] * W[c]))
                if abs(val - 1.0) > 1e-9:
                    return False
    for a in range(n*m):
        i, j = divmod(a, m)
        for b in range(m*p):
            jj, k = divmod(b, p)
            if j != jj:
                continue
            for c in range(p*n):
                kk, ii = divmod(c, n)
                if k != kk or i != ii:
                    val = float(np.dot(U[a], V[b] * W[c]))
                    if abs(val) > 1e-9:
                        return False
    return True


def _strassen_2x2x2():
    """Rank-7 Strassen for 2x2*2x2. Returns (U,V,W) as lists."""
    rank = 7
    U = [[0.0]*rank for _ in range(4)]
    V = [[0.0]*rank for _ in range(4)]
    W = [[0.0]*rank for _ in range(4)]

    # M1 = (a00+a11)(b00+b11) -> c00+=M1, c11+=M1
    U[0][0]=1; U[3][0]=1
    V[0][0]=1; V[3][0]=1
    W[0][0]=1; W[3][0]=1

    # M2 = (a10+a11)b00 -> c10+=M2, c11-=M2
    U[2][1]=1; U[3][1]=1
    V[0][1]=1
    W[2][1]=1; W[3][1]=-1

    # M3 = a00(b10-b00) -> c01+=M3, c11+=M3
    U[0][2]=1
    V[2][2]=1; V[0][2]=-1
    W[1][2]=1; W[3][2]=1

    # M4 = a11(b11-b01) -> c00+=M4, c10+=M4
    U[3][3]=1
    V[3][3]=1; V[1][3]=-1
    W[0][3]=1; W[2][3]=1

    # M5 = (a00+a01)b11 -> c01+=M5, c00-=M5
    U[0][4]=1; U[1][4]=1
    V[3][4]=1
    W[1][4]=1; W[0][4]=-1

    # M6 = (a10-a00)(b00+b01) -> c11+=M6
    U[2][5]=1; U[0][5]=-1
    V[0][5]=1; V[1][5]=1
    W[3][5]=1

    # M7 = (a01-a11)(b10+b11) -> c00+=M7
    U[1][6]=1; U[3][6]=-1
    V[2][6]=1; V[3][6]=1
    W[0][6]=1

    return U, V, W


def _strassen_2x2x6():
    """Rank-11 decomposition for 2x2*2x6.
    
    7 Strassen products for columns 0-1,
    4 products for columns 2-5 using shared A-factors.
    """
    rank = 11
    U = [[0.0]*rank for _ in range(4)]
    V = [[0.0]*rank for _ in range(12)]
    W = [[0.0]*rank for _ in range(12)]

    # --- Columns 0-1: standard Strassen (same as 2x2x2) ---
    # M1 = (a00+a11)(b00+b11)
    U[0][0]=1; U[3][0]=1
    V[0][0]=1; V[3][0]=1
    W[0][0]=1; W[3][0]=1

    # M2 = (a10+a11)b00
    U[2][1]=1; U[3][1]=1
    V[0][1]=1
    W[2][1]=1; W[3][1]=-1

    # M3 = a00(b10-b00)
    U[0][2]=1
    V[2][2]=1; V[0][2]=-1
    W[1][2]=1; W[3][2]=1

    # M4 = a11(b11-b01)
    U[3][3]=1
    V[3][3]=1; V[1][3]=-1
    W[0][3]=1; W[2][3]=1

    # M5 = (a00+a01)b11
    U[0][4]=1; U[1][4]=1
    V[3][4]=1
    W[1][4]=1; W[0][4]=-1

    # M6 = (a10-a00)(b00+b01)
    U[2][5]=1; U[0][5]=-1
    V[0][5]=1; V[1][5]=1
    W[3][5]=1

    # M7 = (a01-a11)(b10+b11)
    U[1][6]=1; U[3][6]=-1
    V[2][6]=1; V[3][6]=1
    W[0][6]=1

    # --- Columns 2-5: 4 products with shared structure ---
    # P8: (a00+a11)(b02+b04+b13+b15)
    U[0][7]=1; U[3][7]=1
    V[2][7]=1; V[4][7]=1; V[7][7]=1; V[9][7]=1
    W[2][7]=1; W[4][7]=1; W[7][7]=1; W[9][7]=1

    # P9: (a10+a11)(b02+b04)
    U[2][8]=1; U[3][8]=1
    V[2][8]=1; V[4][8]=1
    W[6][8]=1; W[8][8]=1

    # P10: a00(b12+b14-b02-b04)
    U[0][9]=1
    V[6][9]=1; V[8][9]=1; V[2][9]=-1; V[4][9]=-1
    W[2][9]=1; W[4][9]=1; W[6][9]=1; W[8][9]=1

    # P11: (a01-a11)(b13+b15)
    U[1][10]=1; U[3][10]=-1
    V[7][10]=1; V[9][10]=1
    W[3][10]=1; W[5][10]=1

    return U, V, W


def _try_rank22():
    """Build rank-22 from two 2x2x6 blocks and verify."""
    n, m, p = 2, 2, 12
    U6, V6, W6 = _strassen_2x2x6()
    rank = 22
    U = [[0.0]*rank for _ in range(n*m)]
    V = [[0.0]*rank for _ in range(m*p)]
    W = [[0.0]*rank for _ in range(p*n)]

    for t in range(2):
        base_k = 6*t
        for r in range(11):
            idx = t*11 + r
            for a in range(4):
                U[a][idx] = U6[a][r]
            for b in range(12):
                V[b][idx] = V6[b][r]
            for c in range(12):
                W[c][idx] = W6[c][r]
            # Remap V indices: block uses cols 0..5, shift by base_k
            for b_local in range(12):
                j, k_local = divmod(b_local, 6)
                k_global = base_k + k_local
                V[j*p + k_global][idx] = V[b_local][idx]
            # Remap W indices: block uses cols 0..5, shift by base_k
            for c_local in range(12):
                k_local, i = divmod(c_local, 2)
                k_global = base_k + k_local
                W[k_global*n + i][idx] = W[c_local][idx]

    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    return None


def _fallback_rank42():
    """Rank-42: six independent Strassen 2x2x2 blocks."""
    n, m, p = 2, 2, 12
    U2, V2, W2 = _strassen_2x2x2()
    rank = 42
    U = [[0.0]*rank for _ in range(n*m)]
    V = [[0.0]*rank for _ in range(m*p)]
    W = [[0.0]*rank for _ in range(p*n)]

    for t in range(6):
        base_k = 2*t
        for s in range(7):
            idx = t*7 + s
            for a in range(4):
                U[a][idx] = U2[a][s]
            for b in range(4):
                j, k_local = divmod(b, 2)
                k_global = base_k + k_local
                V[j*p + k_global][idx] = V2[b][s]
            for c in range(4):
                k_local, i = divmod(c, 2)
                k_global = base_k + k_local
                W[k_global*n + i][idx] = W2[c][s]

    return {"U": U, "V": V, "W": W}


def construct():
    """Construct rank-minimal tensor decomposition for 2x2*2x12."""
    start = time.time()
    deadline = start + 130.0  # safety margin

    # Try rank-22
    if time.time() < deadline:
        result = _try_rank22()
        if result is not None:
            return result

    # Fallback: rank-42
    return _fallback_rank42()
# EVOLVE-BLOCK-END
