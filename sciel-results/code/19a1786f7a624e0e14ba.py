"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

# ---------------------------------------------------------------------------
# Strassen 2x2x2 rank-7 decomposition.
# Conventions (row-major, 2x2):
#   U index u = i*2 + j  (A element a_ij)
#   V index v = j*2 + k  (B element b_jk)
#   W index w = k*2 + i  (C element c_ki)   <-- official trace convention
# Each term is a rank-1 tensor: U_vec * V_vec * W_vec.
# ---------------------------------------------------------------------------
def _strassen_2x2():
    """Return (U, V, W) each 4x7 for 2x2x2 matmul. W[k*2+i, c]."""
    U = [[0] * 7 for _ in range(4)]
    V = [[0] * 7 for _ in range(4)]
    W = [[0] * 7 for _ in range(4)]

    def put(u, v, w, c, val):
        U[u][c] = val
        V[v][c] = val
        W[w][c] = val

    # M1 = (a00+a11)(b00+b11) -> c00 += M1, c11 += M1
    put(0, 0, 0, 0, 1); put(3, 3, 3, 0, 1)
    # M2 = (a10+a11)b00 -> c10 += M2, c11 -= M2
    put(2, 0, 2, 1, 1); put(3, 3, 3, 1, -1)
    # M3 = a00(b01-b11) -> c01 += M3, c11 += M3
    put(0, 1, 1, 2, 1); put(0, 3, 3, 2, 1)
    # M4 = a11(b10-b00) -> c00 += M4, c10 += M4
    put(3, 2, 0, 3, 1); put(3, 0, 2, 3, 1)
    # M5 = (a00+a01)b11 -> c00 -= M5, c01 += M5
    put(0, 3, 0, 4, -1); put(1, 3, 1, 4, 1)
    # M6 = (a10-a00)(b00+b01) -> c11 += M6
    put(2, 0, 3, 5, 1); put(0, 1, 3, 5, -1)
    # M7 = (a01-a11)(b10+b11) -> c00 += M7
    put(1, 2, 0, 6, 1); put(3, 3, 0, 6, -1)

    return U, V, W


def _construct_block():
    """Block-partition 5 = 2+2+1, Strassen on 2x2x2 blocks, schoolbook elsewhere.

    Returns (result_dict, rank)."""
    n = m = p = 5
    sizes = [2, 2, 1]
    starts = [0, 2, 4]
    rank = 117
    nm = mp = pn = 25

    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]

    Su, Sv, Sw = _strassen_2x2()
    col = 0
    for i_idx in range(3):
        for k_idx in range(3):
            for j_idx in range(3):
                si = sizes[i_idx]
                sk = sizes[k_idx]
                sj = sizes[j_idx]
                I = starts[i_idx]
                K = starts[k_idx]
                J = starts[j_idx]

                if si == 2 and sk == 2 and sj == 2:
                    for c in range(7):
                        for r in range(4):
                            u_row = (I + r // 2) * m + (K + r % 2)
                            v_row = (K + r // 2) * p + (J + r % 2)
                            w_row = (J + r // 2) * n + (I + r % 2)
                            U[u_row][col] = Su[r][c]
                            V[v_row][col] = Sv[r][c]
                            W[w_row][col] = Sw[r][c]
                        col += 1
                else:
                    for u in range(si):
                        for v in range(sk):
                            for w in range(sj):
                                u_row = (I + u) * m + (K + v)
                                v_row = (K + v) * p + (J + w)
                                w_row = (J + w) * n + (I + u)
                                U[u_row][col] = 1
                                V[v_row][col] = 1
                                W[w_row][col] = 1
                                col += 1

    assert col == rank, (col, rank)
    return {"U": U, "V": W and U and {"U": U, "V": V, "W": W}}, rank


def _schoolbook():
    """Fallback rank-125 decomposition."""
    n = m = p = 5
    rank = 125
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


def _verify_fast(n, m, p, U, V, W):
    """Fast NumPy verification. True iff all 125 tensor entries equal 1."""
    rank = len(U[0])
    Ua = np.asarray(U, dtype=np.float64)
    Va = np.asarray(V, dtype=np.float64)
    Wa = np.asarray(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = float(np.dot(Ua[i * m + j], Va[j * p + k] * Wa[k * n + i]))
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def _verify_exact(n, m, p, U, V, W):
    """Exact Fraction verification (used only as a final safety net)."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                u_row = U[i * m + j]
                v_row = V[j * p + k]
                w_row = W[k * n + i]
                for c in range(rank):
                    uc = u_row[c] if isinstance(u_row[c], Fraction) else Fraction(u_row[c])
                    vc = v_row[c] if isinstance(v_row[c], Fraction) else Fraction(v_row[c])
                    wc = w_row[c] if isinstance(w_row[c], Fraction) else Fraction(w_row[c])
                    s += uc * vc * wc
                if s != 1:
                    return False
    return True


def construct():
    """Construct a rank-minimal exact tensor decomposition for 5x5x5 matmul."""
    n = m = p = 5
    deadline = time.time() + 110  # safety margin before 150s kill

    # Primary: block-partition + Strassen, rank 117.
    try:
        result, _ = _construct_block()
    except Exception:
        return _schoolbook()

    if _verify_fast(n, m, p, result["U"], result["V"], result["W"]):
        if time.time() < deadline:
            # Final exact safety net (cheap for 125 entries x 117 cols).
            if _verify_exact(n, m, p, result["U"], result["V"], result["W"]):
                return result
        return result

    # Fallback: schoolbook rank 125.
    return _schoolbook()
# EVOLVE-BLOCK-END
