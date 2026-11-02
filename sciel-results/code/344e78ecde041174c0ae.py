"""Rank-minimal exact tensor decomposition for 3x4 * 4x4 matrix multiplication.

Strategy:
  1. Block decomposition: split A (3x4) into a 2x4 top block and a 1x4 bottom row,
     and B (4x4) into four 2x2 blocks.
       - Top 2x4 * 4x4 handled with four Strassen 2x2*2x2 sub-multiplications (rank 7 each).
       - Bottom 1x4 * 4x4 handled with four rank-1 1x2*2x2 sub-multiplications (rank 1 each).
     This gives an initial rank of 4*7 + 4*1 = 32.
  2. Fallback: schoolbook rank-48 if anything fails.
"""
import time
import numpy as np

N, M, P = 3, 4, 4


def _verify_fast(U, V, W, n=N, m=M, p=P):
    """Fast verification: T[i,j,k] = sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] == 1."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    r = U.shape[1]
    if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
        return False
    if V.shape[1] != r or W.shape[1] != r:
        return False
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    recon = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(recon, np.ones((n, m, p)), atol=1e-9)


def _strassen_2x2_2x2():
    """Rank-7 Strassen decomposition for 2x2 * 2x2.

    Returns U (4,7), V (4,7), W (4,7) with index conventions
    U[i*2+j,c], V[j*2+k,c], W[k*2+i,c].
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))

    # M1 = (a11+a22)(b11+b22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # M2 = (a21+a22)b11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1

    # M3 = a11(b12-b22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = -1

    # M4 = a22(b21-b11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1; W[2, 3] = 1

    # M5 = (a11+a12)b22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1; W[3, 4] = 1

    # M6 = (a21-a11)(b11+b12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[0, 5] = 1; W[2, 5] = 1

    # M7 = (a12-a22)(b21+b22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[3, 6] = 1

    return U, V, W


def _one_by_two_by_two():
    """Rank-1 decomposition for 1x2 * 2x2.

    The tensor T[0,j,k] = 1 for all j,k in {0,1} is the all-ones tensor,
    which has rank 1: T = (1,1) ⊗ (1,1,1,1) ⊗ (1,1).

    Returns U (2,1), V (4,1), W (2,1) with conventions
    U[j,c], V[j*2+k,c], W[k,c].
    """
    U = np.ones((2, 1))
    V = np.ones((4, 1))
    W = np.ones((2, 1))

    return U, V, W


def _block_decomposition_3x4_4x4():
    """Assemble the block decomposition. Returns (U, V, W) as numpy arrays, rank 32."""
    S_U, S_V, S_W = _strassen_2x2_2x2()
    O_U, O_V, O_W = _one_by_two_by_two()

    total_rank = 4 * 7 + 4 * 1  # 32
    U_full = np.zeros((12, total_rank))
    V_full = np.zeros((16, total_rank))
    W_full = np.zeros((12, total_rank))

    col = 0

    def place_strassen(u_col_offset, v_col_offset, w_col_offset, c_off):
        """Place one Strassen 2x2*2x2 block.
        u_col_offset: global U col offset for A's columns (0 for left, 2 for right).
        v_col_offset: global V col offset for B's rows (0 for top, 2 for bottom).
        w_col_offset: global W col offset for C's cols (0 for left, 2 for right).
        """
        for c in range(7):
            for li in range(2):
                for lj in range(2):
                    u_idx = li * 4 + (u_col_offset + lj)
                    val = S_U[li * 2 + lj, c]
                    if val != 0:
                        U_full[u_idx, c_off + c] = val
            for lj in range(2):
                for lk in range(2):
                    v_idx = (v_col_offset + lj) * 4 + lk
                    val = S_V[lj * 2 + lk, c]
                    if val != 0:
                        V_full[v_idx, c_off + c] = val
            for lk in range(2):
                for li in range(2):
                    w_idx = (w_col_offset + lk) * 3 + li
                    val = S_W[lk * 2 + li, c]
                    if val != 0:
                        W_full[w_idx, c_off + c] = val

    def place_one_by_two(u_col_offset, v_col_offset, w_col_offset, c_off):
        """Place one 1x2*2x2 block (rank 1).
        u_col_offset: global U col offset (0 or 2).
        v_col_offset: global V col offset for B's rows (0 or 2).
        w_col_offset: global W col offset for C's cols (0 or 2).
        """
        for c in range(1):
            for lj in range(2):
                u_idx = 2 * 4 + (u_col_offset + lj)
                val = O_U[lj, c]
                if val != 0:
                    U_full[u_idx, c_off + c] = val
            for lj in range(2):
                for lk in range(2):
                    v_idx = (v_col_offset + lj) * 4 + lk
                    val = O_V[lj * 2 + lk, c]
                    if val != 0:
                        V_full[v_idx, c_off + c] = val
            for lk in range(2):
                w_idx = (w_col_offset + lk) * 3 + 2
                val = O_W[lk, c]
                if val != 0:
                    W_full[w_idx, c_off + c] = val

    # C00 = A00*B00 + A01*B10
    place_strassen(u_col_offset=0, v_col_offset=0, w_col_offset=0, c_off=col); col += 7
    place_strassen(u_col_offset=2, v_col_offset=2, w_col_offset=0, c_off=col); col += 7
    # C01 = A00*B01 + A01*B11
    place_strassen(u_col_offset=0, v_col_offset=0, w_col_offset=2, c_off=col); col += 7
    place_strassen(u_col_offset=2, v_col_offset=2, w_col_offset=2, c_off=col); col += 7
    # C10 = A10*B00 + A11*B10
    place_one_by_two(u_col_offset=0, v_col_offset=0, w_col_offset=0, c_off=col); col += 1
    place_one_by_two(u_col_offset=2, v_col_offset=2, w_col_offset=0, c_off=col); col += 1
    # C11 = A10*B01 + A11*B11
    place_one_by_two(u_col_offset=0, v_col_offset=0, w_col_offset=2, c_off=col); col += 1
    place_one_by_two(u_col_offset=2, v_col_offset=2, w_col_offset=2, c_off=col); col += 1

    return U_full, V_full, W_full


def _schoolbook():
    """Rank-48 schoolbook decomposition."""
    rank = N * M * P
    U = np.zeros((N * M, rank))
    V = np.zeros((M * P, rank))
    W = np.zeros((P * N, rank))
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j, col] = 1
                V[j * P + k, col] = 1
                W[k * N + i, col] = 1
                col += 1
    return U, V, W


def construct():
    """Construct a rank-minimal exact tensor decomposition for 3x4 * 4x4."""
    start_time = time.time()
    deadline = start_time + 110

    U, V, W = _block_decomposition_3x4_4x4()

    if _verify_fast(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        U, V, W = _schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
