import numpy as np
from fractions import Fraction

def _build_3x2x2_factors():
    """
    Build the rank-11 decomposition for 3x2 * 2x2 -> 3x2.
    Indices:
      A: 3x2, index = row*2 + col  (0..5)
      B: 2x2, index = row*2 + col  (0..3)
      C: 3x2, index = row*2 + col  (0..5)
    """
    nU, nV, nW = 6, 4, 6
    rk = 11
    U = np.zeros((nU, rk), dtype=object)
    V = np.zeros((nV, rk), dtype=object)
    W = np.zeros((nW, rk), dtype=object)

    # Strassen for 2x2 * 2x2 (rows 0,1 of A)
    # M1 = (a00+a11)*(b00+b11) -> c00, c11
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    # M2 = (a01+a11)*b00 -> c10, c11(-)
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1; W[3,1]=-1
    # M3 = a00*(b11-b01) -> c01, c11
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[1,2]=1; W[3,2]=1
    # M4 = a11*(b10-b00) -> c00, c10
    U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=1; W[2,3]=1
    # M5 = (a00+a01)*(b10-b11) -> c00(-), c01
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[1,4]=1
    # M6 = (a10-a00)*(b00+b01) -> c11
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    # M7 = (a01-a11)*(b10+b11) -> c00
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1

    # Row 2 (a20, a21) direct terms
    # M8 = a20*b00 -> c20
    U[4,7]=1; V[0,7]=1; W[4,7]=1
    # M9 = a21*b10 -> c20
    U[5,8]=1; V[2,8]=1; W[4,8]=1
    # M10 = a20*b01 -> c21
    U[4,9]=1; V[1,9]=1; W[5,9]=1
    # M11 = a21*b11 -> c21
    U[5,10]=1; V[3,10]=1; W[5,10]=1

    return U, V, W


def _try_reduce_rank(U, V, W, target_rank, timeout=120):
    """
    Attempt to reduce the rank below the current value using a randomized
    column-reduction heuristic. If the current decomposition has rank R,
    we try to see if a subset of columns of size target_rank still spans
    the tensor. This is a heuristic: we try random subsets and check
    whether the decomposition still holds.
    """
    import time
    start = time.time()
    nU, cur_rank = U.shape
    nV, _ = V.shape
    nW, _ = W.shape

    # Build the full tensor T from the decomposition
    T = np.zeros((nU, nV, nW), dtype=object)
    for k in range(cur_rank):
        T += np.einsum('i,j,k->ijk', U[:,k], V[:,k], W[:,k])

    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rk = cur_rank

    for trial_rank in range(target_rank, best_rk):
        if time.time() - start > timeout - 10:
            break
        # Try random subsets of columns
        for attempt in range(50):
            if time.time() - start > timeout - 5:
                break
            idx = np.random.choice(cur_rank, trial_rank, replace=False)
            # Check if the subset of columns still spans T
            # Build the reduced tensor
            T_sub = np.zeros((nU, nV, nW), dtype=object)
            for k in idx:
                T_sub += np.einsum('i,j,k->ijk', U[:,k], V[:,k], W[:,k])
            if np.array_equal(T, T_sub):
                best_rk = trial_rank
                best_U = U[:,idx].copy()
                best_V = V[:,idx].copy()
                best_W = W[:,idx].copy()
                break
        if best_rk == target_rank:
            break

    return best_U, best_V, best_W, best_rk


def construct():
    n, m, p = 3, 4, 4
    N = n * m  # 12
    M = m * p  # 16
    P = p * n  # 12

    # Build local factors
    U_loc, V_loc, W_loc = _build_3x2x2_factors()
    local_rank = 11

    # Assemble the 3x4*4x4 decomposition via 4 blocks
    rank = 4 * local_rank
    U = np.zeros((N, rank), dtype=object)
    V = np.zeros((M, rank), dtype=object)
    W = np.zeros((P, rank), dtype=object)

    def add_block(block_idx, a_col_start, b_row_start, b_col_start, c_col_start):
        off = block_idx * local_rank
        for r in range(3):
            for c in range(2):
                la = r * 2 + c
                ga = r * m + (a_col_start + c)
                U[ga, off:off+local_rank] += U_loc[la, :]
        for r in range(2):
            for c in range(2):
                lb = r * 2 + c
                gb = (b_row_start + r) * p + (b_col_start + c)
                V[gb, off:off+local_rank] += V_loc[lb, :]
        for r in range(3):
            for c in range(2):
                lc = r * 2 + c
                gc = (c_col_start + c) * n + r
                W[gc, off:off+local_rank] += W_loc[lc, :]

    # 4 blocks
    add_block(0, 0, 0, 0, 0)  # A_L * B_11 -> C_L
    add_block(1, 2, 2, 0, 0)  # A_R * B_21 -> C_L
    add_block(2, 0, 0, 2, 2)  # A_L * B_12 -> C_R
    add_block(3, 2, 2, 2, 2)  # A_R * B_22 -> C_R

    # Try to reduce rank
    U_red, V_red, W_red, rk = _try_reduce_rank(U, V, W, target_rank=40, timeout=110)
    if rk < rank:
        U, V, W, rank = U_red, V_red, W_red, rk

    # Convert to list
    def to_list(arr):
        out = []
        for row in arr:
            out.append([float(x) for x in row])
        return out

    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
