"""Tensor decomposition for 3x4x5 matrix multiplication.
Uses a combination of Strassen (2x2x2) and direct constructions.
Aim: achieve rank 55 (better than schoolbook 60) via careful block decomposition.

The 3x4x5 multiplication is split into overlapping 2x2x2 Strassen blocks
and small direct blocks, with shared terms to reduce total rank.
"""

import time
import numpy as np
import sys

DIMENSIONS = [3, 4, 5]


def _verify_decomposition(n, m, p, U, V, W, tol=1e-9):
    """Verify that the tensor decomposition is exact."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True


def _strassen_2x2x2(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols, m, p, n, U, V, W, rank_offset):
    """Add Strassen decomposition for a 2x2 * 2x2 -> 2x2 product.
    
    Returns new rank offset.
    """
    ar1, ar2 = a_rows
    ac1, ac2 = a_cols
    br1, br2 = b_rows
    bc1, bc2 = b_cols
    cr1, cr2 = c_rows
    cc1, cc2 = c_cols
    
    a11_idx = ar1 * m + ac1
    a12_idx = ar1 * m + ac2
    a21_idx = ar2 * m + ac1
    a22_idx = ar2 * m + ac2
    
    b11_idx = br1 * p + bc1
    b12_idx = br1 * p + bc2
    b21_idx = br2 * p + bc1
    b22_idx = br2 * p + bc2
    
    c11_idx = cc1 * n + cr1
    c12_idx = cc1 * n + cr2
    c21_idx = cc2 * n + cr1
    c22_idx = cc2 * n + cr2
    
    terms = [
        ({a11_idx: 1, a22_idx: 1}, {b11_idx: 1, b22_idx: 1}, {c11_idx: 1, c22_idx: 1}),
        ({a21_idx: 1, a22_idx: 1}, {b11_idx: 1}, {c21_idx: 1}),
        ({a11_idx: 1}, {b12_idx: 1, b22_idx: -1}, {c12_idx: 1}),
        ({a22_idx: 1}, {b21_idx: 1, b11_idx: -1}, {c11_idx: -1, c21_idx: 1}),
        ({a11_idx: 1, a12_idx: 1}, {b22_idx: 1}, {c12_idx: 1, c22_idx: -1}),
        ({a21_idx: 1, a11_idx: -1}, {b11_idx: 1, b12_idx: 1}, {c11_idx: 1}),
        ({a12_idx: 1, a22_idx: -1}, {b21_idx: 1, b22_idx: 1}, {c11_idx: -1, c22_idx: 1}),
    ]
    
    for u_d, v_d, w_d in terms:
        for u_idx, u_val in u_d.items():
            for v_idx, v_val in v_d.items():
                for w_idx, w_val in w_d.items():
                    U[u_idx, rank_offset] = u_val
                    V[v_idx, rank_offset] = v_val
                    W[w_idx, rank_offset] = w_val
                    rank_offset += 1
    
    return rank_offset


def _direct_2x2x2(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols, m, p, n, U, V, W, rank_offset):
    """Add direct decomposition for a 2x2 * 2x2 -> 2x2 product."""
    ar1, ar2 = a_rows
    ac1, ac2 = a_cols
    br1, br2 = b_rows
    bc1, bc2 = b_cols
    cr1, cr2 = c_rows
    cc1, cc2 = c_cols
    
    products = [
        (ar1, ac1, br1, bc1, cr1, cc1),
        (ar1, ac1, br1, bc2, cr1, cc2),
        (ar1, ac2, br2, bc1, cr1, cc1),
        (ar1, ac2, br2, bc2, cr1, cc2),
        (ar2, ac1, br1, bc1, cr2, cc1),
        (ar2, ac1, br1, bc2, cr2, cc2),
        (ar2, ac2, br2, bc1, cr2, cc1),
        (ar2, ac2, br2, bc2, cr2, cc2),
    ]
    
    for a_r, a_c, b_r, b_c, c_r, c_c in products:
        U[a_r * m + a_c, rank_offset] = 1
        V[b_r * p + b_c, rank_offset] = 1
        W[c_c * n + c_r, rank_offset] = 1
        rank_offset += 1
    
    return rank_offset


def _direct_1x2x2(a_row, a_cols, b_rows, b_cols, c_row, c_cols, m, p, n, U, V, W, rank_offset):
    """Add direct decomposition for a 1x2 * 2x2 -> 1x2 product."""
    a1 = a_row * m + a_cols[0]
    a2 = a_row * m + a_cols[1]
    
    products = [
        (a1, b_rows[0] * p + b_cols[0], c_row, c_cols[0]),
        (a1, b_rows[0] * p + b_cols[1], c_row, c_cols[1]),
        (a2, b_rows[1] * p + b_cols[0], c_row, c_cols[0]),
        (a2, b_rows[1] * p + b_cols[1], c_row, c_cols[1]),
    ]
    
    for a_idx, b_idx, c_r, c_c in products:
        U[a_idx, rank_offset] = 1
        V[b_idx, rank_offset] = 1
        W[c_c * n + c_r, rank_offset] = 1
        rank_offset += 1
    
    return rank_offset


def _direct_2x2x1(a_rows, a_cols, b_row, b_col, c_rows, c_col, m, p, n, U, V, W, rank_offset):
    """Add direct decomposition for a 2x2 * 2x1 -> 2x1 product."""
    b_idx = b_row * p + b_col
    
    products = [
        (a_rows[0], a_cols[0], c_rows[0]),
        (a_rows[0], a_cols[1], c_rows[0]),
        (a_rows[1], a_cols[0], c_rows[1]),
        (a_rows[1], a_cols[1], c_rows[1]),
    ]
    
    for a_r, a_c, c_r in products:
        U[a_r * m + a_c, rank_offset] = 1
        V[b_idx, rank_offset] = 1
        W[c_col * n + c_r, rank_offset] = 1
        rank_offset += 1
    
    return rank_offset


def _direct_2x1x2(a_rows, a_col, b_row, b_cols, c_rows, c_cols, m, p, n, U, V, W, rank_offset):
    """Add direct decomposition for a 2x1 * 1x2 -> 2x2 product."""
    a1 = a_rows[0] * m + a_col
    a2 = a_rows[1] * m + a_col
    b1 = b_row * p + b_cols[0]
    b2 = b_row * p + b_cols[1]
    
    products = [
        (a1, b1, c_rows[0], c_cols[0]),
        (a1, b2, c_rows[0], c_cols[1]),
        (a2, b1, c_rows[1], c_cols[0]),
        (a2, b2, c_rows[1], c_cols[1]),
    ]
    
    for a_idx, b_idx, c_r, c_c in products:
        U[a_idx, rank_offset] = 1
        V[b_idx, rank_offset] = 1
        W[c_c * n + c_r, rank_offset] = 1
        rank_offset += 1
    
    return rank_offset


def _direct_1x1x1(a_row, a_col, b_row, b_col, c_row, c_col, m, p, n, U, V, W, rank_offset):
    """Add direct decomposition for a 1x1 * 1x1 -> 1x1 product."""
    U[a_row * m + a_col, rank_offset] = 1
    V[b_row * p + b_col, rank_offset] = 1
    W[c_col * n + c_row, rank_offset] = 1
    rank_offset += 1
    return rank_offset


def _direct_1x1x2(a_row, a_col, b_row, b_cols, c_row, c_cols, m, p, n, U, V, W, rank_offset):
    """Add direct decomposition for a 1x1 * 1x2 -> 1x2 product."""
    a_idx = a_row * m + a_col
    for bc, cc in zip(b_cols, c_cols):
        U[a_idx, rank_offset] = 1
        V[b_row * p + bc, rank_offset] = 1
        W[cc * n + c_row, rank_offset] = 1
        rank_offset += 1
    return rank_offset


def _direct_1x2x1(a_row, a_cols, b_row, b_col, c_row, c_col, m, p, n, U, V, W, rank_offset):
    """Add direct decomposition for a 1x2 * 2x1 -> 1x1 product."""
    b_idx = b_row * p + b_col
    for ac in a_cols:
        U[a_row * m + ac, rank_offset] = 1
        V[b_idx, rank_offset] = 1
        W[c_col * n + c_row, rank_offset] = 1
        rank_offset += 1
    return rank_offset


def _direct_2x1x1(a_rows, a_col, b_row, b_col, c_rows, c_col, m, p, n, U, V, W, rank_offset):
    """Add direct decomposition for a 2x1 * 1x1 -> 2x1 product."""
    b_idx = b_row * p + b_col
    for ar, cr in zip(a_rows, c_rows):
        U[ar * m + a_col, rank_offset] = 1
        V[b_idx, rank_offset] = 1
        W[c_col * n + cr, rank_offset] = 1
        rank_offset += 1
    return rank_offset


def construct():
    import time as _time
    _start = _time.time()
    _DEADLINE = 100.0  # safety margin before 120s limit
    
    def _deadline_reached():
        return (_time.time() - _start) > _DEADLINE
    
    n, m, p = DIMENSIONS  # 3, 4, 5
    nm, mp, pn = n * m, m * p, p * n
    
    # Block decomposition to reduce rank below 60.
    # Strategy: split middle dimension (4) into 2+2.
    # A (3x4) = [A_top (2x4); A_bot (1x4)]
    # B (4x5) = [B_top (2x5); B_bot (2x5)]
    # C (3x5) = A_top @ B_top + A_bot @ B_bot
    #
    # A_top @ B_top: (2x4) @ (2x5) -> split A_top cols into 2+2, B_top rows into 2+2
    #   = A_top[:,0:2] @ B_top[0:2,:] + A_top[:,2:4] @ B_top[2:4,:]
    #   Each term: (2x2) @ (2x5) -> split B cols into 2+2+1
    #     (2x2)@(2x2): rank 7, (2x2)@(2x2): rank 7, (2x2)@(2x1): rank 4 => 18
    #   Total: 18 + 18 = 36
    #
    # A_bot @ B_bot: (1x4) @ (2x5) -> split A_bot cols into 2+2, B_bot rows into 2+2
    #   = A_bot[0:2] @ B_bot[0:2,:] + A_bot[2:4] @ B_bot[2:4,:]
    #   Each term: (1x2) @ (2x5) -> rank 10 (direct)
    #   Total: 10 + 10 = 20
    #
    # Grand total: 36 + 20 = 56
    
    rank = 56
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    r = 0  # current rank offset
    
    # Helper: add a 2x2 x 2x2 Strassen block
    def strassen_2x2x2(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols, r_off):
        ar1, ar2 = a_rows
        ac1, ac2 = a_cols
        br1, br2 = b_rows
        bc1, bc2 = b_cols
        cr1, cr2 = c_rows
        cc1, cc2 = c_cols
        
        a11 = ar1 * m + ac1
        a12 = ar1 * m + ac2
        a21 = ar2 * m + ac1
        a22 = ar2 * m + ac2
        
        b11 = br1 * p + bc1
        b12 = br1 * p + bc2
        b21 = br2 * p + bc1
        b22 = br2 * p + bc2
        
        c11 = cc1 * n + cr1
        c12 = cc1 * n + cr2
        c21 = cc2 * n + cr1
        c22 = cc2 * n + cr2
        
        # Strassen 7 terms
        # M1 = (a11+a22)*(b11+b22) -> c11 += M1, c22 += M1
        # M2 = a21*(b11) -> c21 += M2
        # M3 = a11*(b12) -> c12 += M3
        # M4 = a22*(b21-b11) -> c11 -= M4, c21 += M4
        # M5 = (a11+a12)*(b22) -> c12 += M5, c22 -= M5
        # M6 = (a21-a11)*(b11+b12) -> c11 += M6
        # M7 = (a12-a22)*(b21+b22) -> c11 -= M7, c22 += M7
        
        # Term 1: M1 = (a11+a22)*(b11+b22)
        U[a11, r_off] = 1; U[a22, r_off] = 1
        V[b11, r_off] = 1; V[b22, r_off] = 1
        W[c11, r_off] = 1; W[c22, r_off] = 1
        r_off += 1
        
        # Term 2: M2 = a21*b11
        U[a21, r_off] = 1
        V[b11, r_off] = 1
        W[c21, r_off] = 1
        r_off += 1
        
        # Term 3: M3 = a11*b12
        U[a11, r_off] = 1
        V[b12, r_off] = 1
        W[c12, r_off] = 1
        r_off += 1
        
        # Term 4: M4 = a22*(b21-b11)
        U[a22, r_off] = 1
        V[b21, r_off] = 1; V[b11, r_off] = -1
        W[c11, r_off] = -1; W[c21, r_off] = 1
        r_off += 1
        
        # Term 5: M5 = (a11+a12)*b22
        U[a11, r_off] = 1; U[a12, r_off] = 1
        V[b22, r_off] = 1
        W[c12, r_off] = 1; W[c22, r_off] = -1
        r_off += 1
        
        # Term 6: M6 = (a21-a11)*(b11+b12)
        U[a21, r_off] = 1; U[a11, r_off] = -1
        V[b11, r_off] = 1; V[b12, r_off] = 1
        W[c11, r_off] = 1
        r_off += 1
        
        # Term 7: M7 = (a12-a22)*(b21+b22)
        U[a12, r_off] = 1; U[a22, r_off] = -1
        V[b21, r_off] = 1; V[b22, r_off] = 1
        W[c11, r_off] = -1; W[c22, r_off] = 1
        r_off += 1
        
        return r_off
    
    # Helper: add 2x2 x 2x1 direct block (rank 4)
    def direct_2x2x2x1(a_rows, a_cols, b_rows, b_col, c_rows, c_col, r_off):
        # a_rows: [ar1, ar2], a_cols: [ac1, ac2]
        # b_rows: [br1, br2], b_col: single col
        # c_rows: [cr1, cr2], c_col: single col
        for i in range(2):
            for j in range(2):
                U[a_rows[i] * m + a_cols[j], r_off] = 1
                V[b_rows[j] * p + b_col, r_off] = 1
                W[c_col * n + c_rows[i], r_off] = 1
                r_off += 1
        return r_off
    
    # Helper: add 1x2 x 2x5 direct block (rank 10)
    def direct_1x2x2x5(a_row, a_cols, b_rows, b_cols, c_row, c_cols, r_off):
        for ac, bc, cc in zip(a_cols, b_cols, c_cols):
            U[a_row * m + ac, r_off] = 1
            V[b_rows[0] * p + bc, r_off] = 1
            W[cc * n + c_row, r_off] = 1
            r_off += 1
        for ac, bc, cc in zip(a_cols, b_cols, c_cols):
            U[a_row * m + ac, r_off] = 1
            V[b_rows[1] * p + bc, r_off] = 1
            W[cc * n + c_row, r_off] = 1
            r_off += 1
        return r_off
    
    # A_top @ B_top contribution: C[0:2, :]
    # Term 1: A_top[:,0:2] @ B_top[0:2,:] = A[0:2,0:2] @ B[0:2,:]
    #   Split B cols into [0,1], [2,3], [4]
    r = strassen_2x2x2([0,1], [0,1], [0,1], [0,1], [0,1], [0,1], r)
    r = strassen_2x2x2([0,1], [0,1], [0,1], [2,3], [0,1], [2,3], r)
    r = direct_2x2x2x1([0,1], [0,1], [0,1], 4, [0,1], 4, r)
    
    # Term 2: A[0:2,2:4] @ B[2:4,:]
    r = strassen_2x2x2([0,1], [2,3], [2,3], [0,1], [0,1], [0,1], r)
    r = strassen_2x2x2([0,1], [2,3], [2,3], [2,3], [0,1], [2,3], r)
    r = direct_2x2x2x1([0,1], [2,3], [2,3], 4, [0,1], 4, r)
    
    # A_bot @ B_bot contribution: C[2:3, :]
    # Term 1: A_bot[0:2] @ B_bot[0:2,:] = A[2,0:2] @ B[2:4,:]
    r = direct_1x2x2x5(2, [0,1], [2,3], [0,1], 2, [0,1], r)
    r = direct_1x2x2x5(2, [0,1], [2,3], [2,3], 2, [2,3], r)
    r = direct_1x2x2x5(2, [0,1], [2,3], [4], 2, [4], r)
    
    # Term 2: A_bot[2:4] @ B_bot[2:4,:] = A[2,2:4] @ B[2:4,:]
    r = direct_1x2x2x5(2, [2,3], [2,3], [0,1], 2, [0,1], r)
    r = direct_1x2x2x5(2, [2,3], [2,3], [2,3], 2, [2,3], r)
    r = direct_1x2x2x5(2, [2,3], [2,3], [4], 2, [4], r)
    
    assert r == rank, f"Rank mismatch: expected {rank}, got {r}"
    
    # Verify
    if _deadline_reached():
        # Return schoolbook as fallback
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    c = i * m * p + j * p + k
                    U[i * m + j, c] = 1.0
                    V[j * p + k, c] = 1.0
                    W[k * n + i, c] = 1.0
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    if not _verify_decomposition(n, m, p, U, V, W):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    c = i * m * p + j * p + k
                    U[i * m + j, c] = 1.0
                    V[j * p + k, c] = 1.0
                    W[k * n + i, c] = 1.0
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
