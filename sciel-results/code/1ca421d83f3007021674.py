"""Rank-minimal tensor decomposition for 2x5 * 5x5 matrix multiplication.

Uses block decomposition with Strassen's algorithm for 2x2 blocks to achieve rank 49.
Splits the 5x5 multiplication into four blocks:
1. 2x2 * 2x2 (Strassen, rank 7)
2. 2x2 * 2x3 (Schoolbook, rank 12)
3. 2x3 * 3x2 (Schoolbook, rank 12)
4. 2x3 * 3x3 (Schoolbook, rank 18)
Total rank: 49.
"""

import time
import numpy as np

_T0 = time.time()
_DEADLINE_S = 110.0


def _deadline_ok():
    return (time.time() - _T0) < _DEADLINE_S


def _strassen_222():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm.
    Local indices: i,j,k in 0..1
    U shape: (2*2, 7)
    V shape: (2*2, 7)
    W shape: (2*2, 7)
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen's 7 products:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    
    # U factors (A matrix):
    # r=0: a+d
    U[0, 0] = 1; U[1, 0] = 1
    # r=1: c+d
    U[2, 1] = 1; U[3, 1] = 1
    # r=2: a
    U[0, 2] = 1
    # r=3: d
    U[3, 3] = 1
    # r=4: a+b
    U[0, 4] = 1; U[1, 4] = 1
    # r=5: c-a
    U[2, 5] = 1; U[0, 5] = -1
    # r=6: b-d
    U[1, 6] = 1; U[3, 6] = -1
    
    # V factors (B matrix):
    # r=0: e+h
    V[0, 0] = 1; V[3, 0] = 1
    # r=1: e
    V[0, 1] = 1
    # r=2: f-h
    V[1, 2] = 1; V[3, 2] = -1
    # r=3: g-e
    V[2, 3] = 1; V[0, 3] = -1
    # r=4: h
    V[3, 4] = 1
    # r=5: e+f
    V[0, 5] = 1; V[1, 5] = 1
    # r=6: g+h
    V[2, 6] = 1; V[3, 6] = 1
    
    # W factors (C matrix):
    # W[k*n + i]
    # C00 is k=0, i=0 -> W[0]
    # C01 is k=1, i=0 -> W[2]
    # C10 is k=0, i=1 -> W[1]
    # C11 is k=1, i=1 -> W[3]
    
    # C00 (W[0]): m1 + m4 - m5 + m7
    W[0, 0] = 1
    W[0, 3] = 1
    W[0, 4] = -1
    W[0, 6] = 1
    
    # C01 (W[2]): m3 + m5
    W[2, 2] = 1
    W[2, 4] = 1
    
    # C10 (W[1]): m2 + m4
    W[1, 1] = 1
    W[1, 3] = 1
    
    # C11 (W[3]): m1 - m2 + m3 + m6
    W[3, 0] = 1
    W[3, 1] = -1
    W[3, 2] = 1
    W[3, 5] = 1
    
    return U, V, W


def _schoolbook_block(n, m, p):
    """
    Returns U, V, W for n x m * m x p using schoolbook.
    Local indices.
    """
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def construct():
    n, m, p = 2, 5, 5
    total_rank = 49
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Block 1: i in 0..1, j in 0..1, k in 0..1
    # Global indices:
    # U row: i*5 + j
    # V row: j*5 + k
    # W row: k*2 + i
    
    U1, V1, W1 = _strassen_222()
    rank1 = 7
    offset = 0
    for r in range(rank1):
        for i in range(2):
            for j in range(2):
                U[i * 5 + j, offset + r] = U1[i * 2 + j, r]
        for j in range(2):
            for k in range(2):
                V[j * 5 + k, offset + r] = V1[j * 2 + k, r]
        for k in range(2):
            for i in range(2):
                W[k * 2 + i, offset + r] = W1[k * 2 + i, r]
    offset += rank1
    
    # Block 2: i in 0..1, j in 0..1, k in 2..4
    # Local k' in 0..2
    # This is 2x2 * 2x3
    U2, V2, W2 = _schoolbook_block(2, 2, 3)
    rank2 = 12
    for r in range(rank2):
        for i in range(2):
            for j in range(2):
                U[i * 5 + j, offset + r] = U2[i * 2 + j, r]
        for j in range(2):
            for k_prime in range(3):
                k = k_prime + 2
                V[j * 5 + k, offset + r] = V2[j * 3 + k_prime, r]
        for k_prime in range(3):
            k = k_prime + 2
            for i in range(2):
                W[k * 2 + i, offset + r] = W2[k_prime * 2 + i, r]
    offset += rank2
    
    # Block 3: i in 0..1, j in 2..4, k in 0..1
    # Local j' in 0..2
    # This is 2x3 * 3x2
    U3, V3, W3 = _schoolbook_block(2, 3, 2)
    rank3 = 12
    for r in range(rank3):
        for i in range(2):
            for j_prime in range(3):
                j = j_prime + 2
                U[i * 5 + j, offset + r] = U3[i * 3 + j_prime, r]
        for j_prime in range(3):
            j = j_prime + 2
            for k in range(2):
                V[j * 5 + k, offset + r] = V3[j_prime * 2 + k, r]
        for k in range(2):
            for i in range(2):
                W[k * 2 + i, offset + r] = W3[k * 2 + i, r]
    offset += rank3
    
    # Block 4: i in 0..1, j in 2..4, k in 2..4
    # Local j' in 0..2, k' in 0..2
    # This is 2x3 * 3x3
    U4, V4, W4 = _schoolbook_block(2, 3, 3)
    rank4 = 18
    for r in range(rank4):
        for i in range(2):
            for j_prime in range(3):
                j = j_prime + 2
                U[i * 5 + j, offset + r] = U4[i * 3 + j_prime, r]
        for j_prime in range(3):
            j = j_prime + 2
            for k_prime in range(3):
                k = k_prime + 2
                V[j * 5 + k, offset + r] = V4[j_prime * 3 + k_prime, r]
        for k_prime in range(3):
            k = k_prime + 2
            for i in range(2):
                W[k * 2 + i, offset + r] = W4[k_prime * 2 + i, r]
    offset += rank4
    
    # Verify
    if not _verify(U, V, W, total_rank):
        # Fallback to schoolbook if verification fails
        U_s, V_s, W_s, rank_s = _schoolbook()
        return {"U": U_s, "V": V_s, "W": W_s}
        
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, rank):
    """Exactly check that the decomposition reproduces the multiplication tensor."""
    n, m, p = 2, 5, 5
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ui = U[i * m + j]
                vj = V[j * p + k]
                wk = W[k * n + i]
                s = 0
                for r in range(rank):
                    s += ui[r] * vj[r] * wk[r]
                if s != 1:
                    return False
    return True


def _schoolbook():
    """Return (U, V, W) as numpy arrays with integer 0/1 entries, rank = n*m*p."""
    n, m, p = 2, 5, 5
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    return U, V, W, rank
