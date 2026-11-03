import numpy as np
import time
from itertools import product

N, M, P = 2, 4, 6
NM = N * M   # 8
MP = M * P   # 24
PN = P * N   # 12


def _build_strassen_2x2x2():
    """
    Return the 7 Strassen terms for 2x2 x 2x2 matrix multiplication.
    Each term is (u, v, w) where u, v, w are dicts mapping local index to coeff.
    Local indices: 0=A[0,0], 1=A[0,1], 2=A[1,0], 3=A[1,1]
    W local: 0=C[0,0], 1=C[0,1], 2=C[1,0], 3=C[1,1]
    """
    return [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]


def _build_rank42():
    """
    Build the rank-42 decomposition using block Strassen.
    Split A (2x4) into two 2x2 column blocks.
    Split B (4x6) into three 4x2 row-column blocks (split rows into 2x2, cols into 3 groups of 2).
    
    Actually, let's split more carefully:
    A: 2 rows, 4 cols -> split cols into 2 groups of 2: A = [A0 | A1]
    B: 4 rows, 6 cols -> split rows into 2 groups of 2, cols into 3 groups of 2:
       B = [[B00 B01 B02],
            [B10 B11 B12]]
    
    C = A0*B0* + A1*B1* where * means all column blocks
    C[:, 0:2] = A0*B00 + A1*B10
    C[:, 2:4] = A0*B01 + A1*B11
    C[:, 4:6] = A0*B02 + A1*B12
    
    Each term is a sum of two 2x2 x 2x2 products.
    Use Strassen for each pair: (A0+A1)*(B00+B10) gives A0*B00 + A0*B10 + A1*B00 + A1*B10
    We want A0*B00 + A1*B10, so we need to subtract A0*B10 + A1*B00.
    This requires additional terms.
    
    Better approach: Just apply Strassen independently to each 2x2x2 sub-product.
    6 sub-products (2 A-blocks x 3 B-col-blocks, each with 2 B-row-blocks) = 12 2x2x2 products.
    Wait, that's not right either.
    
    Let me re-think. The standard approach:
    A (2x4) = [A_col0 (2x2) | A_col1 (2x2)]
    B (4x6) = rows split: B_row0 (2x6), B_row1 (2x6)
    
    C = A_col0 @ B_row0 + A_col1 @ B_row1
    
    A_col0 @ B_row0: 2x2 @ 2x6. Split B_row0 into 3 2x2 blocks.
    = A_col0 @ B_row0_0 + A_col0 @ B_row0_1 + A_col0 @ B_row0_2
    Each is 2x2 @ 2x2, rank 7. Total: 21.
    
    A_col1 @ B_row1: similarly 21.
    
    Total: 42.
    """
    rank = 42
    U = np.zeros((NM, rank))
    V = np.zeros((MP, rank))
    W = np.zeros((PN, rank))
    
    strassen = _build_strassen_2x2x2()
    
    col = 0
    # For each of the 6 2x2x2 sub-products:
    # A block: row 0-1, col 2*a (a=0,1)
    # B block: row 2*b (b=0,1), col 2*c (c=0,1,2)
    for a in range(2):  # A column block
        for b in range(2):  # B row block
            for c in range(3):  # B column block
                # A local indices: 0=(0,2a), 1=(0,2a+1), 2=(1,2a), 3=(1,2a+1)
                a_map = {
                    0: 0 * M + (2 * a),
                    1: 0 * M + (2 * a + 1),
                    2: 1 * M + (2 * a),
                    3: 1 * M + (2 * a + 1)
                }
                # B local indices: 0=(2b,2c), 1=(2b,2c+1), 2=(2b+1,2c), 3=(2b+1,2c+1)
                b_map = {
                    0: (2 * b) * P + (2 * c),
                    1: (2 * b) * P + (2 * c + 1),
                    2: (2 * b + 1) * P + (2 * c),
                    3: (2 * b + 1) * P + (2 * c + 1)
                }
                # W local indices: 0=(2c,0), 1=(2c+1,0), 2=(2c,1), 3=(2c+1,1)
                # W index: k*N + i where k is col, i is row
                w_map = {
                    0: (2 * c) * N + 0,
                    1: (2 * c + 1) * N + 0,
                    2: (2 * c) * N + 1,
                    3: (2 * c + 1) * N + 1
                }
                
                for u_coeffs, v_coeffs, w_coeffs in strassen:
                    for local_idx, coeff in u_coeffs:
                        U[a_map[local_idx], col] += coeff
                    for local_idx, coeff in v_coeffs:
                        V[b_map[local_idx], col] += coeff
                    for local_idx, coeff in w_coeffs:
                        W[w_map[local_idx], col] += coeff
                    col += 1
    
    return U.tolist(), V.tolist(), W.tolist()


def _verify(U, V, W, tol=1e-9):
    """Verify the decomposition is exact."""
    rank = len(U[0])
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    # Check: sum_r U[i,r] * V[j,r] * W[k,r] = delta(i,j,k)
    # where i indexes A (n*m), j indexes B (m*p), k indexes C (p*n)
    for i in range(NM):
        for j in range(MP):
            for k in range(PN):
                val = np.sum(U[i, :] * V[j, :] * W[k, :])
                # Determine if this should be 1 or 0
                ai, aj = i // M, i % M
                bj, bk = j // P, j % P
                ck, ci = k // N, k % N
                if ai == ci and aj == bj and bk == ck:
                    if abs(val - 1) > tol:
                        return False
                else:
                    if abs(val) > tol:
                        return False
    return True


def _build_schoolbook():
    """Build schoolbook rank-48 decomposition."""
    rank = N * M * P
    U = np.zeros((NM, rank))
    V = np.zeros((MP, rank))
    W = np.zeros((PN, rank))
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j, col] = 1
                V[j * P + k, col] = 1
                W[k * N + i, col] = 1
                col += 1
    return U.tolist(), V.tolist(), W.tolist()


def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 2x4 matrix by a 4x6 matrix.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    Score is the rank (lower is better).
    """
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Start with the proven rank-42 block-Strassen decomposition
    U, V, W = _build_rank42()
    
    if not _verify(U, V, W):
        # Fallback to schoolbook rank 48
        U, V, W = _build_schoolbook()
    
    return {"U": U, "V": V, "W": W}
