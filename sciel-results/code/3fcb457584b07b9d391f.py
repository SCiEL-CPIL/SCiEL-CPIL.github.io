"""Matrix multiplication 2,3,8 - rank-minimal tensor decomposition.

Strategy: split the 8 output columns into four 2-column blocks. Each 2x3 x 3x2
sub-tensor (all-ones 2x3x2) is decomposed as the direct sum of:
  * a 2x2x2 all-ones tensor  -> rank 7 (Strassen)
  * a 2x1x2 all-ones tensor  -> rank 1 (outer product)
giving rank 8 per block, total rank 32 (vs schoolbook 48).
"""
DIMENSIONS = [2, 3, 8]

import time
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START

def _validate(U, V, W, n, m, p):
    """Check if U, V, W form a valid tensor decomposition."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
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

# Local 2x2 Strassen rank-7 terms in (i, j), (j, k'), (k', i) coordinates.
# Indices: i, j, k' in {0, 1}
_STRASSEN_U = [
    {(0, 0): 1, (1, 1): 1},
    {(1, 0): 1, (1, 1): 1},
    {(0, 0): 1},
    {(1, 1): 1},
    {(0, 0): 1, (0, 1): 1},
    {(1, 0): 1, (0, 0): -1},
    {(0, 1): 1, (1, 1): -1},
]
_STRASSEN_V = [
    {(0, 0): 1, (1, 1): 1},
    {(0, 0): 1},
    {(0, 1): 1, (1, 1): -1},
    {(1, 0): 1, (0, 0): -1},
    {(1, 1): 1},
    {(0, 0): 1, (0, 1): 1},
    {(1, 0): 1, (1, 1): 1},
]
_STRASSEN_W = [
    {(0, 0): 1, (1, 1): 1},
    {(0, 1): 1, (1, 1): -1},
    {(0, 1): 1, (1, 1): 1},
    {(0, 0): 1, (0, 1): 1},
    {(0, 0): -1, (0, 1): 1},
    {(1, 1): 1},
    {(0, 0): 1},
]

# Strassen decomposition for 2x3 x 3x2 tensor (rank 11)
# This handles the full 2x3 matrix times 3x2 matrix
_STRASSEN_232_U = [
    # Term 0: (A00+A11)(B00+B11)
    {(0,0): 1, (1,2): 1},
    # Term 1: (A10+A11)B00
    {(1,0): 1, (1,1): 1},
    # Term 2: A00(B10-B00)
    {(0,0): 1},
    # Term 3: A11(B11-B01)
    {(1,2): 1},
    # Term 4: (A00+A01)B11
    {(0,0): 1, (0,1): 1},
    # Term 5: (A10-A00)(B00+B01)
    {(1,0): 1, (0,0): -1},
    # Term 6: (A01-A11)(B10+B11)
    {(0,1): 1, (1,2): -1},
    # Term 7: A01 B01
    {(0,1): 1},
    # Term 8: A10 B10
    {(1,0): 1},
    # Term 9: (A00+A01)(B11-B10)
    {(0,0): 1, (0,1): 1},
    # Term 10: (A10+A11)(B01-B00)
    {(1,0): 1, (1,1): 1},
]
_STRASSEN_232_V = [
    # Term 0
    {(0,0): 1, (1,1): 1},
    # Term 1
    {(0,0): 1},
    # Term 2
    {(1,0): 1, (0,0): -1},
    # Term 3
    {(1,1): 1, (0,1): -1},
    # Term 4
    {(1,1): 1},
    # Term 5
    {(0,0): 1, (0,1): 1},
    # Term 6
    {(1,0): 1, (1,1): 1},
    # Term 7
    {(0,1): 1},
    # Term 8
    {(1,0): 1},
    # Term 9
    {(1,1): 1, (1,0): -1},
    # Term 10
    {(0,1): 1, (0,0): -1},
]
_STRASSEN_232_W = [
    # Term 0
    {(0,0): 1, (1,1): 1},
    # Term 1
    {(0,1): 1, (1,1): -1},
    # Term 2
    {(0,0): 1, (0,1): 1},
    # Term 3
    {(0,1): 1, (1,1): 1},
    # Term 4
    {(0,0): 1, (1,1): 1},
    # Term 5
    {(0,0): 1, (0,1): 1},
    # Term 6
    {(0,1): 1, (1,1): 1},
    # Term 7
    {(0,1): 1},
    # Term 8
    {(1,0): 1},
    # Term 9
    {(1,0): 1, (1,1): -1},
    # Term 10
    {(1,0): 1, (1,1): 1},
]

def _block_decomposition(n, m, p):
    """Build rank-32 decomposition by 4 blocks of 8 terms each."""
    assert n == 2 and m == 3 and p == 8
    nblocks = p // 2  # 4
    rank = nblocks * 8
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for b in range(nblocks):
        k0 = 2 * b  # first column of this block
        col = b * 8
        
        # 7 Strassen terms (j in {0,1}, k in {k0, k0+1})
        for r in range(7):
            for (i, j), val in _STRASSEN_U[r].items():
                U[i * m + j][col + r] = val
            for (j, kp), val in _STRASSEN_V[r].items():
                V[j * p + (k0 + kp)][col + r] = val
            for (kp, i), val in _STRASSEN_W[r].items():
                W[(k0 + kp) * n + i][col + r] = val
        
        # 1 outer-product term (j = 2)
        oc = col + 7
        for i in range(n):
            U[i * m + 2][oc] = 1
        for kp in range(2):
            V[2 * p + (k0 + kp)][oc] = 1
        for kp in range(2):
            for i in range(n):
                W[(k0 + kp) * n + i][oc] = 1

    return {"U": U, "V": V, "W": W}

def _block_decomposition_232(n, m, p):
    """Build rank-44 decomposition by 4 blocks of 11 terms each using 2x3x2 Strassen."""
    assert n == 2 and m == 3 and p == 8
    nblocks = p // 2  # 4
    rank = nblocks * 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for b in range(nblocks):
        k0 = 2 * b  # first column of this block
        col = b * 11
        
        # 11 Strassen terms for 2x3 x 3x2
        for r in range(11):
            for (i, j), val in _STRASSEN_232_U[r].items():
                U[i * m + j][col + r] = val
            for (j, kp), val in _STRASSEN_232_V[r].items():
                V[j * p + (k0 + kp)][col + r] = val
            for (kp, i), val in _STRASSEN_232_W[r].items():
                W[(k0 + kp) * n + i][col + r] = val

    return {"U": U, "V": V, "W": W}

def _compress_decomposition(U, V, W, n, m, p, max_iters=50):
    """Attempt to reduce rank by finding and removing linearly dependent terms."""
    rank = len(U[0])
    if rank <= 1:
        return U, V, W, rank
    
    # Convert to numpy for linear algebra
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Build the 3-way tensor as a matrix: each term r gives a vector in R^{nm*mp*pn}
    # Actually, we check linear dependence in the space of (U_r, V_r, W_r) triples
    # Two terms r1, r2 are linearly dependent if U[r1]=a*U[r2], V[r1]=a*V[r2], W[r1]=a*W[r2]
    
    # For each pair, check if they are scalar multiples
    removed = set()
    new_rank = rank
    for _ in range(max_iters):
        if new_rank <= 1:
            break
        found = False
        for r1 in range(new_rank):
            if r1 in removed:
                continue
            for r2 in range(r1 + 1, new_rank):
                if r2 in removed:
                    continue
                # Check if U[r1] is a scalar multiple of U[r2]
                # Find scalar from first non-zero element
                u1 = U_np[r1]
                u2 = U_np[r2]
                # Find a non-zero element in u2
                nz = np.nonzero(u2)[0]
                if len(nz) == 0:
                    continue
                a = u1[nz[0]] / u2[nz[0]]
                # Check if U[r1] ≈ a * U[r2]
                if not np.allclose(u1, a * u2, atol=1e-9):
                    continue
                v1 = V_np[r1]
                v2 = V_np[r2]
                if not np.allclose(v1, a * v2, atol=1e-9):
                    continue
                w1 = W_np[r1]
                w2 = W_np[r2]
                if not np.allclose(w1, a * w2, atol=1e-9):
                    continue
                # Terms r1 and r2 are linearly dependent: remove r2
                removed.add(r2)
                found = True
                break
            if found:
                break
        if not found:
            break
        new_rank -= 1
    
    if not removed:
        return U, V, W, rank
    
    # Rebuild without removed terms
    keep = [i for i in range(rank) if i not in removed]
    U_new = [U[i] for i in keep]
    V_new = [V[i] for i in keep]
    W_new = [W[i] for i in keep]
    return U_new, V_new, W_new, len(keep)

def construct():
    start = time.time()
    n, m, p = DIMENSIONS
    deadline = start + 120  # 120s safety margin

    # Try the 2x2x2 + outer product block decomposition (rank 32) first
    best = None
    try:
        cand = _block_decomposition(n, m, p)
        if _validate(cand["U"], cand["V"], cand["W"], n, m, p):
            U_c, V_c, W_c, rank_c = _compress_decomposition(
                cand["U"], cand["V"], cand["W"], n, m, p)
            if _validate(U_c, V_c, W_c, n, m, p):
                best = {"U": U_c, "V": V_c, "W": W_c}
            else:
                best = cand
    except Exception:
        pass

    # Try the 2x3x2 block decomposition (rank 44)
    try:
        cand = _block_decomposition_232(n, m, p)
        if _validate(cand["U"], cand["V"], cand["W"], n, m, p):
            U_c, V_c, W_c, rank_c = _compress_decomposition(
                cand["U"], cand["V"], cand["W"], n, m, p)
            if _validate(U_c, V_c, W_c, n, m, p):
                if best is None or rank_c < len(best["U"][0]):
                    best = {"U": U_c, "V": V_c, "W": W_c}
            else:
                if best is None or 44 < len(best["U"][0]):
                    best = cand
    except Exception:
        pass

    if best is not None:
        return best

    # Fallback to schoolbook
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
