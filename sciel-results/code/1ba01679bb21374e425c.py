"""Matrix multiplication 2,3,8 - rank-minimal tensor decomposition.

Strategy: Use Strassen decomposition on 2x3x2 sub-blocks, then attempt rank reduction
through systematic linear algebra and search.
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

# Strassen decomposition for 2x3 x 3x2 tensor (rank 11)
_STRASSEN_232_U = [
    {(0,0): 1, (1,2): 1},
    {(1,0): 1, (1,1): 1},
    {(0,0): 1},
    {(1,2): 1},
    {(0,0): 1, (0,1): 1},
    {(1,0): 1, (0,0): -1},
    {(0,1): 1, (1,2): -1},
    {(0,1): 1},
    {(1,0): 1},
    {(0,0): 1, (0,1): 1},
    {(1,0): 1, (1,1): 1},
]
_STRASSEN_232_V = [
    {(0,0): 1, (1,1): 1},
    {(0,0): 1},
    {(1,0): 1, (0,0): -1},
    {(1,1): 1, (0,1): -1},
    {(1,1): 1},
    {(0,0): 1, (0,1): 1},
    {(1,0): 1, (1,1): 1},
    {(0,1): 1},
    {(1,0): 1},
    {(1,1): 1, (1,0): -1},
    {(0,1): 1, (0,0): -1},
]
_STRASSEN_232_W = [
    {(0,0): 1, (1,1): 1},
    {(0,1): 1, (1,1): -1},
    {(0,0): 1, (0,1): 1},
    {(0,1): 1, (1,1): 1},
    {(0,0): 1, (1,1): 1},
    {(0,0): 1, (0,1): 1},
    {(0,1): 1, (1,1): 1},
    {(0,1): 1},
    {(1,0): 1},
    {(1,0): 1, (1,1): -1},
    {(1,0): 1, (1,1): 1},
]

def _block_decomposition_232(n, m, p):
    """Build rank-44 decomposition by 4 blocks of 11 terms each using 2x3x2 Strassen."""
    assert n == 2 and m == 3 and p == 8
    nblocks = p // 2  # 4
    rank = nblocks * 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for b in range(nblocks):
        k0 = 2 * b
        col = b * 11
        for r in range(11):
            for (i, j), val in _STRASSEN_232_U[r].items():
                U[i * m + j][col + r] = val
            for (j, kp), val in _STRASSEN_232_V[r].items():
                V[j * p + (k0 + kp)][col + r] = val
            for (kp, i), val in _STRASSEN_232_W[r].items():
                W[(k0 + kp) * n + i][col + r] = val

    return {"U": U, "V": V, "W": W}

def _compress_scalar(U, V, W, n, m, p):
    """Remove scalar-multiple terms."""
    rank = len(U[0])
    if rank <= 1:
        return U, V, W, rank
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    removed = set()
    new_rank = rank
    for _ in range(100):
        if new_rank <= 1:
            break
        found = False
        for r1 in range(new_rank):
            if r1 in removed:
                continue
            for r2 in range(r1 + 1, new_rank):
                if r2 in removed:
                    continue
                u1 = U_np[r1]
                u2 = U_np[r2]
                nz = np.nonzero(u2)[0]
                if len(nz) == 0:
                    continue
                a = u1[nz[0]] / u2[nz[0]]
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
    
    keep = [i for i in range(rank) if i not in removed]
    return [U[i] for i in keep], [V[i] for i in keep], [W[i] for i in keep], len(keep)

def _compress_tensor(U, V, W, n, m, p, max_iters=50, tol=1e-8):
    """Compress by removing tensor-linearly dependent terms."""
    rank = len(U[0])
    if rank <= 1:
        return U, V, W, rank
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    nm, mp, pn = n*m, m*p, p*n
    tensors = np.zeros((rank, nm*mp*pn))
    for r in range(rank):
        t = np.outer(np.outer(U_np[r], V_np[r]), W_np[r])
        tensors[r] = t.flatten()
    
    keep = list(range(rank))
    for _ in range(max_iters):
        if len(keep) <= 1:
            break
        found = False
        idx = len(keep) - 1
        r = keep[idx]
        if idx > 0:
            A = tensors[keep[:idx]]
            b = tensors[r]
            try:
                x, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=tol)
                if residuals and len(residuals) > 0:
                    res = np.linalg.norm(b - A @ x)
                    if res < tol * max(1.0, np.linalg.norm(b)):
                        keep.pop(idx)
                        found = True
            except Exception:
                pass
        if not found:
            break
    
    if len(keep) == rank:
        return U, V, W, rank
    
    return [U[i] for i in keep], [V[i] for i in keep], [W[i] for i in keep], len(keep)

def _try_reduce_rank(U, V, W, n, m, p, target_rank, timeout=5):
    """Try to reduce to target_rank by zeroing out some terms."""
    start_time = time.time()
    current_rank = len(U[0])
    
    if current_rank <= target_rank:
        return U, V, W, current_rank
    
    # Try removing terms one at a time and checking validity
    U_work = [row[:] for row in U]
    V_work = [row[:] for row in V]
    W_work = [row[:] for row in W]
    
    removed = set()
    while len(removed) < current_rank - target_rank:
        if time.time() - start_time > timeout:
            break
        found = False
        for r in range(current_rank):
            if r in removed:
                continue
            # Try removing term r
            U_test = [row[:] for row in U_work]
            V_test = [row[:] for row in V_work]
            W_test = [row[:] for row in W_work]
            
            # Zero out column r
            for i in range(len(U_test)):
                U_test[i][r] = 0
            for i in range(len(V_test)):
                V_test[i][r] = 0
            for i in range(len(W_test)):
                W_test[i][r] = 0
            
            if _validate(U_test, V_test, W_test, n, m, p):
                removed.add(r)
                found = True
                break
        
        if not found:
            break
    
    if not removed:
        return U, V, W, current_rank
    
    keep = [i for i in range(current_rank) if i not in removed]
    return [U[i] for i in keep], [V[i] for i in keep], [W[i] for i in keep], len(keep)

def construct():
    start = time.time()
    n, m, p = DIMENSIONS
    deadline = start + 115

    # Start with rank-44 Strassen block decomposition
    best = None
    try:
        cand = _block_decomposition_232(n, m, p)
        if _validate(cand["U"], cand["V"], cand["W"], n, m, p):
            # First compress scalar-multiple terms
            U_c, V_c, W_c, rank_c = _compress_scalar(cand["U"], cand["V"], cand["W"], n, m, p)
            if _validate(U_c, V_c, W_c, n, m, p):
                # Then compress tensor-linearly dependent terms
                U_t, V_t, W_t, rank_t = _compress_tensor(U_c, V_c, W_c, n, m, p)
                if _validate(U_t, V_t, W_t, n, m, p):
                    best = {"U": U_t, "V": V_t, "W": W_t}
                else:
                    best = {"U": U_c, "V": V_c, "W": W_c}
            else:
                best = cand
    except Exception:
        pass

    if best is not None:
        return best

    # Fallback to schoolbook
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
