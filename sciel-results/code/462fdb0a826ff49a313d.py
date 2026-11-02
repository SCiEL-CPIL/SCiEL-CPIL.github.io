"""Rank-minimal exact tensor decomposition for 4x4 * 4x9 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]
_START = time.time()
_DEADLINE = 138.0

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """T[io, u, v] = 1 iff output (i,k) uses a[i,j]*b[j,k]; io=k*n+i, u=i*m+j, v=j*p+k."""
    T = np.zeros((p * n, n * m, m * p), dtype=object)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[k * n + i, i * m + j, j * p + k] = 1
    return T


def _exact_check(U, V, W, T, n, m, p):
    rank = len(U[0])
    Ua = np.array(U, dtype=object)
    Va = np.array(V, dtype=object)
    Wa = np.array(W, dtype=object)
    for io in range(p * n):
        for u in range(n * m):
            for v in range(m * p):
                s = 0
                for c in range(rank):
                    s += Ua[u, c] * Va[v, c] * Wa[io, c]
                if s != T[io, u, v]:
                    return False
    return True


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def _winograd_4point(n, m, p):
    """
    Apply 4-point DFT along i and j indices to diagonalize the 4x4 part.
    This decomposes into 16 frequency components, each requiring p scalar mults.
    Total rank = 16 * p = 144.
    """
    # 4-point DFT matrix (4x4)
    # F[k][i] = omega^(k*i) where omega = exp(2*pi*i/4)
    # For exact computation, we use the discrete values:
    # omega^0 = 1, omega^1 = i, omega^2 = -1, omega^3 = -i
    # But we need integer/half-integer entries, so let's use a different approach.
    
    # Actually, let's use the fact that we can express the 4x4 matrix multiplication
    # using a tensor decomposition based on the group algebra of Z_4 x Z_4.
    # The tensor rank of matrix multiplication n x m x p is at most n*m*p.
    # For 4x4x9, schoolbook gives 144.
    
    # Let's try a different approach: use Strassen-like decomposition on the 4x4 part.
    # 4x4x4 can be done with 16 multiplications (schoolbook) or potentially fewer.
    # Actually, the rank of 4x4x4 matrix multiplication is 16 (it's known).
    
    # For 4x4x9, we can write it as sum of 9 copies of 4x4x1, each with rank 4,
    # giving total rank 36? No, that's not right either.
    
    # Let me think more carefully. The tensor is T[i,j,k] for i in [0,4), j in [0,4), k in [0,9).
    # This is a 4x4x9 tensor. The rank of this tensor is what we want to minimize.
    
    # One approach: view it as 9 independent 4x4x1 tensors. Each 4x4x1 has rank 4
    # (just the diagonal). So total rank = 9 * 4 = 36? That seems too optimistic.
    
    # Actually, a 4x4x1 tensor T[i,j,0] = delta_{i,j} has rank 4 (it's a diagonal matrix).
    # But when we have T[i,j,k] for k=0..8, each "slice" k is a 4x4 matrix.
    # If each slice is independent, we'd need 9*16 = 144 for schoolbook.
    
    # Let's just return schoolbook for now, but try to find a better decomposition.
    return _schoolbook(n, m, p)


def _try_strassen_4x4(n, m, p):
    """
    Try to decompose 4x4x9 using Strassen-like ideas.
    Split 4x4 into 2x2 blocks and use Strassen's 7-multiplication algorithm.
    """
    # For 4x4x4, Strassen gives rank 7*4 = 28 (since each 2x2x2 needs 7 mults,
    # and there are 4 blocks). But actually 4x4x4 rank is 16.
    
    # Let me try a different approach. The key insight is that we can use
    # bilinear decomposition with structured matrices.
    
    # For now, return schoolbook
    return _schoolbook(n, m, p)


def _greedy_eliminate(U, V, W, T, n, m, p, max_time=30.0):
    """
    Try to greedily eliminate tensor factors by checking if removing a column
    still preserves the exact decomposition.
    """
    start = time.time()
    rank = len(U[0])
    nU = len(U)
    nV = len(V)
    nW = len(W)
    
    # Try to eliminate factors one at a time
    eliminated = set()
    
    while time.time() - start < max_time:
        progress = False
        for c in range(rank):
            if c in eliminated:
                continue
            # Check if we can eliminate factor c
            # Temporarily remove factor c
            new_rank = rank - len(eliminated)
            # Check if the decomposition still holds without factor c
            valid = True
            for io in range(p * n):
                if not valid:
                    break
                for u in range(n * m):
                    if not valid:
                        break
                    for v in range(m * p):
                        s = 0
                        for cc in range(rank):
                            if cc in eliminated or cc == c:
                                continue
                            s += U[u][cc] * V[v][cc] * W[io][cc]
                        if s != T[io, u, v]:
                            valid = False
                            break
            if valid:
                eliminated.add(c)
                progress = True
        
        if not progress:
            break
    
    if not eliminated:
        return U, V, W, rank
    
    # Build reduced matrices
    keep = [c for c in range(rank) if c not in eliminated]
    new_rank = len(keep)
    U_new = [[U[i][c] for c in keep] for i in range(nU)]
    V_new = [[V[i][c] for c in keep] for i in range(nV)]
    W_new = [[W[i][c] for c in keep] for i in range(nW)]
    
    return U_new, V_new, W_new, new_rank


def construct():
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    
    best = _schoolbook(n, m, p)
    best_rank = len(best["U"][0])
    
    # Verify baseline
    if _exact_check(best["U"], best["V"], best["W"], T, n, m, p):
        pass
    
    # Try Winograd decomposition
    if time.time() < _DEADLINE - 20:
        cand = _winograd_4point(n, m, p)
        if len(cand["U"][0]) < best_rank:
            if _exact_check(cand["U"], cand["V"], cand["W"], T, n, m, p):
                best = cand
                best_rank = len(best["U"][0])
    
    # Try Strassen-like
    if time.time() < _DEADLINE - 15:
        cand = _try_strassen_4x4(n, m, p)
        if len(cand["U"][0]) < best_rank:
            if _exact_check(cand["U"], cand["V"], cand["W"], T, n, m, p):
                best = cand
                best_rank = len(best["U"][0])
    
    # Try greedy elimination on the best candidate
    if time.time() < _DEADLINE - 5:
        try:
            U, V, W, new_rank = _greedy_eliminate(
                best["U"], best["V"], best["W"], T, n, m, p,
                max_time=min(30, _DEADLINE - 5 - time.time())
            )
            if new_rank < best_rank:
                if _exact_check(U, V, W, T, n, m, p):
                    best = {"U": U, "V": V, "W": W}
                    best_rank = new_rank
        except:
            pass
    
    return best

# EVOLVE-BLOCK-END
