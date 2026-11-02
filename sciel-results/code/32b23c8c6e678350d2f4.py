"""Tensor decomposition for 2x2 by 2x5 matrix multiplication.

Family: two 2x2 Strassen blocks (B-columns 0-1 and 2-3) + schoolbook (B-column 4),
refined by sharing A-side rank-1 factors across the two blocks.  A verified
shared-factor construction is emitted when it is confirmed correct; otherwise the
program falls back to the known-correct rank-18 decomposition.
"""

DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def _verify(U, V, W, n, m, p):
    """Check sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all (i,j,k)."""
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                ui = U[i * m + j]
                vj = V[j * p + k]
                wk = W[k * n + i]
                for r in range(len(ui)):
                    s += ui[r] * vj[r] * wk[r]
                if abs(s - 1.0) > 1e-9:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break
    return ok

def _schoolbook_2x2x5():
    """Known-correct rank-20 schoolbook decomposition."""
    n, m, p = DIMENSIONS
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

def _rank18():
    """2 Strassen blocks (cols 0-1, 2-3) + schoolbook (col 4) = rank 18."""
    n, m, p = DIMENSIONS
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def add_strassen_block(t0, k0):
        k1 = k0 + 1
        # U (A) factors M1..M7
        U[0][t0 + 0] = 1; U[3][t0 + 0] = 1      # M1: A00+A11
        U[2][t0 + 1] = 1; U[3][t0 + 1] = 1      # M2: A10+A11
        U[0][t0 + 2] = 1                          # M3: A00
        U[3][t0 + 3] = 1                          # M4: A11
        U[0][t0 + 4] = 1; U[1][t0 + 4] = 1       # M5: A00+A01
        U[2][t0 + 5] = 1; U[0][t0 + 5] = -1      # M6: A10-A00
        U[1][t0 + 6] = 1; U[3][t0 + 6] = -1      # M7: A01-A11
        # V (B) factors
        V[0 * p + k0][t0 + 0] = 1; V[1 * p + k1][t0 + 0] = 1  # M1
        V[0 * p + k0][t0 + 1] = 1                            # M2
        V[0 * p + k1][t0 + 2] = 1; V[1 * p + k1][t0 + 2] = -1# M3
        V[1 * p + k0][t0 + 3] = 1; V[0 * p + k0][t0 + 3] = -1# M4
        V[1 * p + k1][t0 + 4] = 1                            # M5
        V[0 * p + k0][t0 + 5] = 1; V[0 * p + k1][t0 + 5] = 1 # M6
        V[1 * p + k0][t0 + 6] = 1; V[1 * p + k1][t0 + 6] = 1 # M7
        # W (C) factors
        W[k0 * n + 0][t0 + 0] = 1; W[k1 * n + 1][t0 + 0] = 1  # M1
        W[k0 * n + 1][t0 + 1] = 1; W[k1 * n + 1][t0 + 1] = -1 # M2
        W[k1 * n + 0][t0 + 2] = 1; W[k1 * n + 1][t0 + 2] = 1  # M3
        W[k0 * n + 0][t0 + 3] = 1; W[k0 * n + 1][t0 + 3] = 1  # M4
        W[k0 * n + 0][t0 + 4] = -1; W[k1 * n + 0][t0 + 4] = 1 # M5
        W[k1 * n + 1][t0 + 5] = 1                            # M6
        W[k0 * n + 0][t0 + 6] = 1                            # M7

    add_strassen_block(0, 0)
    add_strassen_block(7, 2)
    # schoolbook for column 4
    t = 14
    for (i, j) in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        U[i * m + j][t] = 1
        V[j * p + 4][t] = 1
        W[4 * n + i][t] = 1
        t += 1
    return {"U": U, "V": V, "W": W}

def _shared_factor():
    """Shared-factor construction: 11 shared A-factors + 12 block-local
    (6 per block) + 4 schoolbook for column 4 = rank 27, then verified."""
    n, m, p = DIMENSIONS
    # Shared A-factors (indices 0..10)
    # 0: A00+A11, 1: A10+A11, 2: A00, 3: A11, 4: A00+A01, 5: A10-A00,
    # 6: A01-A11, 7: A00-A10, 8: A10, 9: A01, 10: A00+A10
    shared_A = {
        0: [(0, 1), (3, 1)],
        1: [(2, 1), (3, 1)],
        2: [(0, 1)],
        3: [(3, 1)],
        4: [(0, 1), (1, 1)],
        5: [(2, 1), (0, -1)],
        6: [(1, 1), (3, -1)],
        7: [(0, 1), (2, -1)],
        8: [(2, 1)],
        9: [(1, 1)],
        10: [(0, 1), (2, 1)],
    }
    # Block-local A-factors (indices 11..22)
    # block 0: 11: A00+A10, 12: A11-A01, 13: A00+A11, 14: A01, 15: A10, 16: A00
    # block 1: 17: A00+A10, 18: A11-A01, 19: A00+A11, 20: A01, 21: A10, 22: A00
    local_A = {
        11: [(0, 1), (2, 1)], 12: [(3, 1), (1, -1)], 13: [(0, 1), (3, 1)],
        14: [(1, 1)], 15: [(2, 1)], 16: [(0, 1)],
        17: [(0, 1), (2, 1)], 18: [(3, 1), (1, -1)], 19: [(0, 1), (3, 1)],
        20: [(1, 1)], 21: [(2, 1)], 22: [(0, 1)],
    }
    # Per-block B-factors (V) and C-factors (W) for the 6 local terms
    # term order: (A00+A10), (A11-A01), (A00+A11), A01, A10, A00
    def block_V(k0):
        k1 = k0 + 1
        return [
            [(0 * p + k0, 1), (1 * p + k0, 1)],
            [(0 * p + k1, 1), (1 * p + k1, -1)],
            [(0 * p + k0, 1), (1 * p + k1, 1)],
            [(0 * p + k1, 1)],
            [(1 * p + k0, 1)],
            [(0 * p + k0, 1)],
        ]
    def block_W(k0):
        k1 = k0 + 1
        return [
            [(k0 * n + 0, 1), (k1 * n + 1, 1)],
            [(k0 * n + 1, 1), (k1 * n + 0, 1)],
            [(k0 * n + 0, 1), (k1 * n + 0, 1), (k1 * n + 1, 1)],
            [(k0 * n + 1, 1), (k1 * n + 1, -1)],
            [(k0 * n + 1, 1)],
            [(k0 * n + 0, 1)],
        ]

    rank = 27
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for idx, pairs in shared_A.items():
        for (a, c) in pairs:
            U[a][idx] += c
    for idx, pairs in local_A.items():
        for (a, c) in pairs:
            U[a][idx] += c
    for b, k0 in [(0, 0), (1, 2)]:
        base = 11 + 6 * b
        for t in range(6):
            for (v, c) in block_V(k0)[t]:
                V[v][base + t] += c
            for (w, c) in block_W(k0)[t]:
                W[w][base + t] += c
    # schoolbook for column 4 (indices 23..26)
    for t, (i, j) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        U[i * m + j][23 + t] = 1
        V[j * p + 4][23 + t] = 1
        W[4 * n + i][23 + t] = 1
    return {"U": U, "V": V, "W": W}

def _als_search(rank, max_iter=200, tol=1e-10, seed=42):
    """ALS-based tensor decomposition search with rational recovery."""
    n, m, p = DIMENSIONS
    np.random.seed(seed)
    
    # Build target tensor T[i,j,k] = delta_{i,j,k} for matrix mult
    # Actually, for matrix multiplication: T[i,j,k] = 1 if we're computing C[i,k] += A[i,j]*B[j,k]
    # The tensor is T[i,j,k] = 1 for all i,j,k in the standard basis
    # We need: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i,j,k
    
    # Initialize with random factors
    U = np.random.randn(n*m, rank)
    V = np.random.randn(m*p, rank)
    W = np.random.randn(p*n, rank)
    
    # Normalize to avoid scale issues
    for r in range(rank):
        norm = np.linalg.norm(U[:, r])
        if norm > 1e-12:
            U[:, r] /= norm
            V[:, r] *= norm
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        # Update U
        for i in range(n*m):
            # For each row i of U, solve: U[i,:] * (V * W) = target
            # Reshape: for fixed i, we need sum_r U[i,r] * V[j*p+k,r] * W[k*n+i',r]
            # This is complex; use gradient-free ALS step
            
            # Compute gradient for U[i,:]
            # d/dU[i,r] sum_{j,k} (sum_r' U[i,r']*V[j*p+k,r']*W[k*n+i',r']) - 1
            # where i' is determined by i = i'*m + j
            i_idx, j_idx = divmod(i, m)
            
            grad = np.zeros(rank)
            for j in range(m):
                for k in range(p):
                    # Target: 1 if (i_idx, j, k) corresponds to a valid entry
                    # In matrix mult: C[i_idx, k] = sum_j A[i_idx, j] * B[j, k]
                    # So T[i_idx, j, k] = 1 for all i_idx, j, k
                    target = 1.0
                    current = 0.0
                    for r in range(rank):
                        current += U[i, r] * V[j*p+k, r] * W[k*n+i_idx, r]
                    # Gradient w.r.t. U[i, r]
                    for r in range(rank):
                        grad[r] += (current - target) * V[j*p+k, r] * W[k*n+i_idx, r]
            
            # Update U[i] using gradient descent
            lr = 0.01
            U[i] -= lr * grad
        
        # Update V
        for j in range(m*p):
            j_idx, k_idx = divmod(j, p)
            grad = np.zeros(rank)
            for i in range(n):
                for k in range(p):
                    target = 1.0
                    current = 0.0
                    for r in range(rank):
                        current += U[i*m+j_idx, r] * V[j, r] * W[k*n+i, r]
                    for r in range(rank):
                        grad[r] += (current - target) * U[i*m+j_idx, r] * W[k*n+i, r]
            
            lr = 0.01
            V[j] -= lr * grad
        
        # Update W
        for k in range(p*n):
            k_idx, i_idx = divmod(k, n)
            grad = np.zeros(rank)
            for i in range(n):
                for j in range(m):
                    target = 1.0
                    current = 0.0
                    for r in range(rank):
                        current += U[i*m+j, r] * V[j*p+k_idx, r] * W[k, r]
                    for r in range(rank):
                        grad[r] += (current - target) * U[i*m+j, r] * V[j*p+k_idx, r]
            
            lr = 0.01
            W[k] -= lr * grad
        
        # Compute error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    err += (s - 1.0) ** 2
        err = np.sqrt(err)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < tol:
            break
    
    # Rational recovery
    def recover_rational(x, max_denom=100):
        """Recover a rational number close to x."""
        if abs(x) < 1e-12:
            return 0
        # Try simple fractions
        for den in range(1, max_denom + 1):
            num = round(x * den)
            if abs(x - num/den) < 1e-6:
                return Fraction(num, den)
        return Fraction(str(x))
    
    # Convert to rational
    U_rat = [[0] * rank for _ in range(n*m)]
    V_rat = [[0] * rank for _ in range(m*p)]
    W_rat = [[0] * rank for _ in range(p*n)]
    
    for i in range(n*m):
        for r in range(rank):
            U_rat[i][r] = recover_rational(best_U[i, r])
    for j in range(m*p):
        for r in range(rank):
            V_rat[j][r] = recover_rational(best_V[j, r])
    for k in range(p*n):
        for r in range(rank):
            W_rat[k][r] = recover_rational(best_W[k, r])
    
    # Verify
    if _verify(U_rat, V_rat, W_rat, n, m, p):
        return {"U": U_rat, "V": V_rat, "W": W_rat, "rank": rank}
    
    return None

def construct():
    deadline = time.time() + 110  # safety margin vs 120 s limit
    n, m, p = DIMENSIONS
    
    # 1) Try ALS-based search for lower rank (15, 16)
    for target_rank in [15, 16]:
        if time.time() > deadline - 60:
            break
        result = _als_search(target_rank, max_iter=100, seed=42)
        if result is not None:
            return result
    
    # 2) Try the verified shared-factor construction (improvement attempt)
    if time.time() < deadline - 30:
        cand = _shared_factor()
        if _verify(cand["U"], cand["V"], cand["W"], n, m, p):
            return cand

    # 3) known-correct rank-18 fallback
    if time.time() < deadline:
        cand = _rank18()
        if _verify(cand["U"], cand["V"], cand["W"], n, m, p):
            return cand

    # 4) ultimate schoolbook fallback
    return _schoolbook_2x2x5()
# EVOLVE-BLOCK-END
