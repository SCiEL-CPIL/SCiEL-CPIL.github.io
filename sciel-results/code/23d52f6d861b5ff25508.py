"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _try_rank(rank, n, m, p, max_iter=5000, seed=42):
    """Try to find a decomposition of given rank using iterative refinement."""
    rng = np.random.RandomState(seed)
    U = rng.randn(n * m, rank)
    V = rng.randn(m * p, rank)
    W = rng.randn(p * n, rank)
    
    # Target tensor values
    targets = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                targets[(i*m+j, j*p+k, k*n+i)] = 1.0
    
    for it in range(max_iter):
        # Update U
        for a in range(n * m):
            for c in range(rank):
                num = 0.0
                den = 0.0
                for (ta, tb, tc), tv in targets.items():
                    if ta == a:
                        num += tv * V[tb, c] * W[tc, c]
                        den += V[tb, c] * W[tc, c] * V[tb, c] * W[tc, c]
                if den > 1e-12:
                    U[a, c] = num / den
        
        # Update V
        for b in range(m * p):
            for c in range(rank):
                num = 0.0
                den = 0.0
                for (ta, tb, tc), tv in targets.items():
                    if tb == b:
                        num += tv * U[ta, c] * W[tc, c]
                        den += U[ta, c] * W[tc, c] * U[ta, c] * W[tc, c]
                if den > 1e-12:
                    V[b, c] = num / den
        
        # Update W
        for cc in range(p * n):
            for c in range(rank):
                num = 0.0
                den = 0.0
                for (ta, tb, tc), tv in targets.items():
                    if tc == cc:
                        num += tv * U[ta, c] * V[tb, c]
                        den += U[ta, c] * V[tb, c] * U[ta, c] * V[tb, c]
                if den > 1e-12:
                    W[cc, c] = num / den
        
        # Check convergence
        if it % 100 == 0:
            err = 0.0
            for (ta, tb, tc), tv in targets.items():
                s = sum(U[ta, c] * V[tb, c] * W[tc, c] for c in range(rank))
                err += (s - tv) ** 2
            if err < 1e-12:
                return U.tolist(), V.tolist(), W.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin
    
    # Try ranks from a reasonable lower bound up to schoolbook
    # Lower bound: at least n*m, m*p, p*n must be <= rank for the matrices to have enough rows
    # Actually rank must be >= max(n*m, m*p, p*n) / something... 
    # The tensor has 44 non-zero entries, so rank >= 1, but practically rank >= max(n*m, m*p, p*n) is not required
    # Let's try ranks from 20 up to 44
    schoolbook_rank = n * m * p
    
    for rank in range(20, schoolbook_rank + 1):
        if time.time() > deadline - 5:
            break
        result = _try_rank(rank, n, m, p, max_iter=2000, seed=rank)
        if result is not None:
            U, V, W = result
            # Verify
            if _verify(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook
    rank = schoolbook_rank
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
# EVOLVE-BLOCK-END
