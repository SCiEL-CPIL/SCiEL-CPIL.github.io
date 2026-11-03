"""Optimized tensor decomposition for 3x3 * 3x6 matrix multiplication."""
import time
import random

DIMENSIONS = [3, 3, 6]
_DEADLINE = None
_TIME_LIMIT = 120.0  # wall-clock limit in seconds

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return the standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that sum_r U[:,r] V[:,r] W[:,r]^T equals the matmul tensor."""
    rank = len(U[0])
    # The tensor T[i*m+j, j2*p+k, r_idx] where r_idx = k*n+i
    # Actually the convention: W[k*n + i, r]
    # T[A, B, C] where A = i*m + j (row of A matrix), B = j2*p + k (col of B matrix), C = k*n + i (row of C matrix)
    # T[A, B, C] = 1 if A corresponds to (i,j), B to (j,k) with same j, and C to (k,i)
    # i.e., T[i*m+j, j*p+k, k*n+i] = 1
    # Sum_r U[A,r] * V[B,r] * W[C,r] should equal T[A,B,C]
    
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                for k in range(p):
                    A = i * m + j
                    B = j2 * p + k
                    C = k * n + i
                    expected = 1.0 if (j == j2) else 0.0
                    actual = 0.0
                    for r in range(rank):
                        actual += U[A][r] * V[B][r] * W[C][r]
                    if abs(actual - expected) > tol:
                        return False
    return True


def _try_rank(n, m, p, rank, seed=42, max_iter=200, timeout=None):
    """
    Try to find a rank-`rank` decomposition using iterative refinement.
    Returns (U, V, W) as lists or None.
    """
    global _DEADLINE
    if _DEADLINE is not None and time.time() > _DEADLINE:
        return None
    
    rng = random.Random(seed)
    
    # Initialize with random small values
    U = [[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(n * m)]
    V = [[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(m * p)]
    W = [[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(p * n)]
    
    # Precompute the target tensor (sparse representation)
    # Target: T[i*m+j, j*p+k, k*n+i] = 1
    # We'll compute residuals on the fly
    
    # Iterative refinement: coordinate descent / alternating minimization
    # For each r, the contribution is U[:,r] * V[:,r] * W[:,r]^T
    # The target is fixed. We can't easily do coordinate descent on rank directly.
    # Instead, use a simple gradient-free local search on the entries.
    
    # Actually, a simpler approach: random restarts + local search on entries
    # Since we want integer/half-integer entries, we can round at the end and verify.
    
    best_err = float('inf')
    best_U, best_V, best_W = None, None, None
    
    for it in range(max_iter):
        if timeout is not None and time.time() > timeout:
            break
        
        # Pick a random entry to perturb
        mat_idx = rng.randint(0, 2)  # 0=U, 1=V, 2=W
        if mat_idx == 0:
            row = rng.randint(0, n * m - 1)
            col = rng.randint(0, rank - 1)
            delta = rng.choice([-0.5, 0.5, -1.0, 1.0, -0.1, 0.1, -0.01, 0.01])
            old_val = U[row][col]
            U[row][col] += delta
        elif mat_idx == 1:
            row = rng.randint(0, m * p - 1)
            col = rng.randint(0, rank - 1)
            delta = rng.choice([-0.5, 0.5, -1.0, 1.0, -0.1, 0.1, -0.01, 0.01])
            old_val = V[row][col]
            V[row][col] += delta
        else:
            row = rng.randint(0, p * n - 1)
            col = rng.randint(0, rank - 1)
            delta = rng.choice([-0.5, 0.5, -1.0, 1.0, -0.1, 0.1, -0.01, 0.01])
            old_val = W[row][col]
            W[row][col] += delta
        
        # Compute error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for j2 in range(m):
                    for k in range(p):
                        A = i * m + j
                        B = j2 * p + k
                        C = k * n + i
                        expected = 1.0 if (j == j2) else 0.0
                        actual = 0.0
                        for r in range(rank):
                            actual += U[A][r] * V[B][r] * W[C][r]
                        err += (actual - expected) ** 2
        
        if err < best_err:
            best_err = err
            best_U = [row[:] for row in U]
            best_V = [row[:] for row in V]
            best_W = [row[:] for row in W]
        else:
            # Revert
            if mat_idx == 0:
                U[row][col] = old_val
            elif mat_idx == 1:
                V[row][col] = old_val
            else:
                W[row][col] = old_val
        
        # Early stop if error is negligible
        if best_err < 1e-10:
            break
    
    if best_err < 1e-9:
        return best_U, best_V, best_W
    return None


def construct():
    global _DEADLINE
    _DEADLINE = time.time() + _TIME_LIMIT - 5  # 5s safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    fallback = _schoolbook(n, m, p)
    
    # Try to find a lower-rank decomposition
    # Lower bounds: max(n*m, m*p, p*n) = max(9, 18, 18) = 18
    # Known: rank(3x3*3x3) <= 23, so rank(3x3*3x6) <= 46
    # Try ranks from 18 up to 53
    
    best_result = fallback
    best_rank = n * m * p
    
    for rank in range(18, best_rank):
        if time.time() > _DEADLINE:
            break
        
        # Try a few random seeds
        for seed in [42, 123, 789]:
            if time.time() > _DEADLINE:
                break
            
            result = _try_rank(n, m, p, rank, seed=seed, max_iter=500, timeout=_DEADLINE)
            if result is not None:
                U, V, W = result
                # Verify
                if _verify(U, V, W, n, m, p):
                    best_result = {"U": U, "V": V, "W": W}
                    best_rank = rank
                    break
        
        if best_rank == rank:
            break
    
    return best_result

# EVOLVE-BLOCK-END
