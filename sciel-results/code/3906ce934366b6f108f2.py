"""Search for a rank-minimal exact tensor decomposition for multiplying a 2x5 matrix
by a 5x5 matrix.

Strategy:
  * Start from the schoolbook decomposition (rank = n*m*p = 50).
  * Use a structured parameterization that exploits the small dimension n=2.
  * Use randomized local search with rank reduction attempts.
  * Keep an explicit deadline guard.

The returned dict has keys 'U','V','W' with shapes (n*m,rank),(m*p,rank),(p*n,rank)
and integer/half-integer entries that exactly decompose the matrix multiplication tensor.
"""

import time
import random
import itertools

DIMENSIONS = [2, 5, 5]


# ---------- helpers ----------

def _tensor_flat(n, m, p):
    """Return the flat list of the matmul tensor T[(i,k), j] style flattened."""
    return n, m, p


def _is_half_int(x):
    return abs(2 * x - round(2 * x)) < 1e-6


def _validate(U, V, W, n, m, p, tol=1e-6):
    """Check that the factors exactly (within tol) reproduce T[i,j,k]=1 for all (i,j,k).

    Also check half-integer entries and correct shapes.
    """
    rank = len(U[0]) if U else 0
    if len(U) != n * m:
        return False, "U rows"
    if len(V) != m * p:
        return False, "V rows"
    if len(W) != p * n:
        return False, "W rows"
    for row in U:
        if len(row) != rank:
            return False, "U cols"
    for row in V:
        if len(row) != rank:
            return False, "V cols"
    for row in W:
        if len(row) != rank:
            return False, "W cols"
    # half-integer check
    for row in U:
        for x in row:
            if not _is_half_int(x):
                return False, "U not half-int"
    for row in V:
        for x in row:
            if not _is_half_int(x):
                return False, "V not half-int"
    for row in W:
        for x in row:
            if not _is_half_int(x):
                return False, "W not half-int"
    # reconstruction
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > tol:
                    return False, "recon"
    return True, "ok"


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1.0
                V[j * p + k][c] = 1.0
                W[k * n + i][c] = 1.0
                c += 1
    return {"U": U, "V": V, "W": W}


# ---------- search ----------

def _rand_factors(rank, n, m, p, scale=1.0):
    U = [[random.uniform(-scale, scale) for _ in range(rank)] for _ in range(n * m)]
    V = [[random.uniform(-scale, scale) for _ in range(rank)] for _ in range(m * p)]
    W = [[random.uniform(-scale, scale) for _ in range(rank)] for _ in range(p * n)]
    return U, V, W


def _local_search(U, V, W, n, m, p, iters=200, seed=None):
    """Randomized perturbation + coordinate re-solve to try to reduce rank."""
    best = (len(U[0]), U, V, W)
    rank = best[0]
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    ok, _ = _validate(U, V, W, n, m, p)
    if not ok:
        return best
    for _ in range(iters):
        r = random.randrange(rank)
        d = random.gauss(0, 0.3)
        for i in range(n * m):
            U[i][r] += d
        for j in range(m * p):
            V[j][r] -= d * 0.5
        ok, _ = _validate(U, V, W, n, m, p, tol=1e-4)
        if ok:
            best = (rank, [row[:] for row in U], [row[:] for row in V], [row[:] for row in W])
        # revert
        for i in range(n * m):
            U[i][r] -= d
        for j in range(m * p):
            V[j][r] += d * 0.5
    return best


def _structured_search(n, m, p, target_rank, attempts=20, deadline=None):
    """Try to find a valid decomposition of the given target rank using structured parameterization."""
    start = time.time()
    
    # For n=2, we can exploit the structure
    # Try parameterization where each column is associated with a (j,k) pair
    # and the factors have a specific form that respects the multiplication structure
    
    for attempt in range(attempts):
        if deadline and time.time() > deadline:
            break
        
        # Initialize with a structured pattern
        U = [[0.0] * target_rank for _ in range(n * m)]
        V = [[0.0] * target_rank for _ in range(m * p)]
        W = [[0.0] * target_rank for _ in range(p * n)]
        
        # Try different random seeds for the structured pattern
        rng = random.Random(attempt * 12345 + 678)
        
        # Assign random half-integer values with some structure
        # The key insight: for the diagonal terms to sum to 1, we need
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        # and for off-diagonal terms to sum to 0
        
        # Try a sparse pattern first
        for c in range(target_rank):
            # Choose which (i,j,k) this column contributes to
            i = rng.randint(0, n - 1)
            j = rng.randint(0, m - 1)
            k = rng.randint(0, p - 1)
            
            # Set values that might help
            val = rng.choice([1.0, -1.0, 0.5, -0.5])
            U[i * m + j][c] = val
            V[j * p + k][c] = 1.0
            W[k * n + i][c] = 1.0 / val if val != 0 else 1.0
        
        # Validate
        ok, _ = _validate(U, V, W, n, m, p)
        if ok:
            return {"U": U, "V": V, "W": W}
        
        # Try local search from this initialization
        local_rank, lU, lV, lW = _local_search(U, V, W, n, m, p, iters=10)
        ok, _ = _validate(lU, lV, lW, n, m, p)
        if ok:
            return {"U": lU, "V": lV, "W": lW}
    
    return None


def construct():
    n, m, p = DIMENSIONS
    deadline = 120.0
    start = time.time()
    safety = 15.0  # seconds margin

    best = _schoolbook(n, m, p)
    best_rank = n * m * p

    def remaining():
        return deadline - safety - (time.time() - start)

    rng = random.Random(12345)

    # 1) Try structured search at various ranks
    # Focus on ranks that might be achievable given the structure
    candidate_ranks = [40, 35, 30, 25, 20, 15]
    
    for target_rank in candidate_ranks:
        if remaining() <= 0:
            break
        
        # Try structured search
        result = _structured_search(n, m, p, target_rank, attempts=10, deadline=start + deadline - safety)
        if result:
            best = result
            best_rank = target_rank
            break
    
    # 2) If still at schoolbook, try random search with local improvement
    if remaining() > 0 and best_rank == n * m * p:
        # Try random restarts at decreasing rank
        for target_rank in range(best_rank - 10, best_rank - 1):
            if remaining() <= 0:
                break
            
            for a in range(5):
                if remaining() <= 0:
                    break
                
                U, V, W = _rand_factors(target_rank, n, m, p, scale=2.0)
                local_best_rank, lU, lV, lW = _local_search(
                    U, V, W, n, m, p, iters=20
                )
                ok, _ = _validate(lU, lV, lW, n, m, p)
                if ok and local_best_rank < best_rank:
                    best_rank = local_best_rank
                    best = {"U": lU, "V": lV, "W": lW}
                    break
    
    # 3) Final check: try to reduce schoolbook by merging columns
    if remaining() > 0 and best_rank == n * m * p:
        U = [row[:] for row in best["U"]]
        V = [row[:] for row in best["V"]]
        W = [row[:] for row in best["W"]]
        
        # Try merging pairs of columns
        for r1 in range(best_rank - 1):
            if remaining() <= 0:
                break
            for r2 in range(r1 + 1, best_rank):
                if remaining() <= 0:
                    break
                
                # Try averaging
                for i in range(n * m):
                    U[i][r1] = (U[i][r1] + U[i][r2]) / 2.0
                for j in range(m * p):
                    V[j][r1] = (V[j][r1] + V[j][r2]) / 2.0
                for i in range(p * n):
                    W[i][r1] = (W[i][r1] + W[i][r2]) / 2.0
                
                # Drop r2
                for i in range(n * m):
                    U[i].pop(r2)
                for j in range(m * p):
                    V[j].pop(r2)
                for i in range(p * n):
                    W[i].pop(r2)
                
                ok, _ = _validate(U, V, W, n, m, p)
                if ok:
                    best = {"U": U, "V": V, "W": W}
                    best_rank = len(U[0])
                    break
                
                # Revert
                U = [row[:] for row in best["U"]]
                V = [row[:] for row in best["V"]]
                W = [row[:] for row in best["W"]]
    
    return best


# EVOLVE-BLOCK-END
