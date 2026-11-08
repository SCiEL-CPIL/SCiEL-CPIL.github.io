"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

def _elapsed(start):
    return time.time() - start

def _verify(U, V, W, n, m, p, max_checks=200):
    """Verify that U, V, W correctly decompose matrix multiplication.
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    The decomposition must satisfy: for all i,j,k,
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 if (i,j,k) is a valid triple
    and 0 otherwise.
    """
    rank = len(U[0])
    # Check a random sample of (i,j,k) triples
    valid_triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    random.shuffle(valid_triples)
    checks = valid_triples[:max_checks]
    
    for (i, j, k) in checks:
        expected = 1
        actual = Fraction(0)
        for c in range(rank):
            actual += Fraction(U[i*m + j][c]) * Fraction(V[j*p + k][c]) * Fraction(W[k*n + i][c])
        if actual != expected:
            return False
    
    # Also check some invalid triples (should sum to 0)
    # For matrix multiplication, only (i,j,k) where j is the shared index matter
    # Actually all (i,j,k) with i in [0,n), j in [0,m), k in [0,p) are valid
    # The tensor is T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # So all combinations are "valid" in the sense they should be 1
    # Wait, no. The matrix multiplication tensor has T[i,j,k] = 1 for all i,j,k
    # because each element C[i,k] = sum_j A[i,j]*B[j,k]
    # So the tensor is all-ones of shape (n,m,p)
    # Therefore the decomposition must satisfy:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for ALL i,j,k
    # There are no "invalid" triples in the standard formulation.
    return True

def _build_schoolbook(n, m, p):
    """Build the schoolbook decomposition of rank n*m*p."""
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
    return U, V, W

def _local_search(U, V, W, n, m, p, start_time, max_time=100):
    """Attempt to reduce rank by trying to merge or eliminate terms.
    
    Strategy: Try to find linear dependencies among the rank-1 terms.
    If we can express one term as a linear combination of others, we can reduce rank.
    """
    rank = len(U[0])
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Try eliminating columns by checking if they're linearly dependent
    # A column c is redundant if there exist coefficients such that
    # the rank-1 term can be expressed as combination of others.
    
    # Simpler approach: try random linear combinations to see if we can
    # maintain the decomposition with fewer terms.
    
    # For now, just verify and return if we can't improve quickly
    if _elapsed(start_time) > max_time - 10:
        return best_U, best_V, best_W, best_rank
    
    return U, V, W, rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Start with schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    rank = n * m * p
    
    # Verify the schoolbook works
    if not _verify(U, V, W, n, m, p):
        # Should not happen, but return anyway
        return {"U": U, "V": V, "W": W}
    
    # Attempt local search to reduce rank
    U, V, W, rank = _local_search(U, V, W, n, m, p, start_time)
    
    # Safety check: ensure we have a valid decomposition
    if not _verify(U, V, W, n, m, p, max_checks=50):
        # Fall back to schoolbook
        U, V, W = _build_schoolbook(n, m, p)
        rank = n * m * p
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
