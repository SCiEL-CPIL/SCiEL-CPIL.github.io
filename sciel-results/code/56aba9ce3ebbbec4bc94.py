"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import time
import itertools
from fractions import Fraction

DEADLINE = 140.0  # safety margin before 150s kill

def _verify_fast(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication using Fraction."""
    rank = len(U[0])
    # Check dimensions
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if any(len(row) != rank for row in U):
        return False
    if any(len(row) != rank for row in V):
        return False
    if any(len(row) != rank for row in W):
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(rank):
                    total += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if total != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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
    return U, V, W

def _try_merge_fast(U, V, W, n, m, p, c1, c2, factor):
    """Try to merge columns c1 and c2 by adding factor * c2 to c1 and removing c2."""
    rank = len(U[0])
    if c1 == c2:
        return None
    
    new_rank = rank - 1
    new_U = [[0] * new_rank for _ in range(n * m)]
    new_V = [[0] * new_rank for _ in range(m * p)]
    new_W = [[0] * new_rank for _ in range(p * n)]
    
    new_col = 0
    for c in range(rank):
        if c == c2:
            continue
        if c == c1:
            for i in range(n * m):
                new_U[i][new_col] = U[i][c1] + factor * U[i][c2]
            for j in range(m * p):
                new_V[j][new_col] = V[j][c1] + factor * V[j][c2]
            for k in range(p * n):
                new_W[k][new_col] = W[k][c1] + factor * W[k][c2]
        else:
            for i in range(n * m):
                new_U[i][new_col] = U[i][c]
            for j in range(m * p):
                new_V[j][new_col] = V[j][c]
            for k in range(p * n):
                new_W[k][new_col] = W[k][c]
        new_col += 1
    
    if _verify_fast(new_U, new_V, new_W, n, m, p):
        return new_U, new_V, new_W
    return None

def _construct_direct(n, m, p):
    """Try to construct a lower-rank decomposition using algebraic structure.
    
    For 2x5x5 matrix multiplication, we can try to exploit the fact that
    the output has only 2 rows. We use a structured approach:
    
    (AB)[i,k] = sum_j A[i,j] * B[j,k]
    
    We can write this as a sum of rank-1 terms in the tensor sense.
    """
    # For small dimensions, try known constructions
    # Rank lower bound: at least max(n*m, m*p, n*p) is not quite right
    # For M_{2,5,5}: known bounds suggest rank >= 2*5 = 10 (from n*m)
    # but actual rank may be higher
    
    # Try a structured decomposition
    # Idea: use the fact that we only need 2 output rows
    # (AB)[0,k] = sum_j A[0,j]*B[j,k]
    # (AB)[1,k] = sum_j A[1,j]*B[j,k]
    
    # We can try to find a decomposition of rank r where r < 50
    # by exploiting linear dependencies
    
    # For now, return schoolbook if no better found
    U, V, W = _schoolbook(n, m, p)
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_U, best_V, best_W = U, V, W
    best_rank = schoolbook_rank
    
    start_time = time.time()
    
    # Try to find lower rank decompositions via merging
    # Only try a limited number of merge attempts to stay within time
    max_merge_attempts = 20000
    attempts = 0
    factors_to_try = [Fraction(1), Fraction(-1), Fraction(1,2), Fraction(-1,2), Fraction(2), Fraction(-2), Fraction(3), Fraction(-3), Fraction(1,3), Fraction(-1,3)]
    
    current_U, current_V, current_W = U, V, W
    current_rank = best_rank
    
    while current_rank > 10 and time.time() < DEADLINE:
        merged = False
        for c1 in range(current_rank):
            if time.time() > DEADLINE:
                break
            for c2 in range(c1 + 1, current_rank):
                if time.time() > DEADLINE:
                    break
                for factor in factors_to_try:
                    attempts += 1
                    if attempts > max_merge_attempts:
                        break
                    result = _try_merge_fast(current_U, current_V, current_W, n, m, p, c1, c2, factor)
                    if result is not None:
                        current_U, current_V, current_W = result
                        current_rank -= 1
                        best_U, best_V, best_W = current_U, current_V, current_W
                        best_rank = current_rank
                        merged = True
                        break
                if merged or attempts > max_merge_attempts:
                    break
            if merged or attempts > max_merge_attempts:
                break
        
        if not merged:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
