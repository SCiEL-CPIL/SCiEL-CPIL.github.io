"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import time
import itertools
from fractions import Fraction

DEADLINE = 145.0  # safety margin before 150s kill

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
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
    
    # Build the tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # For matrix multiplication, T[i,j,k] should be 1 if we're computing (AB)[i,k] = sum_j A[i,j]*B[j,k]
    # Actually, the tensor for multiplying (n x m) by (m x p) to get (n x p) is:
    # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # And the decomposition must satisfy: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = T[i,j,k]
    
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

def _try_merge(U, V, W, n, m, p, c1, c2, factor):
    """Try to merge columns c1 and c2 by adding factor * c2 to c1 and removing c2."""
    rank = len(U[0])
    if c1 == c2:
        return None
    
    # Create new matrices without column c2
    new_rank = rank - 1
    new_U = [[0] * new_rank for _ in range(n * m)]
    new_V = [[0] * new_rank for _ in range(m * p)]
    new_W = [[0] * new_rank for _ in range(p * n)]
    
    new_col = 0
    for c in range(rank):
        if c == c2:
            continue
        if c == c1:
            # Merge c2 into c1
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
    
    # Verify
    if _verify(new_U, new_V, new_W, n, m, p):
        return new_U, new_V, new_W
    return None

def _search_rank(n, m, p, target_rank, start_rank):
    """Try to find a decomposition of rank target_rank starting from start_rank."""
    if start_rank < target_rank:
        return None
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    current_rank = start_rank
    
    # Try to reduce rank by merging columns
    # We'll try all pairs and a set of candidate factors
    factors_to_try = [Fraction(1), Fraction(-1), Fraction(1,2), Fraction(-1,2), Fraction(2), Fraction(-2)]
    
    while current_rank > target_rank:
        merged = False
        for c1 in range(current_rank):
            for c2 in range(c1 + 1, current_rank):
                for factor in factors_to_try:
                    result = _try_merge(U, V, W, n, m, p, c1, c2, factor)
                    if result is not None:
                        U, V, W = result
                        current_rank -= 1
                        merged = True
                        break
                if merged:
                    break
            if merged:
                break
        
        if not merged:
            return None
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_U, best_V, best_W = U, V, W
    best_rank = schoolbook_rank
    
    start_time = time.time()
    
    # Try to find lower rank decompositions
    for target_rank in range(schoolbook_rank - 1, 0, -1):
        if time.time() > DEADLINE:
            break
        
        result = _search_rank(n, m, p, target_rank, best_rank)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = target_rank
            # Continue trying to go lower
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
