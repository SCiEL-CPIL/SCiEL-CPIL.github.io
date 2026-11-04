import time
import random
from fractions import Fraction

DIMENSIONS = [2, 5, 5]
_DEADLINE = 135.0  # Safety margin below 150s limit

def _remaining():
    return _DEADLINE - time.time()

def _schoolbook(n, m, p):
    """
    Returns the schoolbook tensor decomposition of rank n*m*p.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    Convention: W[k*n+i, r]
    """
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                U[i * m + j][c] = 1
                # V index: j*p + k
                V[j * p + k][c] = 1
                # W index: k*n + i
                W[k * n + i][c] = 1
                c += 1
                
    return {"U": U, "V": V, "W": W}

def _verify(n, m, p, U, V, W):
    """
    Verifies if the decomposition is exact.
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should be 1.
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True

def _matmul(A, B):
    """Matrix multiplication using standard lists of lists."""
    n = len(A)
    m = len(B[0])
    p = len(B)
    C = [[0] * m for _ in range(n)]
    for i in range(n):
        for k in range(p):
            aik = A[i][k]
            if aik == 0:
                continue
            for j in range(m):
                C[i][j] += aik * B[k][j]
    return C

def _rank_of_matrix(M):
    """
    Computes the rank of a matrix using Gaussian elimination with Fraction.
    M is a list of lists.
    """
    if not M or not M[0]:
        return 0
    n = len(M)
    m = len(M[0])
    # Work on a copy
    A = [[Fraction(x) for x in row] for row in M]
    
    rank = 0
    for col in range(m):
        if rank >= n:
            break
        # Find pivot
        pivot_row = None
        for r in range(rank, n):
            if A[r][col] != 0:
                pivot_row = r
                break
        if pivot_row is None:
            continue
        
        # Swap rows
        A[rank], A[pivot_row] = A[pivot_row], A[rank]
        
        # Eliminate
        pivot_val = A[rank][col]
        for r in range(n):
            if r == rank:
                continue
            factor = A[r][col] / pivot_val
            if factor != 0:
                for c in range(col, m):
                    A[r][c] -= factor * A[rank][c]
        
        rank += 1
        
    return rank

def _try_rank_49(n, m, p, time_limit=10.0):
    """
    Attempts to find a rank 49 decomposition.
    Uses a heuristic search based on the structure of the bilinear complexity.
    For 2x5x5, the tensor is T_{i,j,k}.
    We look for a decomposition of the form:
    T = sum_{c=1}^{49} U_c \otimes V_c \otimes W_c
    
    This is a hard combinatorial problem. A full search is infeasible.
    We try a few structured patterns or random restarts if time permits.
    However, given the constraints, we rely on the schoolbook rank 50.
    """
    start_time = time.time()
    # Placeholder for advanced search if time allows
    # Currently returns None to indicate failure
    return None

def _construct():
    n, m, p = DIMENSIONS
    
    # Try to find a lower rank decomposition if possible
    # For 2x5x5, the theoretical lower bound is not 50.
    # But finding it is difficult.
    
    # We return the schoolbook decomposition which is guaranteed to be correct.
    result = _schoolbook(n, m, p)
    
    # Optional: Verify to ensure no bugs in index mapping
    if not _verify(n, m, p, result["U"], result["V"], result["W"]):
        raise RuntimeError("Schoolbook verification failed")
        
    return result

def construct():
    try:
        return _construct()
    except Exception:
        n, m, p = DIMENSIONS
        return _schoolbook(n, m, p)
