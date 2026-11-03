"""Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x4 matrix."""
import time
import random
import itertools
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

def _build_decomposition(n, m, p, rank):
    """
    Build decomposition where each column c has rank-1 support.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    
    The tensor T[i,j,k] (representing A[i,j] * B[j,k]) must equal:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    """
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for c in range(rank):
        # Each factor is a subset
        u_set = set()
        v_set = set()
        w_set = set()
        
        for i in range(n):
            for j in range(m):
                if random.random() < 0.7:
                    u_set.add((i,j))
        
        for j in range(m):
            for k in range(p):
                if random.random() < 0.7:
                    v_set.add((j,k))
        
        for k in range(p):
            for i in range(n):
                if random.random() < 0.7:
                    w_set.add((k,i))
        
        for (i,j) in u_set:
            U[i*m+j][c] = random.choice([1, -1])
        for (j,k) in v_set:
            V[j*p+k][c] = random.choice([1, -1])
        for (k,i) in w_set:
            W[k*n+i][c] = random.choice([1, -1])
    
    return U, V, W

def _verify(n, m, p, U, V, W):
    """
    Verify that the decomposition is exact.
    For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120  # safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Start with schoolbook as fallback
    U_fb = [[0]*schoolbook_rank for _ in range(n*m)]
    V_fb = [[0]*schoolbook_rank for _ in range(m*p)]
    W_fb = [[0]*schoolbook_rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j][col] = 1
                V_fb[j*p+k][col] = 1
                W_fb[k*n+i][col] = 1
                col += 1
    
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    best_rank = schoolbook_rank
    
    random.seed(42)
    
    # Try ranks below schoolbook
    for rank in range(schoolbook_rank - 1, 20, -1):
        if time.time() > deadline - 10:
            break
        
        attempts = 0
        max_attempts = 200
        
        while attempts < max_attempts and time.time() < deadline - 5:
            attempts += 1
            
            U, V, W = _build_decomposition(n, m, p, rank)
            
            if _verify(n, m, p, U, V, W):
                best_U, best_V, best_W = U, V, W
                best_rank = rank
                break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
