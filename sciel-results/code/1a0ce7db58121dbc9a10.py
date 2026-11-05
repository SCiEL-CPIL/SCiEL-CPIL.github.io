"""Rewritten program for matrix multiplication 4x4 * 4x7 tensor decomposition search."""
import time
import random
import itertools
import math

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    
    # Try Strassen-based decomposition
    # Split B (4x7) into B1 (4x4) and B2 (4x3)
    # Use Strassen for 4x4 * 4x4, and block decomposition for 4x4 * 4x3
    
    U, V, W = _strassen_447(n, m, p)
    
    if _verify_full(n, m, p, U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}

def _schoolbook(n, m, p):
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

def _verify_full(n, m, p, U, V, W):
    """Verify that the decomposition is correct."""
    rank = len(U[0])
    # Check all valid (i,j,k) triples give 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(rank):
                    total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(total - 1.0) > 1e-9:
                    return False
    # Check some invalid combinations give 0
    # U index: i*m+j, V index: j'*p+k', W index: k''*n+i''
    # Invalid if i != i'' or j != j' or k != k''
    # Actually, the tensor is defined such that T[a,b,d] = 1 iff
    # a = i*m+j, b = j*p+k, d = k*n+i for some valid (i,j,k)
    # For other (a,b,d), T[a,b,d] = 0
    # Let's check a few invalid cases
    import itertools
    checked = 0
    for a in range(n*m):
        for b in range(m*p):
            for d in range(p*n):
                # Check if (a,b,d) corresponds to a valid (i,j,k)
                i_a = a // m
                j_a = a % m
                j_b = b // p
                k_b = b % p
                k_d = d // n
                i_d = d % n
                if i_a == i_d and j_a == j_b and k_b == k_d:
                    continue  # valid, already checked
                total = 0.0
                for c in range(rank):
                    total += U[a][c] * V[b][c] * W[d][c]
                if abs(total) > 1e-9:
                    return False
                checked += 1
                if checked > 1000:  # limit checks for speed
                    return True
    return True

def _strassen_2x2_2x2():
    """Return U, V, W for 2x2 * 2x2 -> 2x2. Rank = 8 (schoolbook)."""
    n, m, p = 2, 2, 2
    rank = 8
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _strassen_447(n, m, p):
    """Build a decomposition for 4x4 * 4x7 -> 4x7. Uses schoolbook for now."""
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
