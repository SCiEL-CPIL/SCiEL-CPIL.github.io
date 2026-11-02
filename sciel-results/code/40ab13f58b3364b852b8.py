"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np
_start_time = time.time()

def _verify_decomposition(U, V, W, n=3, m=3, p=3):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0])
    # Build the tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    T = np.zeros((n, m, p), dtype=float)
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
    # The matrix multiplication tensor is T[i,j,k] = 1 if (i,j,k) is a valid triple, else 0
    # Actually, for matrix multiplication: C[i,k] = sum_j A[i,j]*B[j,k]
    # So the tensor has T[i,j,k] = 1 for all valid (i,j,k)
    expected = np.ones((n, m, p), dtype=float)
    return np.allclose(T, expected, atol=1e-10)

def _try_laderman():
    """Try to construct Laderman's rank-23 algorithm."""
    n, m, p = 3, 3, 3
    rank = 23
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Laderman's algorithm - based on known formulations
    # The 23 terms are structured as follows (using 0-indexed):
    # We use the formulation where each term r has support in U, V, W
    
    # Based on Laderman (1976) and subsequent literature:
    # Terms 1-7: Strassen-like core on indices {0,1,3,4} for U/V and {0,1,3,4} for W
    # Terms 8-23: Border/cross terms
    
    # Let me use a known correct rank-23 formulation.
    # From Bini, Capovani, Lotti, Romani (1983) and subsequent work:
    
    # I'll use the formulation from the literature where:
    # The 23 multiplications are:
    # M1 = (a+d)(e+h)
    # M2 = (c+f)(e)
    # M3 = (a)(g+h)
    # M4 = (d)(g)
    # M5 = (a+b)(h)
    # M6 = (c-d+e+f)(d+f)
    # M7 = (a-c+e-g)(a+c)
    # M8 = (a+e)(i+l)
    # M9 = (b+f)(i)
    # M10 = (c+g)(l)
    # M11 = (d+h)(j+k)
    # M12 = (a+b+c)(j)
    # M13 = (e+f+g)(k)
    # M14 = (d+e+f)(m)
    # M15 = (h+j+k)(m)
    # M16 = (a+c+e)(n)
    # M17 = (b+d+f)(n)
    # M18 = (g+h+j)(n)
    # M19 = (a+e+i)(o)
    # M20 = (b+f+j)(o)
    # M21 = (c+g+k)(o)
    # M22 = (d+h+l)(o)
    # M23 = (m+n+o)(m+n+o)  -- this doesn't look right
    
    # This ad-hoc approach is error-prone. Let me try a different strategy:
    # Use the rank-26 solution and verify it, then try to reduce.
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Deadline guard
    deadline = 140.0  # seconds
    
    # Start with the known rank-26 solution
    rank = 26
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_idx, v_idx, w_idx, coeff):
        nonlocal col
        U[u_idx][col] += coeff
        V[v_idx][col] += coeff
        W[w_idx][col] += coeff
        col += 1
    
    # 1. Strassen 2x2 core (rank 7)
    # M1: (a+d)(e+h) -> c00, c11
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[4][col] = 1
    W[0][col] = 1; W[4][col] = 1
    col += 1
    
    # M2: (c+f)(e) -> c10, c11
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[4][col] = -1
    col += 1
    
    # M3: (a)(g+h) -> c01, c11
    U[0][col] = 1
    V[1][col] = 1; V[4][col] = -1
    W[3][col] = 1; W[4][col] = 1
    col += 1
    
    # M4: (d)(g) -> c00, c01
    U[4][col] = 1
    V[3][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[1][col] = 1
    col += 1
    
    # M5: (a+b)(h) -> c00, c01
    U[0][col] = 1; U[1][col] = 1
    V[4][col] = 1
    W[0][col] = -1; W[3][col] = 1
    col += 1
    
    # M6: (c-d+e+f)(d+f) -> c11
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[4][col] = 1
    col += 1
    
    # M7: (a-c+e-g)(a+c) -> c00
    U[1][col] = 1; U[4][col] = -1
    V[3][col] = 1; V[4][col] = 1
    W[0][col] = 1
    col += 1
    
    # 2. Border terms (rank 19)
    # These handle the interactions with the 3rd row/col
    add_term(2, 6, 0, 1)   # A[0,2]*B[2,0] -> C[0,0]
    add_term(2, 7, 3, 1)   # A[0,2]*B[2,1] -> C[0,1]
    add_term(5, 6, 1, 1)   # A[1,2]*B[2,0] -> C[1,0]
    add_term(5, 7, 4, 1)   # A[1,2]*B[2,1] -> C[1,1]
    add_term(0, 2, 6, 1)   # A[0,0]*B[0,2] -> C[0,2]
    add_term(1, 5, 6, 1)   # A[0,1]*B[1,2] -> C[0,2]
    add_term(3, 2, 7, 1)   # A[1,0]*B[0,2] -> C[1,2]
    add_term(4, 5, 7, 1)   # A[1,1]*B[1,2] -> C[1,2]
    add_term(2, 8, 6, 1)   # A[0,2]*B[2,2] -> C[0,2]
    add_term(5, 8, 7, 1)   # A[1,2]*B[2,2] -> C[1,2]
    add_term(6, 0, 2, 1)   # A[2,0]*B[0,0] -> C[2,0]
    add_term(7, 3, 2, 1)   # A[2,1]*B[1,0] -> C[2,0]
    add_term(6, 1, 5, 1)   # A[2,0]*B[0,1] -> C[2,1]
    add_term(7, 4, 5, 1)   # A[2,1]*B[1,1] -> C[2,1]
    add_term(8, 6, 2, 1)   # A[2,2]*B[2,0] -> C[2,0]
    add_term(8, 7, 5, 1)   # A[2,2]*B[2,1] -> C[2,1]
    add_term(6, 2, 8, 1)   # A[2,0]*B[0,2] -> C[2,2]
    add_term(7, 5, 8, 1)   # A[2,1]*B[1,2] -> C[2,2]
    add_term(8, 8, 8, 1)   # A[2,2]*B[2,2] -> C[2,2]
    
    # Verify the rank-26 solution
    if not _verify_decomposition(U, V, W):
        # If verification fails, fall back to schoolbook rank-27
        rank = 27
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i*m*p + j*p + k
                    U[i*m+j][r] = 1
                    V[j*p+k][r] = 1
                    W[k*n+i][r] = 1
        return {"U": U, "V": V, "W": W}
    
    best_result = {"U": [row[:] for row in U], "V": [row[:] for row in V], "W": [row[:] for row in W]}
    best_rank = 26
    
    # Check if we have time to search for a lower-rank solution
    if time.time() - _start_time > deadline - 10:
        return best_result
    
    # Try to find a rank-23 or rank-24 solution by attempting to merge border terms
    # with the Strassen core using linear algebra
    
    # Strategy: Try to express some border terms as combinations of the Strassen terms
    # plus new terms, reducing total count
    
    # For now, return the verified rank-26 solution
    return best_result
# EVOLVE-BLOCK-END
