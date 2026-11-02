"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def _rank11_solution():
    """Return the known rank-11 decomposition (Strassen + schoolbook)."""
    n, m, p = DIMENSIONS
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen for 2x2x2 (cols 0,1 of B)
    # M1 = (a11 + a22)(b11 + b22)
    U[0][0] += 1; U[3][0] += 1
    V[0][0] += 1; V[4][0] += 1
    W[0][0] += 1; W[3][0] += 1
    
    # M2 = (a21 + a22)b11
    U[2][1] += 1; U[3][1] += 1
    V[0][1] += 1
    W[1][1] += 1; W[3][1] += -1
    
    # M3 = a11(b12 - b22)
    U[0][2] += 1
    V[1][2] += 1; V[4][2] += -1
    W[2][2] += 1; W[3][2] += 1
    
    # M4 = a22(b21 - b11)
    U[3][3] += 1
    V[3][3] += 1; V[0][3] += -1
    W[0][3] += 1; W[1][3] += 1
    
    # M5 = (a11 + a12)b22
    U[0][4] += 1; U[1][4] += 1
    V[4][4] += 1
    W[0][4] += -1; W[2][4] += 1
    
    # M6 = (a21 - a11)(b11 + b12)
    U[2][5] += 1; U[0][5] += -1
    V[0][5] += 1; V[1][5] += 1
    W[3][5] += 1
    
    # M7 = (a12 - a22)(b21 + b22)
    U[1][6] += 1; U[3][6] += -1
    V[3][6] += 1; V[4][6] += 1
    W[0][6] += 1
    
    # Schoolbook for 3rd col (col 2 of B)
    # M8 = a11*b13
    U[0][7] += 1
    V[2][7] += 1
    W[4][7] += 1
    
    # M9 = a12*b23
    U[1][8] += 1
    V[5][8] += 1
    W[4][8] += 1
    
    # M10 = a21*b13
    U[2][9] += 1
    V[2][9] += 1
    W[5][9] += 1
    
    # M11 = a22*b23
    U[3][10] += 1
    V[5][10] += 1
    W[5][10] += 1

    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # The tensor T_{i,j,k} = 1 if (i,j) in A, (j,k) in B, output at (i,k)
    # Decomposition: T = sum_r U[:,r] * V[:,r] * W[:,r]
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    
    # Index conventions:
    # A[i,j] -> index i*m + j in U
    # B[j,k] -> index j*p + k in V
    # C[i,k] -> index k*n + i in W (trace convention W[k*n+i, r])
    
    # Reconstruct tensor
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U_arr[:, r], V_arr[:, r], W_arr[:, r])
    
    # Check against expected tensor
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                if abs(T[a_idx, b_idx, c_idx] - expected) > 1e-10:
                    return False
                # All other entries should be 0
    # Check all other entries are zero
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                i = a // m
                j = a % m
                jj = b // p
                k = b % p
                kk = c // n
                ii = c % n
                if (j != jj) or (i != ii) or (k != kk):
                    if abs(T[a, b, c]) > 1e-10:
                        return False
    return True

def _search_rank10(deadline):
    """Attempt to find a rank-10 decomposition using random initialization and gradient descent."""
    n, m, p = DIMENSIONS
    rank = 10
    
    # Number of free variables: rank*(n*m + m*p + p*n) = 10*(4+6+6) = 160
    # Number of constraints: n*m*p = 12 (non-zero entries) + rest must be 0
    # Total tensor entries: 4*6*6 = 144
    
    # We'll use a simple approach: random initialization, then try to verify
    # If we can find any rank-10 solution, great. Otherwise return None.
    
    # Try random search with rational-like values
    import random
    random.seed(42)
    
    for attempt in range(200):
        if time.time() > deadline - 30:
            break
        
        # Initialize with small random values
        U = [[random.choice([-2, -1, 0, 1, 2]) for _ in range(rank)] for _ in range(n * m)]
        V = [[random.choice([-2, -1, 0, 1, 2]) for _ in range(rank)] for _ in range(m * p)]
        W = [[random.choice([-2, -1, 0, 1, 2]) for _ in range(rank)] for _ in range(p * n)]
        
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Try to find rank-10 solution
    result = _search_rank10(deadline)
    if result is not None:
        return result
    
    # Fallback to rank-11
    return _rank11_solution()
# EVOLVE-BLOCK-END
