import numpy as np
import time

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START

# Candidate rank-24 decomposition from Inspiration 2 (verified at runtime).
# Each term is (U_idx, V_idx, W_idx, coeff).
# U_idx = i*3 + j, V_idx = j*3 + k, W_idx = k*3 + i.
_CANDIDATE_TERMS = [
    (0, 0, 0, 1), (0, 3, 3, 1), (0, 6, 6, -1),
    (3, 1, 4, 1), (3, 4, 7, 1), (3, 7, 0, -1),
    (6, 2, 5, 1), (6, 5, 8, 1), (6, 8, 3, -1),
    (1, 0, 1, -1), (1, 1, 4, 1), (1, 2, 7, 1),
    (4, 0, 2, -1), (4, 1, 5, 1), (4, 2, 8, 1),
    (7, 0, 3, -1), (7, 1, 6, 1), (7, 2, 0, 1),
    (2, 1, 2, -1), (2, 2, 5, 1), (2, 0, 8, -1),
    (5, 1, 3, -1), (5, 2, 6, 1), (5, 0, 0, -1),
]

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _build(terms):
    n, m, p = DIMENSIONS
    rank = len(terms)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    for r, (ui, vi, wi, c) in enumerate(terms):
        U[ui][r] = float(c)
        V[vi][r] = 1.0
        W[wi][r] = 1.0
    return {"U": U, "V": V, "W": W}

def _verify(decomp, tol=1e-9):
    n, m, p = DIMENSIONS
    U, V, W = decomp["U"], decomp["V"], decomp["W"]
    rank = len(U[0])
    
    # Precompute U, V, W as numpy arrays for speed
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    # Check all alpha, beta, gamma
    for alpha in range(n * m):
        for beta in range(m * p):
            for gamma in range(p * n):
                # Calculate target T[alpha, beta, gamma]
                i = alpha // m
                j = alpha % m
                j_prime = beta // p
                k = beta % p
                k_prime = gamma // n
                i_prime = gamma % n
                
                # T is 1 if j == j_prime and k == k_prime and i == i_prime
                # Note: The tensor indices are coupled:
                # alpha = i*m + j
                # beta = j*p + k
                # gamma = k*n + i
                # So valid if:
                # alpha % m == beta // p
                # beta % p == gamma // n
                # gamma % n == alpha // m
                
                target = 1.0 if (j == j_prime and k == k_prime and i == i_prime) else 0.0
                
                # Calculate sum_r U[alpha, r] * V[beta, r] * W[gamma, r]
                val = np.dot(np.dot(U_np[alpha], V_np[beta]), W_np[gamma])
                
                if abs(val - target) > tol:
                    return False
    return True

def construct():
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin before 150s kill
    
    # Try the candidate rank-24 decomposition
    decomp = _build(_CANDIDATE_TERMS)
    
    # Verify it
    if _verify(decomp):
        return decomp
    
    # Fallback to schoolbook rank-27
    return _schoolbook()

# EVOLVE-BLOCK-END
