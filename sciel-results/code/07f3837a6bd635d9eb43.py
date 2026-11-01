"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
import time
import numpy as np

DEADLINE_SECONDS = 110  # Safety margin within 120s evaluation limit

def _schoolbook_decomposition(n, m, p):
    """Return (U, V, W) for the trivial schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Check that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    # The tensor is: T[i*m+j, j'*p+k, k'*n+i'] = 1 if i=i', j=j', k=k' else 0
    # We verify by checking all combinations
    for i in range(n):
        for j in range(m):
            for j_prime in range(m):
                for k in range(p):
                    for k_prime in range(p):
                        for i_prime in range(n):
                            a = i * m + j
                            b = j_prime * p + k
                            c = k_prime * n + i_prime
                            expected = 1.0 if (i == i_prime and j == j_prime and k == k_prime) else 0.0
                            actual = 0.0
                            for r in range(rank):
                                actual += U[a, r] * V[b, r] * W[c, r]
                            if abs(actual - expected) > tol:
                                return False
    return True

def _try_find_decomposition(n, m, p, max_rank, time_limit):
    """Try to find a decomposition of rank <= max_rank using random search."""
    start_time = time.time()
    # This is a placeholder; implementing a full search is complex.
    # For now, just return None to indicate no better decomposition found.
    # In a real implementation, this would use random linear algebra or structured search.
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U, V, W = _schoolbook_decomposition(n, m, p)
    rank = U.shape[1]
    
    # Convert to lists as required by the interface
    U_list = [[float(U[i, j]) for j in range(rank)] for i in range(len(U))]
    V_list = [[float(V[i, j]) for j in range(rank)] for i in range(len(V))]
    W_list = [[float(W[i, j]) for j in range(rank)] for i in range(len(W))]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
