"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces the matmul tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # Tensor T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # Check: T[i*m+k, k*p+j, j*n+i] should be 1, else 0
    rank = U.shape[1]
    # Build full tensor via einsum or loop
    # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # Use broadcasting
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # T = np.einsum('ar,br,cr->abc', U, V, W)
    T = np.einsum('ar,br,cr->abc', U, V, W)
    # Check all required entries
    for i in range(n):
        for k in range(m):
            for j in range(p):
                a = i * m + k
                b = k * p + j
                c = j * n + i
                if abs(T[a, b, c] - 1.0) > 1e-9:
                    return False
    # Check a sample of zero entries
    import random
    random.seed(42)
    for _ in range(200):
        a = random.randint(0, n*m-1)
        b = random.randint(0, m*p-1)
        c = random.randint(0, p*n-1)
        # Determine if this should be zero
        i_a = a // m
        k_a = a % m
        k_b = b // p
        j_b = b % p
        j_c = c // n
        i_c = c % n
        should_be_one = (i_a == i_c and k_a == k_b and j_b == j_c)
        if should_be_one:
            if abs(T[a, b, c] - 1.0) > 1e-9:
                return False
        else:
            if abs(T[a, b, c]) > 1e-9:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120s limit, leave margin
    
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition: rank = n*m*p
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U[i * m + k][col] = 1
                V[k * p + j][col] = 1
                W[j * n + i][col] = 1
                col += 1
    
    # Verify the schoolbook decomposition
    if not _verify(U, V, W, n, m, p):
        # Should never happen, but return empty as fallback
        return {"U": [], "V": [], "W": []}
    
    # Try to improve: attempt a rank-reduction search
    # Strategy: try to find a decomposition with rank < 144
    # Using a simple approach: try rank 36, 48, 64, etc.
    
    # For now, return the schoolbook which is guaranteed correct
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
