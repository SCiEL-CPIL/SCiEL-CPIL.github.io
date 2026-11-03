import numpy as np
import time

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Indices
    # I: 0..n*m-1 (10)
    # J: 0..m*p-1 (25)
    # K: 0..p*n-1 (10)
    
    # Target Tensor T (sparse representation)
    # T[I, J, K] = 1 if I=(i,j), J=(j,k), K=(i,k)
    
    # We will search for rank r
    # Start from 49 down to 25
    
    # Precompute the structure of the equations
    # For each K, we have a system A_K W_K = b_K
    # A_K is (n*m)*(m*p) x r ? No.
    # Variables: W[K, 0..r-1]
    # Equations: For each I, J
    # Sum_c U[I, c] V[J, c] W[K, c] = T[I, J, K]
    
    # We can vectorize the solution for all K.
    # But let's just loop K.
    
    def verify(U, V, W):
        # U: (n*m, r)
        # V: (m*p, r)
        # W: (p*n, r)
        # Check non-zero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    I = i * m + j
                    J = j * p + k
                    K = k * n + i
                    # Compute sum
                    s = np.dot(U[I, :], V[J, :] * W[K, :])
                    if abs(s - 1.0) > 1e-9:
                        return False
        # Check some zero entries (random sample)
        # If non-zero are correct, likely zero are correct if rank is low enough?
        # No, we must check all.
        # But checking all 2500 is fast.
        for I in range(n * m):
            for J in range(m * p):
                for K in range(p * n):
                    # Skip non-zero
                    i1 = I // m
                    j1 = I % m
                    j2 = J // p
                    k2 = J % p
                    k1 = K // n
                    i2 = K % n
                    if (i1 == i2 and j1 == j2 and k1 == k2):
                        continue
                    s = np.dot(U[I, :], V[J, :] * W[K, :])
                    if abs(s) > 1e-9:
                        return False
        return True

    def solve_for_W(U, V):
        # U: (10, r), V: (25, r)
        W = np.zeros((p * n, r))
        for K in range(p * n):
            # Build A_K (250 x r) and b_K (250)
            # A_K[I*25 + J, c] = U[I, c] * V[J, c]
            A_K = np.zeros((n * m * m * p, r))
            b_K = np.zeros(n * m * m * p)
            
            idx = 0
            for I in range(n * m):
                for J in range(m * p):
                    A_K[idx, :] = U[I, :] * V[J, :]
                    # b_K is T[I, J, K]
                    # Determine if T[I, J, K] is 1
                    i1 = I // m
                    j1 = I % m
                    j2 = J // p
                    k2 = J % p
                    k1 = K // n
                    i2 = K % n
                    if (i1 == i2 and j1 == j2 and k1 == k2):
                        b_K[idx] = 1.0
                    else:
                        b_K[idx] = 0.0
                    idx += 1
            
            # Solve A_K W_K = b_K
            # Use least squares
            sol, residuals, rank, sv = np.linalg.lstsq(A_K, b_K, rcond=None)
            W[K, :] = sol
        return W

    # Search
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    for r in range(49, 24, -1):
        if time.time() > deadline:
            break
        found = False
        for _ in range(200): # 200 iterations per rank
            if time.time() > deadline:
                break
            # Random U, V
            # Use integer or small rational entries?
            # Random normal is fine for search, but result must be exact.
            # If we find a solution, it will be rational if U, V are rational.
            # We can use integers.
            U = np.random.randint(-2, 3, size=(n * m, r))
            V = np.random.randint(-2, 3, size=(m * p, r))
            
            W = solve_for_W(U, V)
            
            if verify(U, V, W):
                # Convert to lists
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
        
    # If not found, return schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
