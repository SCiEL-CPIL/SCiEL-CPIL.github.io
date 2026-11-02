"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    n, m, p = DIMENSIONS
    
    # The trivial rank is n * m * p
    rank = n * m * p
    
    # Initialize matrices with zeros
    # Using lists to match the expected return type, though numpy is used for verification
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a][column] = 1
                V[b][column] = 1
                W[c][column] = 1
                column += 1
                
    # Verification step to ensure correctness
    # This is fast enough for rank 105
    try:
        U_np = np.array(U, dtype=np.float64)
        V_np = np.array(V, dtype=np.float64)
        W_np = np.array(W, dtype=np.float64)
        
        # Compute T_computed = sum_r U[:, r] * V[:, r] * W[:, r]
        # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        T_computed = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)
        
        # Build target tensor
        T_target = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T_target[a, b, c] = 1.0
        
        # Check if they match
        if not np.allclose(T_computed, T_target, atol=1e-9):
            # If verification fails (should not happen), we still return the trivial solution
            # as it is the best known fallback
            pass
    except Exception:
        # If verification fails due to memory or other issues, return the constructed lists
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
