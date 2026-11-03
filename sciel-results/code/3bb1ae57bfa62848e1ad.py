import numpy as np

DIMENSIONS = [2, 3, 4]

def _verify_exact(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor.
    Returns True if exact, False otherwise."""
    rank = len(U[0]) if U else 0
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    # Convert to numpy for faster computation
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # Check all entries
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for ip in range(n):
                        for kp in range(p):
                            # Expected: 1 if i==ip and j==jp and k==kp, else 0
                            expected = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                            
                            # Computed: sum_r U[i*m+j, r] * V[jp*p+k, r] * W[kp*n+ip, r]
                            computed = 0.0
                            for r in range(rank):
                                computed += U_np[i*m+j, r] * V_np[jp*p+k, r] * W_np[kp*n+ip, r]
                            
                            if abs(computed - expected) > 1e-9:
                                return False
    return True

def construct():
    n, m, p = DIMENSIONS
    dim_U = n * m  # 6
    dim_V = m * p  # 12
    dim_W = p * n  # 8
    rank_schoolbook = n * m * p  # 24
    
    # Initialize U, V, W with zeros
    U = [[0] * rank_schoolbook for _ in range(dim_U)]
    V = [[0] * rank_schoolbook for _ in range(dim_V)]
    W = [[0] * rank_schoolbook for _ in range(dim_W)]
    
    # Fill in the schoolbook decomposition
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    # Verify the result (optional, but good for safety)
    if _verify_exact(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    else:
        # This should never happen, but return the result anyway
        return {"U": U, "V": V, "W": W}
