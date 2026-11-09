import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 58  # 7 + 7 + 44
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Helper to add a Strassen block
    def add_strassen(I, J, K):
        nonlocal col
        # I, J, K are start indices, size 2
        # Local indices i', j', k' in {0, 1}
        
        # Term 1
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[(I+0)*m + (J+0)] = 1; u[(I+1)*m + (J+1)] = 1
        v[(J+0)*p + (K+0)] = 1; v[(J+1)*p + (K+1)] = 1
        w[(K+0)*n + (I+0)] = 1; w[(K+1)*n + (I+1)] = 1
        U[:, col] = u; V[:, col] = v; W[:, col] = w; col += 1
        
        # Term 2
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[(I+1)*m + (J+0)] = 1; u[(I+1)*m + (J+1)] = 1
        v[(J+0)*p + (K+0)] = 1
        w[(K+1)*n + (I+0)] = 1; w[(K+1)*n + (I+1)] = -1
        U[:, col] = u; V[:, col] = v; W[:, col] = w; col += 1
        
        # Term 3
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[(I+0)*m + (J+0)] = 1
        v[(J+0)*p + (K+1)] = 1; v[(J+1)*p + (K+1)] = -1
        w[(K+0)*n + (I+1)] = 1; w[(K+1)*n + (I+1)] = 1
        U[:, col] = u; V[:, col] = v; W[:, col] = w; col += 1
        
        # Term 4
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[(I+1)*m + (J+1)] = 1
        v[(J+1)*p + (K+0)] = 1; v[(J+0)*p + (K+0)] = -1
        w[(K+0)*n + (I+0)] = 1; w[(K+1)*n + (I+0)] = 1
        U[:, col] = u; V[:, col] = v; W[:, col] = w; col += 1
        
        # Term 5
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[(I+0)*m + (J+0)] = 1; u[(I+0)*m + (J+1)] = 1
        v[(J+1)*p + (K+1)] = 1
        w[(K+0)*n + (I+0)] = -1; w[(K+0)*n + (I+1)] = 1
        U[:, col] = u; V[:, col] = v; W[:, col] = w; col += 1
        
        # Term 6
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[(I+1)*m + (J+0)] = 1; u[(I+0)*m + (J+0)] = -1
        v[(J+0)*p + (K+0)] = 1; v[(J+0)*p + (K+1)] = 1
        w[(K+1)*n + (I+1)] = 1
        U[:, col] = u; V[:, col] = v; W[:, col] = w; col += 1
        
        # Term 7
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[(I+0)*m + (J+1)] = 1; u[(I+1)*m + (J+1)] = -1
        v[(J+1)*p + (K+0)] = 1; v[(J+1)*p + (K+1)] = 1
        w[(K+0)*n + (I+0)] = 1
        U[:, col] = u; V[:, col] = v; W[:, col] = w; col += 1

    # Add Strassen blocks
    add_strassen(0, 0, 0) # j in {0,1}, k in {0,1}
    add_strassen(0, 2, 2) # j in {2,3}, k in {2,3}
    
    # Add schoolbook for the rest
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip if in Strassen blocks
                if (j in [0,1] and k in [0,1]):
                    continue
                if (j in [2,3] and k in [2,3]):
                    continue
                
                u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
                u[i*m + j] = 1
                v[j*p + k] = 1
                w[k*n + i] = 1
                U[:, col] = u; V[:, col] = v; W[:, col] = w; col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
