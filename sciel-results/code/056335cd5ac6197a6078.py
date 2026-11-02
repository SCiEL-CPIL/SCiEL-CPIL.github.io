import numpy as np

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build the target tensor T
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    # Try to find minimal rank decomposition
    # Use SVD-guided search: for each candidate rank, try to find integer/half-integer U, V, W
    
    def check_decomposition(U, V, W, T):
        """Check if U, V, W exactly decompose T."""
        nm, R = U.shape
        mp, _ = V.shape
        pn, _ = W.shape
        
        # Compute T_hat[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # Efficient: for each r, add outer product
        T_hat = np.zeros((nm, mp, pn), dtype=float)
        for r in range(R):
            T_hat += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        
        return np.allclose(T_hat, T, atol=1e-10)
    
    def try_rank(R, max_attempts=50):
        """Try to find a rank-R decomposition."""
        # Strategy: use the structure of the tensor
        # Each of the 24 terms (i,j,k) contributes to T[i*m+j, j*p+k, k*n+i]
        # We need to cover all 24 positions with R rank-1 tensors
        
        # For small R, we need each rank-1 tensor to cover multiple terms
        # A rank-1 tensor U[:,r] * V[:,r] * W[:,r] covers terms where:
        #   U[i*m+j, r] != 0, V[j*p+k, r] != 0, W[k*n+i, r] != 0
        #   and the product equals 1 (or -1, etc.)
        
        # Try random search with integer entries
        for attempt in range(max_attempts):
            # Initialize with small integers
            scale = 2
            U = np.random.randint(-scale, scale + 1, size=(nm, R)).astype(float)
            V = np.random.randint(-scale, scale + 1, size=(mp, R)).astype(float)
            W = np.random.randint(-scale, scale + 1, size=(pn, R)).astype(float)
            
            if check_decomposition(U, V, W, T):
                return U, V, W
        
        # Try a structured approach: start from schoolbook and merge
        # Schoolbook has rank 24, one per (i,j,k)
        # Try to find linear dependencies
        
        return None, None, None
    
    # Try ranks from low to high
    for R in range(16, 25):
        U, V, W = try_rank(R, max_attempts=100)
        if U is not None:
            # Convert to lists of lists with integer/half-integer values
            U_list = [[float(U[i,j]) for j in range(R)] for i in range(nm)]
            V_list = [[float(V[i,j]) for j in range(R)] for i in range(mp)]
            W_list = [[float(W[i,j]) for j in range(R)] for i in range(pn)]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook (rank 24)
    rank = nm * mp * pn
    U = [[0.0] * rank for _ in range(nm)]
    V = [[0.0] * rank for _ in range(mp)]
    W = [[0.0] * rank for _ in range(pn)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1.0
                V[j * p + k][column] = 1.0
                W[k * n + i][column] = 1.0
                column += 1
    return {"U": U, "V": V, "W": W}
