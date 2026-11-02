<<<<<<< SEARCH
def _verify(U, V, W, n, m, p, test_size=5, seed=42):
    """Verify the decomposition with random integer matrices."""
    np.random.seed(seed)
    # Generate random integer matrices
    A = np.random.randint(-5, 5, size=(n, m))
    B = np.random.randint(-5, 5, size=(m, p))
    C_expected = A @ B
    
    # Compute via decomposition
    C_computed = np.zeros((n, p), dtype=object)
    U_arr = np.array(U, dtype=object)
    V_arr = np.array(V, dtype=object)
    W_arr = np.array(W, dtype=object)
    A_arr = A.astype(object)
    B_arr = B.astype(object)
    
    rank = len(U[0]) if U else 0
    for r in range(rank):
        # left_r = sum_j U[i*m+j, r] * A[i,j] for each i
        left = np.zeros(n, dtype=object)
        for i in range(n):
            s = 0
            for j in range(m):
                s += U_arr[i*m+j, r] * A_arr[i, j]
            left[i] = s
        
        # right_r = sum_l V[l*p+k, r] * B[l,k] for each k
        right = np.zeros(p, dtype=object)
        for k in range(p):
            s = 0
            for l in range(m):
                s += V_arr[l*p+k, r] * B_arr[l, k]
            right[k] = s
        
        # Add contribution: W[k*n+i, r] * left[i] * right[k]
        for i in range(n):
            for k in range(p):
                C_computed[i, k] += W_arr[k*n+i, r] * left[i] * right[k]
    
    for i in range(n):
        for k in range(p):
            if C_computed[i, k] != C_expected[i, k]:
                return False, i, k, C_computed[i, k], C_expected[i, k]
    return True, None, None, None, None
=======
def _verify(U, V, W, n, m, p, test_size=5, seed=42):
    """Verify the decomposition with random integer matrices."""
    np.random.seed(seed)
    A = np.random.randint(-5, 5, size=(n, m))
    B = np.random.randint(-5, 5, size=(m, p))
    C_expected = A @ B
    
    C_computed = np.zeros((n, p), dtype=object)
    U_arr = np.array(U, dtype=object)
    V_arr = np.array(V, dtype=object)
    W_arr = np.array(W, dtype=object)
    A_arr = A.astype(object)
    B_arr = B.astype(object)
    
    rank = len(U[0]) if U else 0
    for r in range(rank):
        left = np.zeros(n, dtype=object)
        for i in range(n):
            s = 0
            for j in range(m):
                s += U_arr[i*m+j, r] * A_arr[i, j]
            left[i] = s
        
        right = np.zeros(p, dtype=object)
        for k in range(p):
            s = 0
            for l in range(m):
                s += V_arr[l*p+k, r] * B_arr[l, k]
            right[k] = s
        
        for i in range(n):
            for k in range(p):
                C_computed[i, k] += W_arr[k*n+i, r] * left[i] * right[k]
    
    for i in range(n):
        for k in range(p):
            if C_computed[i, k] != C_expected[i, k]:
                return False, i, k, C_computed[i, k], C_expected[i, k]
    return True, None, None, None, None
>>>>>>> REPLACE
