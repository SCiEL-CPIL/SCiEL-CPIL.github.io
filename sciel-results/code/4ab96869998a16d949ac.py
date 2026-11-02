# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify if the given factor matrices decompose the matrix multiplication tensor.
    U: (n*m, R)
    V: (m*p, R)
    W: (p*n, R)
    Convention: W[k*n + i, r] corresponds to output C[i, k]
    """
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # D[x, y, z] = sum_r U[x, r] * V[y, r] * W[z, r]
    # x = i*m + j
    # y = j*p + k
    # z = k*n + i
    D = np.einsum('xr,yr,zr->xyz', U_np, V_np, W_np)
    
    expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                x = i * m + j
                y = j * p + k
                z = k * n + i
                expected[x, y, z] = 1.0
                
    return np.allclose(D, expected, atol=tol)

def _generate_schoolbook(n, m, p):
    """Generate the standard schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _snap_to_grid(matrix, step=0.5):
    """Snap matrix entries to the nearest multiple of step."""
    return np.round(matrix / step) * step

def _search_decomposition(n, m, p, target_rank, time_limit=10.0):
    """
    Search for a decomposition of the given rank using random restarts and gradient descent.
    Returns (U, V, W) if found, else None.
    """
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Target tensor
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                x = i * m + j
                y = j * p + k
                z = k * n + i
                T[x, y, z] = 1.0
                
    start_time = time.time()
    
    # Try a few random initializations
    for restart in range(100):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize with small random values
        U = np.random.randn(dim_u, target_rank) * 0.5
        V = np.random.randn(dim_v, target_rank) * 0.5
        W = np.random.randn(dim_w, target_rank) * 0.5
        
        # Gradient descent
        lr = 0.05
        for step in range(500):
            if time.time() - start_time > time_limit:
                break
                
            # Compute D
            D = np.einsum('xr,yr,zr->xyz', U, V, W)
            diff = D - T
            loss = np.sum(diff**2)
            
            # Gradients
            grad_U = np.einsum('xyz,yr,zr->xr', diff, V, W)
            grad_V = np.einsum('xyz,xr,zr->yr', diff, U, W)
            grad_W = np.einsum('xyz,xr,yr->zr', diff, U, V)
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Check convergence
            if loss < 1e-4:
                # Snap to grid
                U_snap = _snap_to_grid(U)
                V_snap = _snap_to_grid(V)
                W_snap = _snap_to_grid(W)
                
                if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p, tol=1e-9):
                    return U_snap, V_snap, W_snap
                else:
                    # Try integer snap
                    U_int = np.round(U)
                    V_int = np.round(V)
                    W_int = np.round(W)
                    if _verify_decomposition(U_int, V_int, W_int, n, m, p, tol=1e-9):
                        return U_int, V_int, W_int
                    # If not, break and try next restart
                    break

    return None

def construct():
    """
    Constructs the minimal rank tensor decomposition for 3x4 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    
    # Try to find a lower rank decomposition
    # Start from the theoretical lower bound or a guess
    # For 3x4x5, lower bound is 15. Let's try 20 as suggested by inspiration.
    # We will try ranks 20 to 30.
    for rank in range(20, 31):
        res = _search_decomposition(n, m, p, rank, time_limit=10.0)
        if res is not None:
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Fallback to schoolbook
    return _generate_schoolbook(n, m, p)

# EVOLVE-BLOCK-END
