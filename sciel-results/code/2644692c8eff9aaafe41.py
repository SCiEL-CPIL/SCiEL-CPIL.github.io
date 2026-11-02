# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Tensor T[i,j,k] = 1
    # We check sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    # Reshape to 3D arrays for broadcasting
    # U shape: (n*m, rank) -> (n, m, rank)
    U_3d = U_np.reshape(n, m, rank)
    # V shape: (m*p, rank) -> (m, p, rank)
    V_3d = V_np.reshape(m, p, rank)
    # W shape: (p*n, rank) -> (p, n, rank)
    W_3d = W_np.reshape(p, n, rank)
    
    # We want S[i, j, k] = sum_c U_3d[i, j, c] * V_3d[j, k, c] * W_3d[k, i, c]
    # Note: W index is k*n+i. So W_3d[k, i, c].
    
    # Use einsum for efficiency
    # S[i, j, k] = sum_c U[i, j, c] V[j, k, c] W[k, i, c]
    # U: i j c
    # V: j k c
    # W: k i c
    # Result: i j k
    S = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
    
    return np.allclose(S, 1.0, atol=1e-9)

def _is_half_integer(x):
    """Check if x is an integer or half-integer."""
    if isinstance(x, Fraction):
        return x.denominator in (1, 2)
    fx = float(x)
    return abs(fx * 2 - round(fx * 2)) < 1e-9

def _snap_to_half_integer(x):
    """Snap a value to the nearest half-integer."""
    return Fraction(round(float(x) * 2), 2)

def _try_rank(rank, n, m, p, time_limit=10):
    """Try to find a rank-`rank` decomposition using random initialization and refinement."""
    start_time = time.time()
    
    # Initialize with random half-integer values
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for i in range(n*m):
        for c in range(rank):
            U[i][c] = random.choice([-1, 0, 1])
    for j in range(m*p):
        for c in range(rank):
            V[j][c] = random.choice([-1, 0, 1])
    for k in range(p*n):
        for c in range(rank):
            W[k][c] = random.choice([-1, 0, 1])
    
    if _verify_decomposition(U, V, W, n, m, p):
        return U, V, W
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    def loss_and_grad(U, V, W):
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        
        S = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        err = S - 1.0
        loss = np.sum(err**2)
        
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        # Gradients via einsum
        # dL/dU[i,j,c] = 2 * sum_k err[i,j,k] * V[j,k,c] * W[k,i,c]
        # V_3d[j,k,c], W_3d[k,i,c]
        # grad_U_3d[i,j,c] = 2 * sum_k err[i,j,k] V_3d[j,k,c] W_3d[k,i,c]
        grad_U_3d = 2 * np.einsum('ijk,jkc,kic->ijc', err, V_3d, W_3d)
        grad_U = grad_U_3d.reshape(n*m, rank)
        
        # dL/dV[j,k,c] = 2 * sum_i err[i,j,k] U[i,j,c] W[k,i,c]
        grad_V_3d = 2 * np.einsum('ijk,ijc,kic->jkc', err, U_3d, W_3d)
        grad_V = grad_V_3d.reshape(m*p, rank)
        
        # dL/dW[k,i,c] = 2 * sum_j err[i,j,k] U[i,j,c] V[j,k,c]
        grad_W_3d = 2 * np.einsum('ijk,ijc,jkc->kic', err, U_3d, V_3d)
        grad_W = grad_W_3d.reshape(p*n, rank)
        
        return loss, grad_U, grad_V, grad_W
    
    lr = 0.01
    for step in range(500):
        if time.time() - start_time > time_limit:
            break
        
        loss, gU, gV, gW = loss_and_grad(U_np, V_np, W_np)
        
        if loss < 1e-6:
            U_snap = [[_snap_to_half_integer(U_np[i,c]) for c in range(rank)] for i in range(n*m)]
            V_snap = [[_snap_to_half_integer(V_np[j,c]) for c in range(rank)] for j in range(m*p)]
            W_snap = [[_snap_to_half_integer(W_np[k,c]) for c in range(rank)] for k in range(p*n)]
            if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p):
                return U_snap, V_snap, W_snap
            break
        
        U_np -= lr * gU
        V_np -= lr * gV
        W_np -= lr * gW
    
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    U = [[0] * schoolbook_rank for _ in range(n * m)]
    V = [[0] * schoolbook_rank for _ in range(m * p)]
    W = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best_U, best_V, best_W = U, V, W
    best_rank = schoolbook_rank
    
    deadline = time.time() + 120
    for rank in range(schoolbook_rank - 1, max(10, schoolbook_rank - 20), -1):
        if time.time() > deadline - 5:
            break
        
        result = _try_rank(rank, n, m, p, time_limit=min(10, deadline - time.time() - 2))
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W = U, V, W
            best_rank = rank
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
