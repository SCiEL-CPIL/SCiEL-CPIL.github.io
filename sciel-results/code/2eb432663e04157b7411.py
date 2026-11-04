"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor as a 3D array."""
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _evaluate_decomp(U, V, W, T):
    """Compute the tensor from U,V,W and compare to T."""
    # U: (n*m, r), V: (m*p, r), W: (p*n, r)
    r = U.shape[1]
    # Reconstruct: T_recon[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # Use einsum for efficiency
    T_recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    return np.max(np.abs(T_recon - T))

def _try_rank(n, m, p, rank, T, time_limit=30.0):
    """Try to find a rank-r decomposition using randomized optimization."""
    start_time = time.time()
    best_error = float('inf')
    best_U, best_V, best_W = None, None, None
    
    nm, mp, pn = n*m, m*p, p*n
    
    # Initialize with random values
    for attempt in range(20):
        if time.time() - start_time > time_limit:
            break
        
        U = np.random.randn(nm, rank) * 0.5
        V = np.random.randn(mp, rank) * 0.5
        W = np.random.randn(pn, rank) * 0.5
        
        # Optimize using gradient descent on the error
        lr = 0.01
        for step in range(500):
            if time.time() - start_time > time_limit:
                break
            
            # Compute error
            T_recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            error = np.max(np.abs(T_recon - T))
            
            if error < best_error:
                best_error = error
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                if error < 1e-10:
                    return U, V, W, error
            
            # Compute gradients
            # dT_recon/dU[a,r] = sum_{b,c} (T_recon[a,b,c] - T[a,b,c]) * V[b,r] * W[c,r]
            # But we want to minimize max error, which is non-smooth. Use sum of squares instead.
            
            # Switch to sum of squares for smooth optimization
            diff = T_recon - T
            # Gradient w.r.t. U: grad_U[a,r] = sum_{b,c} diff[a,b,c] * V[b,r] * W[c,r]
            grad_U = np.einsum('abc,br,cr->ar', diff, V, W, optimize=True)
            grad_V = np.einsum('abc,ar,cr->br', diff, U, W, optimize=True)
            grad_W = np.einsum('abc,ar,br->cr', diff, U, V, optimize=True)
            
            # Adam-like update with simple momentum
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
        
        # Try to reduce magnitude
        U *= 1.0
        V *= 1.0
        W *= 1.0
    
    if best_U is not None:
        return best_U, best_V, best_W, best_error
    return None, None, None, float('inf')

def construct():
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    
    # Start from schoolbook rank and try to reduce
    schoolbook_rank = n * m * p
    
    # Try various ranks below schoolbook
    for rank in range(schoolbook_rank - 1, max(schoolbook_rank - 15, 80), -1):
        U, V, W, error = _try_rank(n, m, p, rank, T, time_limit=8.0)
        if U is not None and error < 1e-8:
            # Convert to lists with rounded values
            U_list = [[round(float(x), 10) for x in row] for row in U]
            V_list = [[round(float(x), 10) for x in row] for row in V]
            W_list = [[round(float(x), 10) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fall back to schoolbook
    rank = schoolbook_rank
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
# EVOLVE-BLOCK-END
