"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def _is_half_integer(x, tol=1e-9):
    """Check if x is a half-integer (k/2 for integer k)."""
    x = np.asarray(x)
    return np.allclose(x * 2, np.round(x * 2), atol=tol)

def _snap_to_half_integer(x):
    """Snap values to nearest half-integer."""
    return np.round(np.asarray(x) * 2) / 2.0

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor.
    
    The tensor T[i,j,k] = 1 for all i,j,k.
    Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j], V[j*p+k]) * 0 + np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                # Actually: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _compute_residual(U, V, W, n, m, p):
    """Compute the residual tensor R[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1."""
    rank = U.shape[1]
    R = np.zeros((n, m, p))
    for r in range(rank):
        Ur = U[:, r].reshape(n, m)
        Vr = V[:, r].reshape(m, p)
        Wr = W[:, r].reshape(p, n)
        # R[i,j,k] += Ur[i,j] * Vr[j,k] * Wr[k,i]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    R[i,j,k] += Ur[i,j] * Vr[j,k] * Wr[k,i]
    R -= 1.0
    return R

def _compute_gradients(U, V, W, R, n, m, p):
    """Compute gradients of 0.5*||R||^2 with respect to U, V, W."""
    rank = U.shape[1]
    grad_U = np.zeros_like(U)
    grad_V = np.zeros_like(V)
    grad_W = np.zeros_like(W)
    
    for r in range(rank):
        Ur = U[:, r].reshape(n, m)
        Vr = V[:, r].reshape(m, p)
        Wr = W[:, r].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = R[i,j,k]
                    grad_U[i*m+j, r] += Vr[j,k] * Wr[k,i] * val
                    grad_V[j*p+k, r] += Ur[i,j] * Wr[k,i] * val
                    grad_W[k*n+i, r] += Ur[i,j] * Vr[j,k] * val
    
    return grad_U, grad_V, grad_W

def _try_gradient_descent(rank, n, m, p, max_iters=50, lr=0.01, seed=42):
    """Try to find a rank-rank decomposition using gradient descent."""
    rng = np.random.RandomState(seed)
    
    U = rng.randn(n * m, rank) * 0.3
    V = rng.randn(m * p, rank) * 0.3
    W = rng.randn(p * n, rank) * 0.3
    
    best_residual = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        R = _compute_residual(U, V, W, n, m, p)
        res_norm = np.linalg.norm(R)
        
        if res_norm < best_residual:
            best_residual = res_norm
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if res_norm < 1e-7:
            break
        
        grad_U, grad_V, grad_W = _compute_gradients(U, V, W, R, n, m, p)
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
    
    # Check if we found a valid solution
    if _verify(best_U, best_V, best_W, n, m, p):
        return best_U, best_V, best_W
    
    # Try snapping to half-integers
    U_snap = _snap_to_half_integer(best_U)
    V_snap = _snap_to_half_integer(best_V)
    W_snap = _snap_to_half_integer(best_W)
    if _verify(U_snap, V_snap, W_snap, n, m, p):
        return U_snap, V_snap, W_snap
    
    return None

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Start with the schoolbook decomposition as a fallback
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook
    
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin
    
    # Try to find lower-rank decompositions using gradient descent
    # Try various ranks from schoolbook-1 down
    for rank in range(rank_schoolbook - 1, 15, -1):
        if time.time() > deadline:
            break
        
        # Try a few random initializations
        found = False
        for seed in range(3):
            if time.time() > deadline:
                break
            
            result = _try_gradient_descent(rank, n, m, p, max_iters=30, lr=0.02, seed=seed)
            if result is not None:
                U, V, W = result
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = rank
                found = True
                break
        
        # If we found a decomposition at this rank, continue to try lower ranks
        # If not, continue to next rank
    
    return best_result
# EVOLVE-BLOCK-END
