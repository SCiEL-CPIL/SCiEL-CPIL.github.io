"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_tensor(U, V, W, n, m, p, rank):
    """Compute the reconstructed tensor from factor matrices."""
    T = np.zeros((n, m, p))
    for c in range(rank):
        u = U[:, c].reshape(n, m)
        v = V[:, c].reshape(m, p)
        w = W[:, c].reshape(p, n)
        T += np.einsum('ij,jk,ki->ijk', u, v, w)
    return T

def _compute_error(U, V, W, n, m, p, rank):
    """Compute Frobenius norm of error."""
    T_recon = _compute_tensor(U, V, W, n, m, p, rank)
    T_target = np.ones((n, m, p))
    return np.linalg.norm(T_recon - T_target)

def _validate(U, V, W, n, m, p, tol=1e-6):
    """Check if U, V, W form an exact decomposition."""
    rank = U.shape[1]
    T_recon = _compute_tensor(U, V, W, n, m, p, rank)
    T_target = np.ones((n, m, p))
    return np.max(np.abs(T_recon - T_target)) < tol

def _snap_to_half_integers(arr, tol=1e-3):
    """Snap values to nearest integer or half-integer."""
    snapped = np.round(arr * 2) / 2
    if np.max(np.abs(arr - snapped)) > tol:
        return None
    return snapped

def _alternating_optimization(U, V, W, n, m, p, rank, max_iters=200, tol=1e-8):
    """Alternating least squares style optimization."""
    U_cur = U.copy()
    V_cur = V.copy()
    W_cur = W.copy()
    T_target = np.ones((n, m, p))
    
    for it in range(max_iters):
        # Update U: fix V and W, solve for each column of U
        for c in range(rank):
            # T_target = sum_c' U_c' outer V_c' outer W_c'
            # For fixed c, we want to update U[:, c]
            # T_target[i,j,k] = U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] + sum_{c'!=c} ...
            # This is complex; use gradient step instead
            pass
        
        # Use gradient-based updates
        # Compute gradient for U
        dU = np.zeros_like(U_cur)
        dV = np.zeros_like(V_cur)
        dW = np.zeros_like(W_cur)
        
        for c in range(rank):
            u = U_cur[:, c].reshape(n, m)
            v = V_cur[:, c].reshape(m, p)
            w = W_cur[:, c].reshape(p, n)
            
            # Gradient of ||T - T_target||^2 w.r.t. U[:, c]
            # dT/dU[i,j,c] = V[j,k,c] * W[k,i,c] summed over k
            # = (v @ w.T) but with appropriate reshaping
            # v is (m,p), w is (p,n)
            # For each (i,j): sum_k v[j,k]*w[k,i] = (v @ w)[j,i] but v@(w) gives (m,n)
            grad_u = np.einsum('jk,ki->ij', v, w)  # (n, m)
            # Error at this c: T_recon - T_target
            T_recon = _compute_tensor(U_cur, V_cur, W_cur, n, m, p, rank)
            err = T_recon - T_target  # (n,m,p)
            # Gradient: for U[i,j,c], it's sum over k of err[i,j,k] * v[j,k,c] * w[k,i,c]
            # But we already have grad_u[i,j] = sum_k v[j,k]*w[k,i]
            # Actually the gradient w.r.t U[i,j,c] is sum_k err[i,j,k] * v[j,k,c] * w[k,i,c]
            # = einsum over k: err[i,j,k] * v[j,k] * w[k,i]
            # This is not simply grad_u * err because err depends on all c
            # Let's just compute it directly
            grad_u_direct = np.zeros((n, m))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        grad_u_direct[i, j] += err[i, j, k] * v[j, k] * w[k, i]
            dU[:, c] = grad_u_direct.reshape(-1)
            
            # Gradient for V
            grad_v_direct = np.zeros((m, p))
            for j in range(m):
                for k in range(p):
                    for i in range(n):
                        grad_v_direct[j, k] += err[i, j, k] * u[i, j] * w[k, i]
            dV[:, c] = grad_v_direct.reshape(-1)
            
            # Gradient for W
            grad_w_direct = np.zeros((p, n))
            for k in range(p):
                for i in range(n):
                    for j in range(m):
                        grad_w_direct[k, i] += err[i, j, k] * u[i, j] * v[j, k]
            dW[:, c] = grad_w_direct.reshape(-1)
        
        # Gradient descent step
        lr = 0.01
        U_cur -= lr * dU
        V_cur -= lr * dV
        W_cur -= lr * dW
        
        # Check convergence
        if it % 10 == 0:
            err_now = _compute_error(U_cur, V_cur, W_cur, n, m, p, rank)
            if err_now < tol:
                break
    
    return U_cur, V_cur, W_cur, _compute_error(U_cur, V_cur, W_cur, n, m, p, rank)

def _try_rank(n, m, p, target_rank, num_trials=30, seed=42, time_limit=50):
    """Try to find a decomposition of given rank."""
    start_time = time.time()
    
    for trial in range(num_trials):
        if time.time() - start_time > time_limit:
            break
        
        rng = np.random.RandomState(seed + trial)
        
        # Initialize with small random values
        U_rand = rng.randn(n * m, target_rank) * 0.5
        V_rand = rng.randn(m * p, target_rank) * 0.5
        W_rand = rng.randn(p * n, target_rank) * 0.5
        
        # Run alternating optimization
        U_cur, V_cur, W_cur, error = _alternating_optimization(
            U_rand, V_rand, W_rand, n, m, p, target_rank, max_iters=100
        )
        
        if error < 1e-6:
            # Try to snap to half-integers
            U_snapped = _snap_to_half_integers(U_cur, tol=1e-2)
            V_snapped = _snap_to_half_integers(V_cur, tol=1e-2)
            W_snapped = _snap_to_half_integers(W_cur, tol=1e-2)
            
            if U_snapped is not None and V_snapped is not None and W_snapped is not None:
                if _validate(U_snapped, V_snapped, W_snapped, n, m, p, target_rank):
                    return U_snapped, V_snapped, W_snapped
        
        # Also try without snapping
        if error < 1e-6:
            if _validate(U_cur, V_cur, W_cur, n, m, p, target_rank):
                return U_cur, V_cur, W_cur
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Schoolbook baseline
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_result = {
        "U": [[U[i, c] for c in range(rank)] for i in range(n * m)],
        "V": [[V[i, c] for c in range(rank)] for i in range(m * p)],
        "W": [[W[i, c] for c in range(rank)] for i in range(p * n)]
    }
    best_rank = rank
    
    # Try lower ranks
    # Lower bound is max(n*m, m*p, p*n) = 12
    for target_rank in range(12, best_rank):
        if time.time() > deadline - 10:
            break
        
        result = _try_rank(n, m, p, target_rank, num_trials=20, seed=42, time_limit=15)
        
        if result is not None:
            U_found, V_found, W_found = result
            best_rank = target_rank
            best_result = {
                "U": [[U_found[i, c] for c in range(target_rank)] for i in range(n * m)],
                "V": [[V_found[i, c] for c in range(target_rank)] for i in range(m * p)],
                "W": [[W_found[i, c] for c in range(target_rank)] for i in range(p * n)]
            }
            break
    
    return best_result
# EVOLVE-BLOCK-END
