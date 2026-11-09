"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """Check if the decomposition is valid: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k"""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _trivial_decomposition(n, m, p):
    """Return the trivial rank n*m*p decomposition"""
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

def _als_optimize(U, V, W, n, m, p, max_iters=100, tol=1e-9):
    """
    Alternating Least Squares to minimize ||T(U,V,W) - 1||_F^2.
    U: (n*m, r), V: (m*p, r), W: (p*n, r)
    T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
    """
    r = U.shape[1]
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    U_reshaped = U.reshape(n, m, r)
    V_reshaped = V.reshape(m, p, r)
    W_reshaped = W.reshape(p, n, r)
    
    prev_loss = float('inf')
    
    for it in range(max_iters):
        # Update U: For each (i,j), U[i,j,:] should minimize
        # sum_k (T[i,j,k] - 1)^2
        # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
        # Let A[i,j,k,c] = V[j,k,c] * W[k,i,c]
        # Then T[i,j,k] = sum_c U[i,j,c] A[i,j,k,c]
        # We want U[i,j,:] to minimize sum_k (sum_c U[i,j,c] A[i,j,k,c] - 1)^2
        # This is a least squares problem: min ||A[i,j,:] U[i,j,:]^T - 1||^2
        # where A[i,j,:] has shape (p, r)
        
        # Efficient computation using einsum
        # A[i,j,k,c] = V[j,k,c] * W[k,i,c]
        # A_reshaped: (n, m, p, r)
        A = np.einsum('jkc,kic->ijkc', V_reshaped, W_reshaped, optimize=True)
        
        # For each (i,j), solve: min ||A[i,j] U[i,j]^T - 1||^2
        # A[i,j] is (p, r), U[i,j] is (r,)
        # This is min ||A[i,j] u - 1||^2 where u is (r,) and 1 is (p,)
        # Normal equations: (A[i,j]^T A[i,j]) u = A[i,j]^T 1
        # A[i,j]^T A[i,j] is (r, r), A[i,j]^T 1 is (r,)
        
        # Compute Gram matrix G[i,j] = A[i,j]^T A[i,j] (r, r)
        # G[i,j,a,b] = sum_k A[i,j,k,a] A[i,j,k,b]
        G = np.einsum('ijka,ijkb->ijab', A, A, optimize=True)
        
        # Compute rhs[i,j] = A[i,j]^T 1 (r,)
        # rhs[i,j,a] = sum_k A[i,j,k,a]
        rhs = np.einsum('ijka->ija', A, optimize=True)
        
        # Solve G[i,j] u[i,j] = rhs[i,j]
        # Use np.linalg.lstsq for stability
        G_flat = G.reshape(n*m, r, r)
        rhs_flat = rhs.reshape(n*m, r)
        U_flat = U_reshaped.reshape(n*m, r)
        
        try:
            U_new_flat, _, _, _ = np.linalg.lstsq(G_flat, rhs_flat, rcond=None)
        except np.linalg.LinAlgError:
            break
        
        U_new = U_new_flat.reshape(n, m, r)
        
        # Update V: For each (j,k), V[j,k,:] should minimize
        # sum_i (T[i,j,k] - 1)^2
        # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
        # Let B[i,j,k,c] = U[i,j,c] * W[k,i,c]
        # B_reshaped: (n, m, p, r)
        B = np.einsum('ijc,kic->ijkc', U_new, W_reshaped, optimize=True)
        
        G_v = np.einsum('ijka,ijkb->jkab', B, B, optimize=True)
        rhs_v = np.einsum('ijka->jka', B, optimize=True)
        
        G_v_flat = G_v.reshape(m*p, r, r)
        rhs_v_flat = rhs_v.reshape(m*p, r)
        
        try:
            V_new_flat, _, _, _ = np.linalg.lstsq(G_v_flat, rhs_v_flat, rcond=None)
        except np.linalg.LinAlgError:
            break
        
        V_new = V_new_flat.reshape(m, p, r)
        
        # Update W: For each (k,i), W[k,i,:] should minimize
        # sum_j (T[i,j,k] - 1)^2
        # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
        # Let C[i,j,k,c] = U[i,j,c] * V[j,k,c]
        # C_reshaped: (n, m, p, r)
        C = np.einsum('ijc,jkc->ijkc', U_new, V_new, optimize=True)
        
        G_w = np.einsum('ijka,ijkb->khab', C, C, optimize=True)
        rhs_w = np.einsum('ijka->kha', C, optimize=True)
        
        G_w_flat = G_w.reshape(p*n, r, r)
        rhs_w_flat = rhs_w.reshape(p*n, r)
        
        try:
            W_new_flat, _, _, _ = np.linalg.lstsq(G_w_flat, rhs_w_flat, rcond=None)
        except np.linalg.LinAlgError:
            break
        
        W_new = W_new_flat.reshape(p, n, r)
        
        # Update
        U_reshaped = U_new
        V_reshaped = V_new
        W_reshaped = W_new
        
        # Compute loss
        T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped, optimize=True)
        loss = np.sum((T - 1.0)**2)
        
        if abs(prev_loss - loss) < tol:
            break
        prev_loss = loss
    
    U = U_reshaped.reshape(n*m, r)
    V = V_reshaped.reshape(m*p, r)
    W = W_reshaped.reshape(p*n, r)
    
    return U, V, W

def _try_rank_r(n, m, p, r, seed, max_restarts=10, max_als_iters=50):
    """Try to find a rank-r decomposition using random initialization + ALS + projection"""
    rng = np.random.RandomState(seed)
    
    for restart in range(max_restarts):
        # Random initialization with small values
        U = rng.randn(n*m, r)
        V = rng.randn(m*p, r)
        W = rng.randn(p*n, r)
        
        # Run ALS
        U_opt, V_opt, W_opt = _als_optimize(U, V, W, n, m, p, max_iters=max_als_iters)
        
        # Project to nearest half-integer
        U_proj = np.round(U_opt * 2) / 2
        V_proj = np.round(V_opt * 2) / 2
        W_proj = np.round(W_opt * 2) / 2
        
        # Verify
        if _verify_decomposition(U_proj, V_proj, W_proj, n, m, p, tol=1e-4):
            U_list = U_proj.tolist()
            V_list = V_proj.tolist()
            W_list = W_proj.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
        
        # Also try integer projection
        U_int = np.round(U_opt)
        V_int = np.round(V_opt)
        W_int = np.round(W_opt)
        if _verify_decomposition(U_int, V_int, W_int, n, m, p, tol=1e-4):
            U_list = U_int.tolist()
            V_list = V_int.tolist()
            W_list = W_int.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Start with trivial decomposition
    best = _trivial_decomposition(n, m, p)
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Lower bound: max(n*m, m*p, p*n) = max(15, 35, 21) = 35
    lower_bound = max(n*m, m*p, p*n)
    
    seed = 42
    for r in range(lower_bound, best_rank + 1):
        if time.time() > deadline - 20:
            break
        result = _try_rank_r(n, m, p, r, seed, max_restarts=5, max_als_iters=30)
        if result is not None:
            best = result
            best_rank = r
            break
        seed += 1
    
    return best
# EVOLVE-BLOCK-END
