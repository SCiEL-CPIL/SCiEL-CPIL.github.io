"""Matrix multiplication 2x3x3 rank-minimal exact tensor decomposition search."""
import time
import numpy as np
from fractions import Fraction
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 3]
_DEADLINE = 145.0
_START_TIME = time.time()

def _elapsed():
    return time.time() - _START_TIME

def _remaining():
    return _DEADLINE - _elapsed()

def construct():
    n, m, p = DIMENSIONS
    # Tensor T has shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    T_flat = T.reshape(-1)
    T_size = T_flat.size
    
    def _tensor_residual(U, V, W, T_flat):
        """Compute residual ||sum_r U[:,r] V[:,r] W[:,r] - T||_F"""
        T_approx = np.einsum('ar,br,cr->abc', U, V, W).reshape(-1)
        return np.linalg.norm(T_approx - T_flat)
    
    def _tensor_residual_grad(U, V, W, T_flat):
        """Compute gradient of residual squared w.r.t. U, V, W"""
        T_approx = np.einsum('ar,br,cr->abc', U, V, W).reshape(-1)
        diff = T_approx - T_flat
        # d/dU: sum_{b,c} diff[b,c] V[b,r] W[c,r] for each a,r
        # Gradient w.r.t. U[a,r] = sum_{b,c} diff[a,b,c] V[b,r] W[c,r]
        grad_U = np.einsum('abc,br,cr->ar', diff, V, W)
        grad_V = np.einsum('abc,ar,cr->br', diff, U, W)
        grad_W = np.einsum('abc,ar,br->cr', diff, U, V)
        return 2 * grad_U, 2 * grad_V, 2 * grad_W
    
    def _als_step(U, V, W, T, rank):
        """Perform one ALS step: update U given V, W, then V given U, W, then W given U, V."""
        # Update U
        X = np.zeros((m*p, p*n, rank))
        for r in range(rank):
            X[:, :, r] = np.outer(V[:, r], W[:, r])
        X_flat = X.reshape(-1, rank)
        
        try:
            A = X_flat @ X_flat.T
            B = X_flat @ T_flat.T
            U_new = np.linalg.lstsq(A, B, rcond=None)[0].T
        except:
            U_new = U
            
        # Update V
        Y = np.zeros((n*m, p*n, rank))
        for r in range(rank):
            Y[:, :, r] = np.outer(U[:, r], W[:, r])
        Y_flat = Y.reshape(-1, rank)
        T_b_flat = T.transpose(1, 0, 2).reshape(m*p, -1)
        
        try:
            A = Y_flat @ Y_flat.T
            C = Y_flat @ T_b_flat.T
            V_new = np.linalg.lstsq(A, C, rcond=None)[0].T
        except:
            V_new = V
            
        # Update W
        Z = np.zeros((n*m, m*p, rank))
        for r in range(rank):
            Z[:, :, r] = np.outer(U[:, r], V[:, r])
        Z_flat = Z.reshape(-1, rank)
        T_c_flat = T.transpose(2, 0, 1).reshape(p*n, -1)
        
        try:
            A = Z_flat @ Z_flat.T
            D = Z_flat @ T_c_flat.T
            W_new = np.linalg.lstsq(A, D, rcond=None)[0].T
        except:
            W_new = W
            
        return U_new, V_new, W_new
    
    def _gradient_descent_polish(U, V, W, T_flat, max_iters=200, tol=1e-12):
        """Polish using gradient descent with adaptive step size"""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        rank = U.shape[1]
        
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_res = _tensor_residual(U, V, W, T_flat)
        
        lr = 0.01
        for it in range(max_iters):
            if _remaining() < 5:
                break
            
            res = _tensor_residual(U, V, W, T_flat)
            if res < best_res:
                best_res = res
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if res < tol:
                break
            
            gU, gV, gW = _tensor_residual_grad(U, V, W, T_flat)
            
            # Adaptive step size based on gradient norm
            gnorm = np.sqrt(np.sum(gU**2) + np.sum(gV**2) + np.sum(gW**2))
            if gnorm > 0:
                step = lr / gnorm
            else:
                break
            
            U -= step * gU
            V -= step * gV
            W -= step * gW
            
            # Check if we made progress
            new_res = _tensor_residual(U, V, W, T_flat)
            if new_res > res * 1.01:
                # Step was too large, reduce learning rate
                lr *= 0.5
                U += step * gU
                V += step * gV
                W += step * gW
            else:
                # Good progress, maybe increase lr slightly
                lr = min(lr * 1.1, 0.1)
        
        return best_U, best_V, best_W
    
    def _verify(U, V, W):
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        return np.max(np.abs(T_approx - T)) < 1e-9
    
    def _rational_reconstruct(U, V, W, max_denom=16):
        """Try to find rational approximations with small denominators"""
        from math import gcd
        
        def _to_fraction(x, max_denom):
            """Convert float to Fraction with denominator <= max_denom"""
            # Try denominators from 1 to max_denom
            for d in range(1, max_denom + 1):
                num = round(x * d)
                if abs(x * d - num) < 1e-9:
                    g = gcd(abs(num), d)
                    return Fraction(num // g, d // g)
            return None
        
        # First check if values are already close to integers or half-integers
        for denom in [1, 2, 3, 4, 6, 8, 12, 16]:
            U_r = np.round(U * denom) / denom
            V_r = np.round(V * denom) / denom
            W_r = np.round(W * denom) / denom
            
            # Check if exactly representable
            if not np.allclose(U_r * denom, np.round(U_r * denom)): 
                continue
            if not np.allclose(V_r * denom, np.round(V_r * denom)): 
                continue
            if not np.allclose(W_r * denom, np.round(W_r * denom)): 
                continue
            
            T_approx = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
            if np.max(np.abs(T_approx - T)) < 1e-10:
                return U_r, V_r, W_r
        
        # Try per-element rational reconstruction
        U_frac = np.zeros_like(U)
        V_frac = np.zeros_like(V)
        W_frac = np.zeros_like(W)
        
        success = True
        for i in range(U.shape[0]):
            for j in range(U.shape[1]):
                f = _to_fraction(U[i,j], max_denom)
                if f is None:
                    success = False
                    break
                U_frac[i,j] = float(f)
            if not success:
                break
        
        if not success:
            for i in range(V.shape[0]):
                for j in range(V.shape[1]):
                    f = _to_fraction(V[i,j], max_denom)
                    if f is None:
                        success = False
                        break
                    V_frac[i,j] = float(f)
                if not success:
                    break
        
        if not success:
            for i in range(W.shape[0]):
                for j in range(W.shape[1]):
                    f = _to_fraction(W[i,j], max_denom)
                    if f is None:
                        success = False
                        break
                    W_frac[i,j] = float(f)
                if not success:
                    break
        
        if success:
            T_approx = np.einsum('ar,br,cr->abc', U_frac, V_frac, W_frac)
            if np.max(np.abs(T_approx - T)) < 1e-10:
                return U_frac, V_frac, W_frac
        
        return None
    
    best_U, best_V, best_W = None, None, None
    best_rank = n * m * p  # 18
    
    rng = np.random.default_rng(42)
    
    # Try ranks from 13 up to 17
    for rank in range(13, 18):
        if _remaining() < 15:
            break
            
        found = False
        max_restarts = 300 if rank == 13 else 500
        
        for restart in range(max_restarts):
            if _remaining() < 10:
                break
                
            # Multiple initialization strategies
            if restart % 3 == 0:
                # Random normal
                scale = 0.5
                U = rng.standard_normal((n*m, rank)) * scale
                V = rng.standard_normal((m*p, rank)) * scale
                W = rng.standard_normal((p*n, rank)) * scale
            elif restart % 3 == 1:
                # Sparse initialization
                U = np.zeros((n*m, rank))
                V = np.zeros((m*p, rank))
                W = np.zeros((p*n, rank))
                for r in range(rank):
                    idx_u = rng.integers(0, n*m, size=2)
                    idx_v = rng.integers(0, m*p, size=2)
                    idx_w = rng.integers(0, p*n, size=2)
                    U[idx_u, r] = rng.choice([-1, 1], size=2)
                    V[idx_v, r] = rng.choice([-1, 1], size=2)
                    W[idx_w, r] = rng.choice([-1, 1], size=2)
            else:
                # Structured initialization based on tensor
                U = np.zeros((n*m, rank))
                V = np.zeros((m*p, rank))
                W = np.zeros((p*n, rank))
                for r in range(rank):
                    i = rng.integers(0, n)
                    j = rng.integers(0, m)
                    k = rng.integers(0, p)
                    U[i*m+j, r] = 1.0
                    V[j*p+k, r] = 1.0
                    W[k*n+i, r] = 1.0
            
            # ALS iterations
            prev_err = np.inf
            for it in range(150):
                if _remaining() < 5:
                    break
                    
                U, V, W = _als_step(U, V, W, T, rank)
                
                # Check error
                T_approx = np.einsum('ar,br,cr->abc', U, V, W)
                err = np.max(np.abs(T_approx - T))
                
                if err < 1e-9:
                    # Try rational reconstruction
                    result = _rational_reconstruct(U, V, W)
                    if result is not None:
                        best_U, best_V, best_W = result
                        best_rank = rank
                        found = True
                        break
                
                # Check for convergence stagnation
                if abs(prev_err - err) < 1e-12:
                    # Try gradient descent polish
                    U, V, W = _gradient_descent_polish(U, V, W, T_flat, max_iters=50, tol=1e-10)
                    T_approx = np.einsum('ar,br,cr->abc', U, V, W)
                    err = np.max(np.abs(T_approx - T))
                    if err < 1e-9:
                        result = _rational_reconstruct(U, V, W)
                        if result is not None:
                            best_U, best_V, best_W = result
                            best_rank = rank
                            found = True
                            break
                    break
                prev_err = err
            
            if found:
                break
                
        if found:
            break
    
    # If we didn't find a better rank, return schoolbook
    if best_rank == n * m * p:
        U = np.zeros((n*m, n*m*p))
        V = np.zeros((m*p, n*m*p))
        W = np.zeros((p*n, n*m*p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        best_U, best_V, best_W = U, V, W
    
    # Convert to lists of floats
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
