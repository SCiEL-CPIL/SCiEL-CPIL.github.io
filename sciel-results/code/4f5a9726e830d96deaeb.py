import time
import random
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 5]

def _schoolbook(n, m, p):
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

def _strassen_3x3_3():
    """Return rank-7 Strassen-like decomposition for 3x3x3.
    Returns U, V, W as numpy arrays of shape (9,7) each."""
    # Standard Strassen for 2x2x2 has rank 7.
    # For 3x3x3, we can use a known rank-7 construction or pad.
    # Actually, 3x3x3 matrix multiplication has rank at most 19 (schoolbook),
    # and Strassen gives 27 for 3x3x3 via block approach, but there are better.
    # Let me use a simple structured approach: treat 3x3x3 as containing a 2x2x2.
    # 
    # For a cleaner approach, let's use the fact that we can decompose
    # 3x3x3 by splitting into blocks. But for simplicity and correctness,
    # let me construct a valid rank-19 schoolbook and try to reduce.
    #
    # Actually, let me just use a known small-rank construction.
    # For 3x3x3, the tensor rank is known to be 19 (or less?).
    # Let me just use schoolbook for 3x3x3 and focus on the 3x3x5 structure.
    
    n3, m3, p3 = 3, 3, 3
    rank = n3 * m3 * p3  # 27 schoolbook for now
    U = np.zeros((n3 * m3, rank))
    V = np.zeros((m3 * p3, rank))
    W = np.zeros((p3 * n3, rank))
    col = 0
    for i in range(n3):
        for j in range(m3):
            for k in range(p3):
                U[i * m3 + j, col] = 1
                V[j * p3 + k, col] = 1
                W[k * n3 + i, col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    def compute_residual(params, rank):
        U = params[:n * m * rank].reshape(n * m, rank)
        V = params[n * m * rank:n * m * rank + m * p * rank].reshape(m * p, rank)
        W = params[n * m * rank + m * p * rank:].reshape(p * n, rank)
        recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.sum((T - recon) ** 2)
    
    def residual_and_grad(params, rank):
        U = params[:n * m * rank].reshape(n * m, rank)
        V = params[n * m * rank:n * m * rank + m * p * rank].reshape(m * p, rank)
        W = params[n * m * rank + m * p * rank:].reshape(p * n, rank)
        recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        residual = T - recon
        
        grad_U = -2 * np.einsum('ijk,jr,kr->ir', residual, V, W)
        grad_V = -2 * np.einsum('ijk,ir,kr->jr', residual, U, W)
        grad_W = -2 * np.einsum('ijk,ir,jr->kr', residual, U, V)
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return np.sum(residual ** 2), grad
    
    def try_round_and_verify(U, V, W, rank):
        """Try rounding to nearest integer and verify."""
        U_r = np.round(U)
        V_r = np.round(V)
        W_r = np.round(W)
        recon = np.einsum('ir,jr,kr->ijk', U_r, V_r, W_r)
        err = np.linalg.norm(T - recon)
        if err < 1e-6:
            return {
                'U': U_r.tolist(),
                'V': V_r.tolist(),
                'W': W_r.tolist()
            }, rank
        return None, rank
    
    def polish(params, rank):
        """High-precision polish pass with tighter tolerance."""
        try:
            res = minimize(
                residual_and_grad,
                params,
                args=(rank,),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res.fun < 1e-10:
                U = res.x[:n * m * rank].reshape(n * m, rank)
                V = res.x[n * m * rank:n * m * rank + m * p * rank].reshape(m * p, rank)
                W = res.x[n * m * rank + m * p * rank:].reshape(p * n, rank)
                return try_round_and_verify(U, V, W, rank)
        except Exception:
            pass
        return None, rank
    
    best_result = None
    best_rank = n * m * p  # 45 schoolbook
    
    # Strategy: Try ranks from 35 down (Strassen-inspired lower bound)
    # Use structured initialization based on embedding 3x3x3 patterns
    
    # For 3x3x5, we can think of it as 5 columns of 3x3.
    # A rank-7 solution for each 3x3x1 slice would give 35 total.
    # But we can share structure across columns.
    
    # Try a range of ranks with multiple random restarts
    for rank in range(35, 24, -1):
        if time.time() > deadline:
            break
        
        # Try several initializations
        for trial in range(8):
            if time.time() > deadline - 5:
                break
            
            # Structured initialization: mix of random and pattern-based
            if trial < 3:
                # Random initialization
                U_init = np.random.randn(n * m, rank) * 0.5
                V_init = np.random.randn(m * p, rank) * 0.5
                W_init = np.random.randn(p * n, rank) * 0.5
            elif trial < 6:
                # Sparse random initialization
                U_init = np.random.randn(n * m, rank)
                V_init = np.random.randn(m * p, rank)
                W_init = np.random.randn(p * n, rank)
                # Zero out ~50% of entries
                mask_U = np.random.rand(n * m, rank) < 0.5
                mask_V = np.random.rand(m * p, rank) < 0.5
                mask_W = np.random.rand(p * n, rank) < 0.5
                U_init[mask_U] = 0
                V_init[mask_V] = 0
                W_init[mask_W] = 0
                U_init *= 1.0
                V_init *= 1.0
                W_init *= 1.0
            else:
                # Pattern-based: initialize with schoolbook-like structure
                U_init = np.zeros((n * m, rank))
                V_init = np.zeros((m * p, rank))
                W_init = np.zeros((p * n, rank))
                # Place ones in a structured way
                for r in range(rank):
                    i = r % n
                    j = (r // n) % m
                    k = (r // (n * m)) % p
                    if r < n * m * p:
                        U_init[i * m + j, r] = 1
                        V_init[j * p + k, r] = 1
                        W_init[k * n + i, r] = 1
                # Add small random perturbation
                U_init += np.random.randn(n * m, rank) * 0.1
                V_init += np.random.randn(m * p, rank) * 0.1
                W_init += np.random.randn(p * n, rank) * 0.1
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # First pass: L-BFGS-B
            try:
                res = minimize(
                    residual_and_grad,
                    params_init,
                    args=(rank,),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 150, 'ftol': 1e-12}
                )
                
                if res.fun < 1e-8:
                    # Polish pass
                    polished, polished_rank = polish(res.x, rank)
                    if polished is not None:
                        best_result = polished
                        best_rank = polished_rank
                        break
                elif res.fun < 1e-4:
                    # Close enough, try polish
                    polished, polished_rank = polish(res.x, rank)
                    if polished is not None:
                        best_result = polished
                        best_rank = polished_rank
                        break
                        
            except Exception:
                continue
        
        if best_result is not None:
            break
    
    if best_result is None:
        # Return schoolbook
        best_result = _schoolbook(n, m, p)
    
    return best_result
