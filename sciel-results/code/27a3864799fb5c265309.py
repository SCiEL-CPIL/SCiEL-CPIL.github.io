"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s
    
    # Schoolbook decomposition
    rank_school = n * m * p
    U = np.zeros((n * m, rank_school))
    V = np.zeros((m * p, rank_school))
    W = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    best_rank = rank_school
    
    # Try to find lower rank via numerical optimization
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    def _adam_update(params, grads, m_params, v_params, t, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        """Adam optimizer update"""
        for p, g, mp, vp in zip(params, grads, m_params, v_params):
            mp[:] = beta1 * mp + (1 - beta1) * g
            vp[:] = beta2 * vp + (1 - beta2) * (g ** 2)
            mp_hat = mp / (1 - beta1 ** t)
            vp_hat = vp / (1 - beta2 ** t)
            p[:] -= lr * mp_hat / (np.sqrt(vp_hat) + eps)
    
    def _try_rational_reconstruct(U_r, V_r, W_r, rank, tol=1e-6, max_denom=10000):
        """Try to reconstruct rational entries and verify exactness"""
        from fractions import Fraction
        U_f = np.zeros((n*m, rank), dtype=object)
        V_f = np.zeros((m*p, rank), dtype=object)
        W_f = np.zeros((p*n, rank), dtype=object)
        ok = True
        for i in range(n*m):
            for j in range(rank):
                f = Fraction(U_r[i,j]).limit_denominator(max_denom)
                if abs(float(f) - U_r[i,j]) > tol:
                    return None
                U_f[i,j] = f
        for i in range(m*p):
            for j in range(rank):
                f = Fraction(V_r[i,j]).limit_denominator(max_denom)
                if abs(float(f) - V_r[i,j]) > tol:
                    return None
                V_f[i,j] = f
        for i in range(p*n):
            for j in range(rank):
                f = Fraction(W_r[i,j]).limit_denominator(max_denom)
                if abs(float(f) - W_r[i,j]) > tol:
                    return None
                W_f[i,j] = f
        # Verify exactness with high precision
        U_np = np.array([[float(x) for x in row] for row in U_f], dtype=float)
        V_np = np.array([[float(x) for x in row] for row in V_f], dtype=float)
        W_np = np.array([[float(x) for x in row] for row in W_f], dtype=float)
        D_exact = np.einsum('ar,br,cr->abc', U_np, V_np, W_np, optimize=True)
        if np.allclose(D_exact, T, atol=1e-10):
            return {
                "U": [[float(x) for x in row] for row in U_f],
                "V": [[float(x) for x in row] for row in V_f],
                "W": [[float(x) for x in row] for row in W_f]
            }
        return None
    
    # Try ranks from 53 down
    for rank in range(rank_school - 1, 30, -1):
        if time.time() > deadline - 5:
            break
        
        # Try multiple random initializations with different strategies
        n_trials = 8
        for trial in range(n_trials):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(123 + trial * 7)
            
            # Different initialization strategies
            if trial < 4:
                # Dense random
                U_r = np.random.randn(n*m, rank) * 0.5
                V_r = np.random.randn(m*p, rank) * 0.5
                W_r = np.random.randn(p*n, rank) * 0.5
            else:
                # Sparse random with some structure
                U_r = np.random.randn(n*m, rank)
                V_r = np.random.randn(m*p, rank)
                W_r = np.random.randn(p*n, rank)
                # Add sparsity
                mask_U = np.random.rand(n*m, rank) < 0.3
                mask_V = np.random.rand(m*p, rank) < 0.3
                mask_W = np.random.rand(p*n, rank) < 0.3
                U_r[mask_U] *= 3
                V_r[mask_V] *= 3
                W_r[mask_W] *= 3
            
            # Adam optimizer
            m_U = np.zeros_like(U_r)
            v_U = np.zeros_like(U_r)
            m_V = np.zeros_like(V_r)
            v_V = np.zeros_like(V_r)
            m_W = np.zeros_like(W_r)
            v_W = np.zeros_like(W_r)
            
            lr = 0.05
            n_steps = 300
            for step in range(n_steps):
                if time.time() > deadline - 5:
                    break
                D = np.einsum('ar,br,cr->abc', U_r, V_r, W_r, optimize=True)
                err = D - T
                err_norm = np.linalg.norm(err)
                
                if err_norm < 1e-10:
                    break
                
                # Compute gradients
                grad_U = np.einsum('abc,br,cr->ar', err, V_r, W_r, optimize=True)
                grad_V = np.einsum('abc,ar,cr->br', err, U_r, W_r, optimize=True)
                grad_W = np.einsum('abc,ar,br->cr', err, U_r, V_r, optimize=True)
                
                # Adam update
                t = step + 1
                _adam_update([U_r], [grad_U], [m_U], [v_U], t, lr)
                _adam_update([V_r], [grad_V], [m_V], [v_V], t, lr)
                _adam_update([W_r], [grad_W], [m_W], [v_W], t, lr)
            
            # Final check
            D_final = np.einsum('ar,br,cr->abc', U_r, V_r, W_r, optimize=True)
            if np.linalg.norm(D_final - T) < 1e-8:
                result = _try_rational_reconstruct(U_r, V_r, W_r, rank)
                if result is not None:
                    best_rank = rank
                    best_result = result
                    break  # Found valid, move to next rank
        
        if time.time() > deadline - 5:
            break
    
    return best_result

# EVOLVE-BLOCK-END
