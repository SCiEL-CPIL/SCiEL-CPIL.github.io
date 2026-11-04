"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys
from itertools import product
import scipy.optimize as opt

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 72
    
    # Target tensor T[i, j, k] = 1 for all i,j,k
    T = np.ones((n, m, p))
    T_flat = T.flatten()
    
    NM, MP, PN = n*m, m*p, p*n

    def check_exact(U, V, W, rank):
        """
        Check if U, V, W with given rank exactly reconstruct T.
        U: shape (n, m, rank)
        V: shape (m, p, rank)
        W: shape (p, n, rank)
        """
        T_rec = np.einsum('ijc,jkc,kic->ijk', U, V, W)
        return np.allclose(T_rec, T, atol=1e-9)

    def schoolbook_init():
        """Create the schoolbook (trivial) decomposition"""
        U = np.zeros((n, m, rank_schoolbook))
        V = np.zeros((m, p, rank_schoolbook))
        W = np.zeros((p, n, rank_schoolbook))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i, j, col] = 1
                    V[j, k, col] = 1
                    W[k, i, col] = 1
                    col += 1
        return U, V, W

    def objective_and_grad(x, rank):
        """
        Compute objective and gradient for L-BFGS-B.
        x: flat vector containing U, V, W
        """
        size_U = n * m * rank
        size_V = m * p * rank
        size_W = p * n * rank
        
        U_flat = x[:size_U]
        V_flat = x[size_U:size_U+size_V]
        W_flat = x[size_U+size_V:]
        
        U = U_flat.reshape(n, m, rank)
        V = V_flat.reshape(m, p, rank)
        W = W_flat.reshape(p, n, rank)
        
        T_rec = np.einsum('ijc,jkc,kic->ijk', U, V, W)
        err = T_rec - T
        loss = np.sum(err**2)
        
        # Gradients
        grad_U = 2 * np.einsum('ijk,jkc,kic->ijc', err, V, W)
        grad_V = 2 * np.einsum('ijk,ijc,kic->jkc', err, U, W)
        grad_W = 2 * np.einsum('ijk,ijc,jkc->kic', err, U, V)
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return loss, grad

    def try_snap(U_sol, V_sol, W_sol, rank):
        """
        Try to snap the solution to integer/half-integer values.
        Returns (U, V, W) if successful, else None.
        """
        # Try rounding to nearest 0.5
        U_snapped = np.round(U_sol * 2) / 2
        V_snapped = np.round(V_sol * 2) / 2
        W_snapped = np.round(W_sol * 2) / 2
        if check_exact(U_snapped, V_snapped, W_snapped, rank):
            return (U_snapped, V_snapped, W_snapped)
        
        # Try rounding to nearest integer
        U_int = np.round(U_sol)
        V_int = np.round(V_sol)
        W_int = np.round(W_sol)
        if check_exact(U_int, V_int, W_int, rank):
            return (U_int, V_int, W_int)
        
        # Try finer grids
        for scale in [4, 8, 10, 16]:
            U_s = np.round(U_sol * scale) / scale
            V_s = np.round(V_sol * scale) / scale
            W_s = np.round(W_sol * scale) / scale
            if check_exact(U_s, V_s, W_s, rank):
                return (U_s, V_s, W_s)
        
        # Try local refinement around snapped values
        # If error is very small, try a few iterations of gradient descent on the snapped values
        best_err = np.inf
        best_sol = None
        
        for scale in [2, 4, 8]:
            U_base = np.round(U_sol * scale) / scale
            V_base = np.round(V_sol * scale) / scale
            W_base = np.round(W_sol * scale) / scale
            
            # Check base
            T_rec = np.einsum('ijc,jkc,kic->ijk', U_base, V_base, W_base)
            err = np.sum((T_rec - T)**2)
            if err < best_err:
                best_err = err
                best_sol = (U_base, V_base, W_base)
            
            if err < 1e-6:
                # Try small perturbations
                for delta in [0.25, -0.25, 0.125, -0.125]:
                    for idx_u in range(n*m*rank):
                        U_test = U_base.flatten().copy()
                        U_test[idx_u] += delta
                        U_test = U_test.reshape(n, m, rank)
                        
                        T_rec = np.einsum('ijc,jkc,kic->ijk', U_test, V_base, W_base)
                        err_test = np.sum((T_rec - T)**2)
                        if err_test < best_err:
                            best_err = err_test
                            best_sol = (U_test, V_base.copy(), W_base.copy())
                    
                    for idx_v in range(m*p*rank):
                        V_test = V_base.flatten().copy()
                        V_test[idx_v] += delta
                        V_test = V_test.reshape(m, p, rank)
                        
                        T_rec = np.einsum('ijc,jkc,kic->ijk', U_base, V_test, W_base)
                        err_test = np.sum((T_rec - T)**2)
                        if err_test < best_err:
                            best_err = err_test
                            best_sol = (U_base.copy(), V_test, W_base.copy())
                    
                    for idx_w in range(p*n*rank):
                        W_test = W_base.flatten().copy()
                        W_test[idx_w] += delta
                        W_test = W_test.reshape(p, n, rank)
                        
                        T_rec = np.einsum('ijc,jkc,kic->ijk', U_base, V_base, W_test)
                        err_test = np.sum((T_rec - T)**2)
                        if err_test < best_err:
                            best_err = err_test
                            best_sol = (U_base.copy(), V_base.copy(), W_test)
        
        if best_sol is not None and best_err < 1e-9:
            return best_sol
        
        return None

    def try_rank_optimization(rank, num_restarts=10, time_per_restart=5.0):
        """
        Try to find a decomposition of given rank using L-BFGS-B optimization.
        Returns (U, V, W) if found, else None.
        """
        best_result = None
        best_err = np.inf
        
        size_U = n * m * rank
        size_V = m * p * rank
        size_W = p * n * rank
        total_size = size_U + size_V + size_W
        
        for r in range(num_restarts):
            if time.time() > deadline - 10:
                break
            
            # Diverse random initialization
            np.random.seed(r * 12345 + 67890)
            # Mix of different scales to explore different regions
            if r % 3 == 0:
                scale = 0.5
            elif r % 3 == 1:
                scale = 1.0
            else:
                scale = 0.25
            
            U_init = np.random.randn(n, m, rank) * scale
            V_init = np.random.randn(m, p, rank) * scale
            W_init = np.random.randn(p, n, rank) * scale
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Set a local deadline for this restart
            restart_deadline = time.time() + time_per_restart
            
            # L-BFGS-B with analytic gradient
            try:
                res = opt.minimize(objective_and_grad, x0, method='L-BFGS-B', 
                                    jac=True, 
                                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 30})
                
                if time.time() > restart_deadline:
                    continue
                    
                err = res.fun
                if err < best_err:
                    best_err = err
                    # Extract solution
                    U_sol = res.x[:size_U].reshape(n, m, rank)
                    V_sol = res.x[size_U:size_U+size_V].reshape(m, p, rank)
                    W_sol = res.x[size_U+size_V:].reshape(p, n, rank)
                    best_result = (U_sol, V_sol, W_sol)
                
                # If error is extremely small, try to snap
                if err < 1e-8:
                    snapped = try_snap(U_sol, V_sol, W_sol, rank)
                    if snapped is not None:
                        return snapped
                        
            except Exception:
                continue
        
        # If we found a good solution but didn't snap perfectly, try more aggressive snapping
        if best_result is not None and best_err < 1e-4:
            U_sol, V_sol, W_sol = best_result
            snapped = try_snap(U_sol, V_sol, W_sol, rank)
            if snapped is not None:
                return snapped
            
            # Try one more optimization pass starting from the best solution
            if time.time() < deadline - 20:
                x0 = np.concatenate([U_sol.flatten(), V_sol.flatten(), W_sol.flatten()])
                try:
                    res = opt.minimize(objective_and_grad, x0, method='L-BFGS-B', 
                                        jac=True,
                                        options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 20})
                    if res.fun < 1e-10:
                        U_sol2 = res.x[:size_U].reshape(n, m, rank)
                        V_sol2 = res.x[size_U:size_U+size_V].reshape(m, p, rank)
                        W_sol2 = res.x[size_U+size_V:].reshape(p, n, rank)
                        snapped = try_snap(U_sol2, V_sol2, W_sol2, rank)
                        if snapped is not None:
                            return snapped
                except Exception:
                    pass
        
        return None

    # Strategy:
    # 1. Start with schoolbook as fallback (always valid)
    # 2. Try to find rank 71 (most likely improvement if any exists)
    # 3. If time permits, try rank 70
    
    U_sb, V_sb, W_sb = schoolbook_init()
    best_result = (U_sb, V_sb, W_sb)
    best_rank = rank_schoolbook
    
    # Try rank 71 with several restarts
    if time.time() < deadline - 60:
        res = try_rank_optimization(71, num_restarts=8, time_per_restart=7.0)
        if res is not None:
            best_result = res
            best_rank = 71
    
    # If we have time and rank 71 failed, try rank 70
    if best_rank == 72 and time.time() < deadline - 30:
        res = try_rank_optimization(70, num_restarts=5, time_per_restart=5.0)
        if res is not None:
            best_result = res
            best_rank = 70
    
    U, V, W = best_result
    
    # Convert to lists of lists with required shapes
    U_out = U.reshape(NM, best_rank).tolist()
    V_out = V.reshape(MP, best_rank).tolist()
    W_out = W.reshape(PN, best_rank).tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
