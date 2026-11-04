import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    best_rank = n * m * p
    best_solution = None
    
    # Generate schoolbook solution as fallback
    def get_schoolbook():
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
        return {"U": U, "V": V, "W": W}, rank

    schoolbook_sol, schoolbook_rank = get_schoolbook()
    best_solution = schoolbook_sol
    best_rank = schoolbook_rank

    def is_valid(U, V, W, rank, atol=1e-9):
        Rec = np.zeros((n, m, p))
        for r in range(rank):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            Rec += np.einsum('ij,jk,ki->ijk', u, v, w, optimize=True)
        return np.allclose(Rec, 1.0, atol=atol)

    def compute_error(U, V, W, rank):
        Rec = np.zeros((n, m, p))
        for r in range(rank):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            Rec += np.einsum('ij,jk,ki->ijk', u, v, w, optimize=True)
        return Rec - 1.0

    def error_and_grad(params, rank):
        size_U = n * m * rank
        size_V = m * p * rank
        size_W = p * n * rank
        
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U + size_V]
        W_flat = params[size_U + size_V:]
        
        U = U_flat.reshape(n * m, rank)
        V = V_flat.reshape(m * p, rank)
        W = W_flat.reshape(p * n, rank)
        
        err = compute_error(U, V, W, rank)
        loss = np.sum(err**2)
        
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for r in range(rank):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            
            grad_U[:, r] = 2.0 * np.einsum('ijk,jk,ki->ij', err, v, w, optimize=True).flatten()
            grad_V[:, r] = 2.0 * np.einsum('ijk,ij,ki->jk', err, u, w, optimize=True).flatten()
            grad_W[:, r] = 2.0 * np.einsum('ijk,ij,jk->ki', err, u, v, optimize=True).flatten()
            
        grad_flat = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        
        return loss, grad_flat

    def polish_solution(U, V, W, rank, max_iters=500):
        """High-precision polish using gradient descent with adaptive step size"""
        U = U.copy().astype(np.float64)
        V = V.copy().astype(np.float64)
        W = W.copy().astype(np.float64)
        
        params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        lr = 0.01
        best_loss = np.inf
        best_params = params.copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
                
            loss, grad = error_and_grad(params, rank)
            
            if loss < best_loss:
                best_loss = loss
                best_params = params.copy()
            
            if loss < 1e-12:
                break
            
            # Adaptive step size
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-15:
                break
            
            step = lr / (1 + it * 0.01)
            params = params - step * grad
            
            # Occasionally try to snap and check
            if it % 50 == 0:
                size_U = n * m * rank
                size_V = m * p * rank
                U_snap = np.round(params[:size_U].reshape(n * m, rank) * 2) / 2
                V_snap = np.round(params[size_U:size_U + size_V].reshape(m * p, rank) * 2) / 2
                W_snap = np.round(params[size_U + size_V:].reshape(p * n, rank) * 2) / 2
                if is_valid(U_snap, V_snap, W_snap, rank, atol=1e-8):
                    return U_snap, V_snap, W_snap
        
        # Final snap attempt
        size_U = n * m * rank
        size_V = m * p * rank
        U_final = best_params[:size_U].reshape(n * m, rank)
        V_final = best_params[size_U:size_U + size_V].reshape(m * p, rank)
        W_final = best_params[size_U + size_V:].reshape(p * n, rank)
        
        # Try snapping to half-integers
        U_snap = np.round(U_final * 2) / 2
        V_snap = np.round(V_final * 2) / 2
        W_snap = np.round(W_final * 2) / 2
        if is_valid(U_snap, V_snap, W_snap, rank, atol=1e-8):
            return U_snap, V_snap, W_snap
        
        # Try integer snapping
        U_int = np.round(U_final)
        V_int = np.round(V_final)
        W_int = np.round(W_final)
        if is_valid(U_int, V_int, W_int, rank, atol=1e-8):
            return U_int, V_int, W_int
        
        return None, None, None

    def try_rank(rank, num_restarts=30):
        nonlocal best_solution, best_rank
        if rank >= best_rank:
            return False
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                return False
                
            # Diversified initialization strategies
            strategy = restart % 5
            
            if strategy == 0:
                # Random Gaussian
                U_init = np.random.randn(n * m, rank)
                V_init = np.random.randn(m * p, rank)
                W_init = np.random.randn(p * n, rank)
                scale = 1.0 / np.sqrt(rank)
                U_init *= scale
                V_init *= scale
                W_init *= scale
            elif strategy == 1:
                # Sparse random with ±1
                U_init = np.random.choice([-1, 0, 1], size=(n * m, rank))
                V_init = np.random.choice([-1, 0, 1], size=(m * p, rank))
                W_init = np.random.choice([-1, 0, 1], size=(p * n, rank))
            elif strategy == 2:
                # Half-integer random
                U_init = np.round(np.random.randn(n * m, rank) * 2) / 2
                V_init = np.round(np.random.randn(m * p, rank) * 2) / 2
                W_init = np.round(np.random.randn(p * n, rank) * 2) / 2
            elif strategy == 3:
                # Structured: try to build from partial solutions
                U_init = np.zeros((n * m, rank))
                V_init = np.zeros((m * p, rank))
                W_init = np.zeros((p * n, rank))
                for r in range(rank):
                    # Assign random patterns
                    for i in range(n):
                        for j in range(m):
                            U_init[i * m + j, r] = np.random.choice([-1, 0, 1])
                    for j in range(m):
                        for k in range(p):
                            V_init[j * p + k, r] = np.random.choice([-1, 0, 1])
                    for k in range(p):
                        for i in range(n):
                            W_init[k * n + i, r] = np.random.choice([-1, 0, 1])
            else:
                # Small random with bias toward 0
                U_init = np.random.randn(n * m, rank) * 0.5
                V_init = np.random.randn(m * p, rank) * 0.5
                W_init = np.random.randn(p * n, rank) * 0.5
                # Snap some to half-integers
                mask = np.random.rand(n * m, rank) < 0.5
                U_init[mask] = np.round(U_init[mask] * 2) / 2
                mask = np.random.rand(m * p, rank) < 0.5
                V_init[mask] = np.round(V_init[mask] * 2) / 2
                mask = np.random.rand(p * n, rank) < 0.5
                W_init[mask] = np.round(W_init[mask] * 2) / 2
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Use L-BFGS-B with tighter tolerances
            try:
                res = minimize(error_and_grad, params_init, method='L-BFGS-B', jac=True, 
                              options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50})
            except Exception:
                continue
                
            if res.fun < 1e-6:
                # Extract optimized U, V, W
                size_U = n * m * rank
                size_V = m * p * rank
                U_opt = res.x[:size_U].reshape(n * m, rank)
                V_opt = res.x[size_U:size_U + size_V].reshape(m * p, rank)
                W_opt = res.x[size_U + size_V:].reshape(p * n, rank)
                
                # Try direct snap first
                U_snap = np.round(U_opt * 2) / 2
                V_snap = np.round(V_opt * 2) / 2
                W_snap = np.round(W_opt * 2) / 2
                if is_valid(U_snap, V_snap, W_snap, rank):
                    best_solution = {"U": U_snap, "V": V_snap, "W": W_snap}
                    best_rank = rank
                    return True
                    
                # Try integer
                U_int = np.round(U_opt)
                V_int = np.round(V_opt)
                W_int = np.round(W_opt)
                if is_valid(U_int, V_int, W_int, rank):
                    best_solution = {"U": U_int, "V": V_int, "W": W_int}
                    best_rank = rank
                    return True
                
                # Polish step
                U_polished, V_polished, W_polished = polish_solution(U_opt, V_opt, W_opt, rank, max_iters=300)
                if U_polished is not None:
                    best_solution = {"U": U_polished, "V": V_polished, "W": W_polished}
                    best_rank = rank
                    return True

        return False

    # Search for lower ranks - start from known good bounds
    # For 3x3x5, literature suggests rank ~23-26
    # Try from 20 upward to find the minimum
    for rank in range(20, best_rank):
        if time.time() > deadline:
            break
        if try_rank(rank, num_restarts=25):
            break

    # If we found something, try to go even lower
    if best_rank < 45:
        for rank in range(15, best_rank):
            if time.time() > deadline:
                break
            if try_rank(rank, num_restarts=15):
                break

    def to_lists(mat):
        return mat.tolist()

    return {
        "U": to_lists(best_solution["U"]),
        "V": to_lists(best_solution["V"]),
        "W": to_lists(best_solution["W"])
    }
# EVOLVE-BLOCK-END
