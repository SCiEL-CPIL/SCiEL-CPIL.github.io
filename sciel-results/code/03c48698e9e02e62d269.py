"""Tensor decomposition search for 3x4 * 4x7 matrix multiplication."""
DIMENSIONS = [3, 4, 7]

import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def _schoolbook_decomposition(n, m, p):
    """Return the standard rank-n*m*p decomposition as a list of (u, v, w) index tuples."""
    decomp = []
    for i, j, k in product(range(n), range(m), range(p)):
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        decomp.append((u_idx, v_idx, w_idx, 1))
    return decomp

def _decomposition_is_valid(U, V, W, n, m, p, tol=1e-8):
    """Check if the decomposition exactly represents the multiplication tensor."""
    rank = len(U[0]) if isinstance(U[0], list) else U.shape[1]
    
    # Build tensor from decomposition
    for i, j, k in product(range(n), range(m), range(p)):
        u_row = i * m + j
        v_row = j * p + k
        w_row = k * n + i
        val = 0.0
        for r in range(rank):
            u_val = U[u_row][r] if isinstance(U[0], list) else U[u_row, r]
            v_val = V[v_row][r] if isinstance(V[0], list) else V[v_row, r]
            w_val = W[w_row][r] if isinstance(W[0], list) else W[w_row, r]
            val += u_val * v_val * w_val
        if abs(val - 1.0) > tol:
            return False
    return True

def _try_round_half_integer(U, V, W, n, m, p):
    """Try to round U, V, W to half-integers while maintaining validity."""
    U_r = np.array(U, dtype=float)
    V_r = np.array(V, dtype=float)
    W_r = np.array(W, dtype=float)
    
    # Round to nearest half-integer
    U_half = np.round(U_r * 2) / 2
    V_half = np.round(V_r * 2) / 2
    W_half = np.round(W_r * 2) / 2
    
    # Check if still valid
    if _decomposition_is_valid(U_half, V_half, W_half, n, m, p, tol=1e-6):
        return U_half, V_half, W_half
    
    return None

def _random_init(n, m, p, rank, scale=1.0):
    """Random initialization for factor matrices."""
    U = np.random.randn(n * m, rank) * scale
    V = np.random.randn(m * p, rank) * scale
    W = np.random.randn(p * n, rank) * scale
    return U, V, W

def _objective_and_grad(params, n, m, p, rank):
    """Objective function and gradient for optimization: sum of squared errors."""
    U = params[:n*m*rank].reshape(n*m, rank)
    V = params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
    W = params[n*m*rank + m*p*rank:].reshape(p*n, rank)
    
    # Compute errors for all (i,j,k)
    errors = 0.0
    gU = np.zeros_like(U)
    gV = np.zeros_like(V)
    gW = np.zeros_like(W)
    
    for i, j, k in product(range(n), range(m), range(p)):
        u_row = i * m + j
        v_row = j * p + k
        w_row = k * n + i
        
        # val = sum_r U[u_row, r] * V[v_row, r] * W[w_row, r]
        val = np.dot(U[u_row], V[v_row] * W[w_row])
        err = val - 1.0
        errors += err * err
        
        # Gradient with respect to U[u_row]
        gU[u_row] += 2 * err * V[v_row] * W[w_row]
        # Gradient with respect to V[v_row]
        gV[v_row] += 2 * err * U[u_row] * W[w_row]
        # Gradient with respect to W[w_row]
        gW[w_row] += 2 * err * U[u_row] * V[v_row]
    
    grad = np.concatenate([gU.flatten(), gV.flatten(), gW.flatten()])
    return errors, grad

def _refine_decomposition(n, m, p, rank, U0, V0, W0, max_iters=5000, tol=1e-12, deadline=None):
    """Use gradient-based optimization to refine the decomposition."""
    U_flat = U0.flatten()
    V_flat = V0.flatten()
    W_flat = W0.flatten()
    x0 = np.concatenate([U_flat, V_flat, W_flat])
    
    # Use scipy minimize with L-BFGS-B
    from scipy.optimize import minimize
    
    result = minimize(_objective_and_grad, x0, args=(n, m, p, rank), method='L-BFGS-B', jac=True, 
                       options={'maxiter': max_iters, 'ftol': tol, 'gtol': 1e-10})
    
    if deadline is not None and time.time() > deadline:
        return None
    
    x_opt = result.x
    U_opt = x_opt[:n*m*rank].reshape(n*m, rank)
    V_opt = x_opt[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
    W_opt = x_opt[n*m*rank + m*p*rank:].reshape(p*n, rank)
    
    obj_val, _ = _objective_and_grad(x_opt, n, m, p, rank)
    if obj_val > 1e-10:
        return None
    
    return U_opt, V_opt, W_opt

def _refine_decomposition_manual(n, m, p, rank, U0, V0, W0, max_iters=5000, tol=1e-12, deadline=None):
    """Manual gradient descent refinement (alternative to scipy)."""
    U = U0.copy()
    V = V0.copy()
    W = W0.copy()
    
    lr = 0.01
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            return None
        
        # Compute gradients
        gU = np.zeros_like(U)
        gV = np.zeros_like(V)
        gW = np.zeros_like(W)
        total_err = 0.0
        
        for i, j, k in product(range(n), range(m), range(p)):
            u_row = i * m + j
            v_row = j * p + k
            w_row = k * n + i
            
            val = np.dot(U[u_row], V[v_row] * W[w_row])
            err = val - 1.0
            total_err += err * err
            
            gU[u_row] += 2 * err * V[v_row] * W[w_row]
            gV[v_row] += 2 * err * U[u_row] * W[w_row]
            gW[w_row] += 2 * err * U[u_row] * V[v_row]
        
        # Check convergence
        if total_err < tol:
            break
        
        # Update
        U -= lr * gU
        V -= lr * gV
        W -= lr * gW
    
    obj_val, _ = _objective_and_grad(
        np.concatenate([U.flatten(), V.flatten(), W.flatten()]), n, m, p, rank)
    if obj_val > 1e-10:
        return None
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook decomposition
    schoolbook = _schoolbook_decomposition(n, m, p)
    
    def _schoolbook_to_matrices(decomp):
        rank = len(decomp)
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for r, (u_idx, v_idx, w_idx, coeff) in enumerate(decomp):
            U[u_idx][r] = coeff
            V[v_idx][r] = coeff
            W[w_idx][r] = coeff
        return U, V, W
    
    U0, V0, W0 = _schoolbook_to_matrices(schoolbook)
    rank0 = len(schoolbook)
    
    # Try to reduce rank by finding lower rank decompositions
    best_U, best_V, best_W = U0, V0, W0
    best_rank = rank0
    
    # Attempt rank reduction through optimization
    # Theoretical lower bounds
    min_possible_rank = max(n * m, m * p, p * n)  # Basic lower bound
    
    # Try candidate ranks from lower bound up to schoolbook rank
    for candidate_rank in range(min_possible_rank, rank0):
        if time.time() > deadline - 10:
            break
        
        success = False
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
            
            U_init, V_init, W_init = _random_init(n, m, p, candidate_rank, scale=1.0)
            
            # Try scipy optimizer
            result = _refine_decomposition(n, m, p, candidate_rank, U_init, V_init, W_init,
                                           max_iters=3000, deadline=deadline - 5)
            if result is not None:
                U_opt, V_opt, W_opt = result
                # Try to round to half-integers
                rounded = _try_round_half_integer(U_opt, V_opt, W_opt, n, m, p)
                if rounded is not None:
                    U_h, V_h, W_h = rounded
                    if _decomposition_is_valid(U_h, V_h, W_h, n, m, p, tol=1e-8):
                        best_U, best_V, best_W = U_h, V_h, W_h
                        best_rank = candidate_rank
                        success = True
                        break
                else:
                    # Use unrounded if valid
                    if _decomposition_is_valid(U_opt, V_opt, W_opt, n, m, p, tol=1e-8):
                        best_U, best_V, best_W = U_opt, V_opt, W_opt
                        best_rank = candidate_rank
                        success = True
                        break
            
            # Try manual gradient descent
            result2 = _refine_decomposition_manual(n, m, p, candidate_rank, U_init, V_init, W_init,
                                                   max_iters=3000, deadline=deadline - 5)
            if result2 is not None:
                U_opt2, V_opt2, W_opt2 = result2
                rounded2 = _try_round_half_integer(U_opt2, V_opt2, W_opt2, n, m, p)
                if rounded2 is not None:
                    U_h2, V_h2, W_h2 = rounded2
                    if _decomposition_is_valid(U_h2, V_h2, W_h2, n, m, p, tol=1e-8):
                        best_U, best_V, best_W = U_h2, V_h2, W_h2
                        best_rank = candidate_rank
                        success = True
                        break
                else:
                    if _decomposition_is_valid(U_opt2, V_opt2, W_opt2, n, m, p, tol=1e-8):
                        best_U, best_V, best_W = U_opt2, V_opt2, W_opt2
                        best_rank = candidate_rank
                        success = True
                        break
        
        if success:
            continue
        else:
            break
    
    # If we found a better rank, return it. Otherwise return schoolbook.
    if best_rank < rank0:
        # Convert to lists if needed
        if isinstance(best_U, np.ndarray):
            best_U = best_U.tolist()
        if isinstance(best_V, np.ndarray):
            best_V = best_V.tolist()
        if isinstance(best_W, np.ndarray):
            best_W = best_W.tolist()
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        return {"U": U0, "V": V0, "W": W0}
# EVOLVE-BLOCK-END
