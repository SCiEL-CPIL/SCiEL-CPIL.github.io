"""Matrix multiplication 3,3,4 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 3, 4]

import time
import random
import math
import numpy as np
from itertools import product
import scipy.optimize as opt

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k."""
    rank = U.shape[1]
    # Build the tensor T of shape (n, m, p)
    T = np.zeros((n, m, p))
    for r in range(rank):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        # Vectorized update using einsum or loops
        # T += np.einsum('ij,jk,ki->ijk', u, v, w)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += u[i, j] * v[j, k] * w[k, i]
    return np.allclose(T, 1.0, atol=tol)

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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
    return U, V, W

def _loss_and_grad(params, n, m, p):
    """Compute loss and gradient for the tensor decomposition."""
    # Unpack params
    # U: (n*m, R), V: (m*p, R), W: (p*n, R)
    # We flatten U, V, W into a single vector.
    # Order: U, then V, then W
    size_U = n * m * params_U_shape[1] # Need shapes passed in or closed over
    # To avoid closure issues in scipy, we define a helper class or use a global context
    # Better: define a function that takes shapes as arguments via a wrapper.
    
    # Let's restructure to use a closure or pass shapes.
    # Since scipy.optimize.minimize expects func(x), we'll use a class or lambda with captured variables.
    pass

def _try_optimize(n, m, p, target_rank, time_limit, seed=42):
    """
    Use scipy.optimize.minimize to find a rank-`target_rank` decomposition.
    """
    rng = np.random.RandomState(seed)
    start = time.time()
    
    # Shapes
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    R = target_rank
    
    def objective(params):
        # Unpack
        U_flat = params[:dim_U * R].reshape(dim_U, R)
        V_flat = params[dim_U * R : dim_U * R + dim_V * R].reshape(dim_V, R)
        W_flat = params[dim_U * R + dim_V * R :].reshape(dim_W, R)
        
        # Compute T
        T = np.zeros((n, m, p))
        for r in range(R):
            u = U_flat[:, r].reshape(n, m)
            v = V_flat[:, r].reshape(m, p)
            w = W_flat[:, r].reshape(p, n)
            # T[i,j,k] += u[i,j] * v[j,k] * w[k,i]
            # Use broadcasting/einsum for speed
            T += np.einsum('ij,jk,ki->ijk', u, v, w)
        
        # Loss
        err = T - 1.0
        loss = np.sum(err * err)
        
        # Gradient
        # dL/dT = 2 * err
        # dT/dU[i*m+j, r] = v[j*p+k, r] * w[k*n+i, r] (summed over k)
        # Actually, for a specific r, U, V, W are independent.
        # dL/dU_flat = 2 * sum_{i,j,k} (T[i,j,k]-1) * V[j*p+k, r] * W[k*n+i, r] for each i,j
        # This can be computed efficiently.
        
        dU = np.zeros_like(U_flat)
        dV = np.zeros_like(V_flat)
        dW = np.zeros_like(W_flat)
        
        # Gradient for U:
        # dU[i*m+j, r] = 2 * sum_k (T[i,j,k]-1) * V[j*p+k, r] * W[k*n+i, r]
        # We can loop over r and use matrix multiplications.
        for r in range(R):
            u = U_flat[:, r].reshape(n, m)
            v = V_flat[:, r].reshape(m, p)
            w = W_flat[:, r].reshape(p, n)
            
            # err is T - 1
            # dU[i,j] = 2 * sum_k err[i,j,k] * v[j,k] * w[k,i]
            # Note: w is indexed by k,i. So w[k,i] is the element at row k, col i.
            # v is indexed by j,k.
            # Let's compute this term by term or with einsum.
            # sum_k err[i,j,k] * v[j,k] * w[k,i]
            # Fix i, j: sum_k err[i,j,k] * (v[j,k] * w[k,i])
            # Let M_ik = v[j,k] * w[k,i] ? No, j is fixed.
            
            # Vectorized approach:
            # For each r:
            # dU_r[i,j] = 2 * sum_k (T[i,j,k]-1) * V_r[j,k] * W_r[k,i]
            # We can compute this as:
            # Let E = T - 1 (shape n,m,p)
            # For fixed r:
            # dU_r = 2 * np.einsum('ijk,jk,ki->ij', E, v, w)
            # dV_r = 2 * np.einsum('ijk,ij,ki->jk', E, u, w)
            # dW_r = 2 * np.einsum('ijk,ij,jk->ki', E, u, v)
            
            dU_r = 2 * np.einsum('ijk,jk,ki->ij', E, v, w)
            dV_r = 2 * np.einsum('ijk,ij,ki->jk', E, u, w)
            dW_r = 2 * np.einsum('ijk,ij,jk->ki', E, u, v)
            
            dU[:, r] = dU_r.flatten()
            dV[:, r] = dV_r.flatten()
            dW[:, r] = dW_r.flatten()
        
        grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
        return loss, grad
    
    # Note: objective needs E (T-1) computed inside.
    # Let's rewrite objective to be self-contained and efficient.
    
    def objective_full(params):
        U_flat = params[:dim_U * R].reshape(dim_U, R)
        V_flat = params[dim_U * R : dim_U * R + dim_V * R].reshape(dim_V, R)
        W_flat = params[dim_U * R + dim_V * R :].reshape(dim_W, R)
        
        T = np.zeros((n, m, p))
        for r in range(R):
            u = U_flat[:, r].reshape(n, m)
            v = V_flat[:, r].reshape(m, p)
            w = W_flat[:, r].reshape(p, n)
            T += np.einsum('ij,jk,ki->ijk', u, v, w)
        
        E = T - 1.0
        loss = np.sum(E * E)
        
        dU = np.zeros_like(U_flat)
        dV = np.zeros_like(V_flat)
        dW = np.zeros_like(W_flat)
        
        for r in range(R):
            u = U_flat[:, r].reshape(n, m)
            v = V_flat[:, r].reshape(m, p)
            w = W_flat[:, r].reshape(p, n)
            
            dU_r = 2 * np.einsum('ijk,jk,ki->ij', E, v, w)
            dV_r = 2 * np.einsum('ijk,ij,ki->jk', E, u, w)
            dW_r = 2 * np.einsum('ijk,ij,jk->ki', E, u, v)
            
            dU[:, r] = dU_r.flatten()
            dV[:, r] = dV_r.flatten()
            dW[:, r] = dW_r.flatten()
        
        grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
        return loss, grad

    # Try a few random initializations
    best_result = None
    best_loss = float('inf')
    
    num_attempts = 5
    for attempt in range(num_attempts):
        if time.time() > start + time_limit - 5:
            break
            
        # Initialize with small random values
        # Scale matters. Schoolbook entries are 1. 
        # Let's try initializing with values around 1/sqrt(R) or similar.
        scale = 1.0 / math.sqrt(R)
        
        U0 = rng.randn(dim_U, R) * scale
        V0 = rng.randn(dim_V, R) * scale
        W0 = rng.randn(dim_W, R) * scale
        
        # Symmetry? Not really.
        
        params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        # Use L-BFGS-B
        try:
            res = opt.minimize(objective_full, params0, method='L-BFGS-B', jac=True, options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-9})
        except Exception as e:
            continue
            
        if res.success or res.fun < best_loss:
            best_loss = res.fun
            if best_loss < 1e-6:
                # Found a good solution
                U_sol = res.x[:dim_U * R].reshape(dim_U, R)
                V_sol = res.x[dim_U * R : dim_U * R + dim_V * R].reshape(dim_V, R)
                W_sol = res.x[dim_U * R + dim_V * R :].reshape(dim_W, R)
                
                # Attempt to snap to half-integers
                # The problem allows integer/half-integer entries.
                # We can try rounding to nearest 0.5
                U_snap = np.round(U_sol * 2) / 2
                V_snap = np.round(V_sol * 2) / 2
                W_snap = np.round(W_sol * 2) / 2
                
                if _verify(U_snap, V_snap, W_snap, n, m, p):
                    return (U_snap, V_snap, W_snap)
                
                # If snapping fails, maybe try integer only?
                U_int = np.round(U_sol)
                V_int = np.round(V_sol)
                W_int = np.round(W_sol)
                if _verify(U_int, V_int, W_int, n, m, p):
                    return (U_int, V_int, W_int)
                    
                # If snapping fails but float is good, store it as a candidate for "polish"
                # But we need exact integer/half-integer.
                # If we can't snap, this rank might be too low or we need better init.
                # Let's store the float solution just in case, but verify will fail if not snapped.
                # Actually, if loss is < 1e-6, it's likely very close to a valid solution.
                # We can try a local search around the snapped values?
                # For now, if snap fails, we continue to next attempt or rank.
                
                # Heuristic: If loss is extremely small, force snap and verify.
                # If that fails, return None for this rank.
                
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # Start with schoolbook
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    best_rank = n * m * p
    
    # Try to find a better decomposition using optimization
    # Theoretical lower bound is likely around 20-25.
    # Let's try from 25 up to 35.
    # If we find one, we return it.
    
    # We prioritize lower ranks.
    for target_rank in range(25, best_rank + 1):
        if time.time() > deadline - 10:
            break
        
        remaining_time = deadline - time.time() - 5
        if remaining_time < 10:
            break
        
        # Use different seeds
        seed = 42 + target_rank
        result = _try_optimize(n, m, p, target_rank, remaining_time, seed=seed)
        
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p):
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                best_rank = target_rank
                break
                
    return best_result

# EVOLVE-BLOCK-END
