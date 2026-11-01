"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]
import time
import itertools
import random
import numpy as np
import scipy.optimize as opt

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
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

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Vectorized verification."""
    r = U.shape[1]
    T = np.einsum('ar,br,cr->abc', U, V, W)
    T_exp = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_exp[i*m+j, j*p+k, k*n+i] = 1.0
    return np.allclose(T, T_exp, atol=tol, rtol=0)

def _tensor_error(params, rank, n, m, p, T_exp):
    """
    Compute the squared error of the tensor reconstruction.
    params: flattened array of U, V, W
    """
    dim_a, dim_b, dim_c = n*m, m*p, p*n
    size_u = dim_a * rank
    size_v = dim_b * rank
    
    U = params[:size_u].reshape(dim_a, rank)
    V = params[size_u:size_u+size_v].reshape(dim_b, rank)
    W = params[size_u+size_v:].reshape(dim_c, rank)
    
    T_approx = np.einsum('ar,br,cr->abc', U, V, W)
    error = np.sum((T_approx - T_exp)**2)
    return error

def _try_rank_optimization(target_rank, n, m, p, deadline, max_attempts=20):
    """
    Attempt to find a decomposition of given rank using scipy optimization.
    """
    dim_a, dim_b, dim_c = n*m, m*p, p*n
    T_exp = np.zeros((dim_a, dim_b, dim_c))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_exp[i*m+j, j*p+k, k*n+i] = 1.0
                
    U0, V0, W0 = _schoolbook(n, m, p)
    
    # We need to reduce rank from 27 to target_rank.
    # Strategy: Start with schoolbook, drop some terms, and optimize the remaining ones
    # to compensate. Or, start with random vectors.
    
    # Better Strategy: Start with schoolbook, but allow all entries to change,
    # and force 4 "columns" to be linear combinations of others?
    # Simpler: Random restarts with structure.
    
    rng = random.Random(42)
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
            
        # Initialize U, V, W
        # We initialize with the schoolbook algorithm, but we only keep 'target_rank' columns.
        # We randomly select which columns to keep, or we merge columns.
        # Merging is hard. Let's try random initialization with some structure.
        
        # Idea: Initialize with 27 terms, then randomly set 4 terms to 0 and optimize the rest?
        # No, the rank is the number of terms.
        
        # Let's try initializing U, V, W with random values, but scaled to match the magnitude.
        U = np.random.randn(dim_a, target_rank) * 0.5
        V = np.random.randn(dim_b, target_rank) * 0.5
        W = np.random.randn(dim_c, target_rank) * 0.5
        
        # Alternatively, initialize from schoolbook by projecting?
        # Let's try a mix: Start with schoolbook, remove 4 random columns, and optimize the rest.
        # This is likely to fail because the remaining columns can't cover the dropped terms easily.
        
        # Let's stick to random restarts but with a smarter initialization:
        # Use the first 'target_rank' columns of schoolbook as a base, and perturb them.
        
        if attempt < 5:
            # Use schoolbook subset
            U_init = U0[:, :target_rank].copy()
            V_init = V0[:, :target_rank].copy()
            W_init = W0[:, :target_rank].copy()
            # Perturb
            U_init += np.random.randn(dim_a, target_rank) * 0.1
            V_init += np.random.randn(dim_b, target_rank) * 0.1
            W_init += np.random.randn(dim_c, target_rank) * 0.1
        else:
            # Random
            U_init = np.random.randn(dim_a, target_rank)
            V_init = np.random.randn(dim_b, target_rank)
            W_init = np.random.randn(dim_c, target_rank)
            # Scale to approximate magnitude
            # The sum of outer products should be ~1.
            # If entries are ~1/sqrt(rank), then product is ~1/rank^(3/2).
            # We want sum of 23 terms to be 1.
            # So each term should be ~1/23.
            # Entries ~ sqrt(1/23) ~ 0.2.
            scale = 0.5
            U_init *= scale
            V_init *= scale
            W_init *= scale

        params0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        # Optimization
        # Use Nelder-Mead for small dimensions?
        # Total params: 9*23 + 9*23 + 9*23 = 621. Nelder-Mead is slow for 621 dims.
        # Use L-BFGS-B.
        
        # Gradient is expensive to compute analytically, but we can use numerical or just L-BFGS-B with approx.
        # Actually, let's use `opt.minimize` with `method='L-BFGS-B'`.
        
        # To speed up, we can use a lower precision check during optimization.
        
        def obj(x):
            return _tensor_error(x, target_rank, n, m, p, T_exp)
        
        # We don't have a good gradient function easily available without autodiff, 
        # but L-BFGS-B can approximate it or we can provide it.
        # Let's try providing a gradient.
        
        def grad(x):
            # Compute gradient of error = sum (T_approx - T_exp)^2
            # dE/dU[a,r] = 2 * sum_{b,c} (T_approx[a,b,c] - T_exp[a,b,c]) * V[b,r] * W[c,r]
            # This is efficient with einsum.
            U = x[:dim_a*target_rank].reshape(dim_a, target_rank)
            V = x[dim_a*target_rank:dim_a*target_rank+dim_b*target_rank].reshape(dim_b, target_rank)
            W = x[dim_a*target_rank+dim_b*target_rank:].reshape(dim_c, target_rank)
            
            T_approx = np.einsum('ar,br,cr->abc', U, V, W)
            diff = T_approx - T_exp
            
            # Gradient for U
            # dE/dU[a,r] = 2 * sum_{b,c} diff[a,b,c] V[b,r] W[c,r]
            # = 2 * einsum('abc,br,cr->ar', diff, V, W)
            gU = 2 * np.einsum('abc,br,cr->ar', diff, V, W)
            
            # Gradient for V
            # dE/dV[b,r] = 2 * sum_{a,c} diff[a,b,c] U[a,r] W[c,r]
            gV = 2 * np.einsum('abc,ar,cr->br', diff, U, W)
            
            # Gradient for W
            # dE/dW[c,r] = 2 * sum_{a,b} diff[a,b,c] U[a,r] V[b,r]
            gW = 2 * np.einsum('abc,ar,br->cr', diff, U, V)
            
            return np.concatenate([gU.flatten(), gV.flatten(), gW.flatten()])
        
        result = opt.minimize(obj, params0, method='L-BFGS-B', jac=grad, options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if result.success or result.fun < 1e-8:
            params_opt = result.x
            U_opt = params_opt[:dim_a*target_rank].reshape(dim_a, target_rank)
            V_opt = params_opt[dim_a*target_rank:dim_a*target_rank+dim_b*target_rank].reshape(dim_b, target_rank)
            W_opt = params_opt[dim_a*target_rank+dim_b*target_rank:].reshape(dim_c, target_rank)
            
            if _verify_fast(U_opt, V_opt, W_opt, n, m, p, tol=1e-7):
                return U_opt, V_opt, W_opt
                
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    # Try rank 23
    if time.time() < deadline:
        result = _try_rank_optimization(23, n, m, p, deadline, max_attempts=10)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Fall back to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
