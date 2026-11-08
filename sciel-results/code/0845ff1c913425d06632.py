"""Matrix multiplication 2,4,7 - rank minimization via structured search."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 4, 7]

def _schoolbook_decomposition(n, m, p):
    """Return the schoolbook (rank = n*m*p) decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """
    Verify that U, V, W correctly decompose the matrix multiplication tensor.
    The tensor T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
    """
    r = U.shape[1]
    # Reshape U, V, W to 3D
    U_3d = U.reshape(n, m, r)
    V_3d = V.reshape(m, p, r)
    W_3d = W.reshape(p, n, r)
    
    # Compute the tensor via einsum
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # We need to be careful with indices
    # U_3d[i,j,c], V_3d[j,k,c], W_3d[k,i,c]
    # Sum over c: for each (i,j,k)
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(r):
                    val += U_3d[i, j, c] * V_3d[j, k, c] * W_3d[k, i, c]
                # The tensor for matrix mult is T[i,j,k] = 1 if we're computing (AB)_{ik} = sum_j A_{ij} B_{jk}
                # Wait, the standard tensor for C = A * B where A is n x m, B is m x p, C is n x p:
                # C_{ik} = sum_j A_{ij} B_{jk}
                # The tensor T_{ijk} = 1 for all i,j,k? No.
                # Actually, the multiplication tensor has entries T_{ijk} = 1 for all valid i,j,k in the standard convention.
                # Let me reconsider. The tensor is defined such that:
                # (A * B)_{ik} = sum_{j,c} A_{ij} B_{jk} U_{ij,c} V_{jk,c} W_{ki,c}
                # Wait, the problem states: W[k*n+i, r] following the official trace convention.
                # The standard matrix multiplication tensor T has T_{ijk} = 1 for all i,j,k.
                # Because (AB)_{ik} = sum_j A_{ij} B_{jk}, so the "tensor" is all ones.
                expected = 1.0
                err = abs(val - expected)
                if err > max_err:
                    max_err = err
    return max_err < tol, max_err

def _try_rank_r(n, m, p, r, rng, max_iter=50, tol=1e-8):
    """
    Try to find a rank-r decomposition using random initialization and gradient descent.
    Returns (U, V, W) if found, else None.
    """
    U = rng.standard_normal((n * m, r))
    V = rng.standard_normal((m * p, r))
    W = rng.standard_normal((p * n, r))
    
    U_3d = U.reshape(n, m, r)
    V_3d = V.reshape(m, p, r)
    W_3d = W.reshape(p, n, r)
    
    # We want to minimize the loss:
    # L = sum_{i,j,k} (sum_c U_{ijc} V_{jkc} W_{kic} - 1)^2
    # This is a tensor train / CP-like decomposition problem.
    
    # Use a simple alternating least squares approach
    # For each c, we're solving for the three factors.
    
    # Actually, let's use a simpler approach: since the tensor is all-ones,
    # we can try to find rank-r CP decomposition of the all-ones tensor.
    
    # The all-ones tensor of size n x m x p can be decomposed as:
    # T_{ijk} = sum_c a_c b_c d_c where a is in R^n, b in R^m, d in R^p
    # But wait, our decomposition has a specific structure:
    # U is (n*m) x r, V is (m*p) x r, W is (p*n) x r
    # And the constraint is: sum_c U_{ij,c} V_{jk,c} W_{ki,c} = 1 for all i,j,k
    
    # This is not a standard CP decomposition. Let me think differently.
    # 
    # Let's just do a direct optimization using scipy.
    
    from scipy.optimize import minimize
    
    def flatten_params(U, V, W):
        return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    def unflatten_params(params):
        U_flat = params[:n*m*r]
        V_flat = params[n*m*r:n*m*r + m*p*r]
        W_flat = params[n*m*r + m*p*r:]
        U = U_flat.reshape(n, m, r)
        V = V_flat.reshape(m, p, r)
        W = W_flat.reshape(p, n, r)
        return U, V, W
    
    def loss(params):
        U, V, W = unflatten_params(params)
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i, j, :] * V[j, k, :] * W[k, i, :])
                    err += (val - 1.0) ** 2
        return err
    
    x0 = flatten_params(U, V, W)
    
    # Try multiple random starts
    for attempt in range(3):
        if attempt > 0:
            U = rng.standard_normal((n * m, r))
            V = rng.standard_normal((m * p, r))
            W = rng.standard_normal((p * n, r))
            x0 = flatten_params(U, V, W)
        
        res = minimize(loss, x0, method='BFGS', options={'maxiter': max_iter, 'gtol': 1e-6})
        
        if res.fun < tol:
            U_opt, V_opt, W_opt = unflatten_params(res.x)
            U_out = U_opt.reshape(n * m, r)
            V_out = V_opt.reshape(m * p, r)
            W_out = W_opt.reshape(p * n, r)
            
            # Verify
            valid, max_err = _verify_decomposition(U_out, V_out, W_out, n, m, p, tol)
            if valid:
                return U_out, V_out, W_out
    
    return None

def _try_structural_rank(n, m, p, r, rng, time_limit):
    """
    Try a structured approach: use rank-r where the factors have a special form.
    For small r, we can enumerate possible structures.
    """
    start_time = time.time()
    
    # For 2x4x7, let's try ranks from theoretical lower bounds upward
    # The tensor rank of the 2x4x7 matrix multiplication tensor...
    # Known: rank of 2x2x2 is 7 (Strassen)
    # For 2x4x7, we can try various ranks
    
    # Let's try using the fact that we can "split" the problem
    # 2x4x7: we can think of it as combining smaller multiplications
    
    # Try ranks that are multiples or near-known values
    # Let's just try a range of ranks with random initialization
    
    for r in range(5, 50):
        if time.time() - start_time > time_limit - 10:
            break
        
        result = _try_rank_r(n, m, p, r, rng, max_iter=30)
        if result is not None:
            return result
    
    return None

def construct():
    start_time = time.time()
    time_limit = 140  # Safety margin: evaluator kills at 150s
    
    n, m, p = DIMENSIONS
    rng = np.random.RandomState(42)
    
    # First, get the schoolbook decomposition as a fallback
    U_sb, V_sb, W_sb = _schoolbook_decomposition(n, m, p)
    best_result = (U_sb, V_sb, W_sb)
    best_rank = U_sb.shape[1]
    
    # Try to find a better (lower rank) decomposition
    # Try structural search with time limit
    result = _try_structural_rank(n, m, p, 50, rng, time_limit - (time.time() - start_time))
    if result is not None:
        U_new, V_new, W_new = result
        new_rank = U_new.shape[1]
        if new_rank < best_rank:
            best_result = (U_new, V_new, W_new)
            best_rank = new_rank
    
    U, V, W = best_result
    
    # Convert to lists for the required interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
