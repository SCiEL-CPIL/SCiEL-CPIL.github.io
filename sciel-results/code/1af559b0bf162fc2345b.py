import numpy as np
import time
import random
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 5]

def _build_target_tensor():
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
    return T

def _verify(U, V, W, T):
    n, m, p = DIMENSIONS
    # Check half-integer constraint
    if not np.allclose(U * 2, np.round(U * 2), atol=1e-8):
        return False
    if not np.allclose(V * 2, np.round(V * 2), atol=1e-8):
        return False
    if not np.allclose(W * 2, np.round(W * 2), atol=1e-8):
        return False
    
    # Check exact decomposition
    T_calc = np.einsum('ar,br,cr->abc', U, V, W)
    if not np.allclose(T_calc, T, atol=1e-8):
        return False
    return True

def _snap_to_half_integers(U, V, W):
    U_s = np.round(U * 2) / 2
    V_s = np.round(V * 2) / 2
    W_s = np.round(W * 2) / 2
    return U_s, V_s, W_s

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    return U, V, W

def _try_structured_3x3_extend(deadline):
    """
    Try to build a decomposition by extending a 3x3x3 approach.
    For 3x3x3, we can use a rank-19 or lower decomposition.
    For 3x3x5, we can think of it as 3x3x3 + 3x3x2.
    """
    n, m, p = DIMENSIONS
    dim_u = n * m  # 9
    dim_v = m * p  # 15
    dim_w = p * n  # 15
    
    # Strategy: Try various structured initializations
    # Key idea: The tensor has a specific algebraic structure.
    # T[i,j,k] = 1 iff the indices form a valid "chain" i->j->k->i
    
    # Try to use a factorization based on the group structure
    # For 3x3x3, one approach is to use the fact that:
    # (a,b,c) terms can be grouped
    
    # Let's try a direct structured approach:
    # Initialize with a pattern that respects the tensor structure
    
    best_result = None
    
    # Try different structured initializations
    for trial in range(20):
        if time.time() > deadline:
            break
        
        if trial == 0:
            # Pure schoolbook-like but with potential for merging
            U = np.zeros((dim_u, 45))
            V = np.zeros((dim_v, 45))
            W = np.zeros((dim_w, 45))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m+j, col] = 1.0
                        V[j*p+k, col] = 1.0
                        W[k*n+i, col] = 1.0
                        col += 1
            # Try to reduce by optimizing
            target_rank = 40
        elif trial == 1:
            # Try a more structured initialization
            # Based on the idea that we can group terms
            target_rank = 35
            U = np.random.randn(dim_u, target_rank) * 0.3
            V = np.random.randn(dim_v, target_rank) * 0.3
            W = np.random.randn(dim_w, target_rank) * 0.3
        else:
            target_rank = 30 + (trial % 10)
            U = np.random.randn(dim_u, target_rank) * 0.5
            V = np.random.randn(dim_v, target_rank) * 0.5
            W = np.random.randn(dim_w, target_rank) * 0.5
        
        # Try to optimize
        def objective(x):
            u = x[:dim_u*target_rank].reshape(dim_u, target_rank)
            v = x[dim_u*target_rank:dim_u*target_rank+dim_v*target_rank].reshape(dim_v, target_rank)
            w = x[dim_u*target_rank+dim_v*target_rank:].reshape(dim_w, target_rank)
            t_calc = np.einsum('ar,br,cr->abc', u, v, w)
            return (t_calc - _build_target_tensor()).flatten()
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            res = least_squares(objective, x0, method='trf', max_nfev=300, ftol=1e-10)
            if res.cost < 1e-6:
                u_opt = res.x[:dim_u*target_rank].reshape(dim_u, target_rank)
                v_opt = res.x[dim_u*target_rank:dim_u*target_rank+dim_v*target_rank].reshape(dim_v, target_rank)
                w_opt = res.x[dim_u*target_rank+dim_v*target_rank:].reshape(dim_w, target_rank)
                
                u_s, v_s, w_s = _snap_to_half_integers(u_opt, v_opt, w_opt)
                T = _build_target_tensor()
                if _verify(u_s, v_s, w_s, T):
                    best_result = (u_s, v_s, w_s)
                    return best_result
        except Exception:
            pass
    
    return best_result

def _try_linear_dependency_elimination(U, V, W, T, deadline):
    """
    Try to reduce rank by finding and eliminating linearly dependent columns.
    """
    rank = U.shape[1]
    
    # Try to eliminate columns one by one
    for col_to_remove in range(rank):
        if time.time() > deadline:
            break
        
        # Check if column col_to_remove is a linear combination of others
        # This is complex, so try a simpler approach:
        # Remove the column and see if we can adjust the others to compensate
        
        if col_to_remove == 0:
            continue
        
        # Create a version without this column
        mask = [i for i in range(rank) if i != col_to_remove]
        U_reduced = U[:, mask]
        V_reduced = V[:, mask]
        W_reduced = W[:, mask]
        
        # Check if the reduced version still works
        T_calc = np.einsum('ar,br,cr->abc', U_reduced, V_reduced, W_reduced)
        if np.allclose(T_calc, T, atol=1e-8):
            # We can eliminate this column
            return U_reduced, V_reduced, W_reduced
    
    return U, V, W

def _try_rank_optimization(R, T, deadline, num_restarts=5):
    """Try to find a rank-R decomposition."""
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Structured random initialization
        U = np.random.randn(dim_u, R) * 0.5
        V = np.random.randn(dim_v, R) * 0.5
        W = np.random.randn(dim_w, R) * 0.5
        
        # Add some structure: ensure certain entries are non-zero
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Bias towards the tensor structure
                    if random.random() < 0.3:
                        c = random.randint(0, R-1)
                        U[i*m+j, c] += 0.5
                        V[j*p+k, c] += 0.5
                        W[k*n+i, c] += 0.5
        
        def objective(x):
            u = x[:dim_u*R].reshape(dim_u, R)
            v = x[dim_u*R:dim_u*R+dim_v*R].reshape(dim_v, R)
            w = x[dim_u*R+dim_v*R:].reshape(dim_w, R)
            t_calc = np.einsum('ar,br,cr->abc', u, v, w)
            return (t_calc - T).flatten()
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            res = least_squares(objective, x0, method='trf', max_nfev=400, ftol=1e-10)
            if res.cost < 1e-8:
                u_opt = res.x[:dim_u*R].reshape(dim_u, R)
                v_opt = res.x[dim_u*R:dim_u*R+dim_v*R].reshape(dim_v, R)
                w_opt = res.x[dim_u*R+dim_v*R:].reshape(dim_w, R)
                
                u_s, v_s, w_s = _snap_to_half_integers(u_opt, v_opt, w_opt)
                if _verify(u_s, v_s, w_s, T):
                    return u_s, v_s, w_s
        except Exception:
            pass
    
    return None

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    T = _build_target_tensor()
    
    # First, try the structured approach
    result = _try_structured_3x3_extend(deadline)
    if result is not None:
        U, V, W = result
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Try rank optimization from high to low
    # Start from a reasonable upper bound
    for R in range(40, 25, -1):
        if time.time() > deadline:
            break
        result = _try_rank_optimization(R, T, deadline, num_restarts=3)
        if result is not None:
            U, V, W = result
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    # Fallback to schoolbook
    U, V, W = _schoolbook()
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
