import numpy as np
import time
import sys

DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    u_dim = n * m  # 4
    v_dim = m * p  # 8
    w_dim = p * n  # 8
    
    # Build the target tensor T of shape (u_dim, v_dim, w_dim)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((u_dim, v_dim, w_dim), dtype=float)
    for i in range(n):
        for j in range(m):
            u_idx = i * m + j
            for k in range(p):
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
    
    # Try rank 13 first
    if _try_rank(13, T, u_dim, v_dim, w_dim):
        return _build_rank13()
    
    # Fallback to rank 14
    return _build_rank14()

def _try_rank(rank, T, u_dim, v_dim, w_dim):
    """Try to find a rank-r decomposition using optimization."""
    try:
        from scipy.optimize import least_squares
        import random
        
        start_time = time.time()
        time_limit = 60  # seconds, leave margin for safety
        
        # Multiple random initializations
        best_cost = np.inf
        best_solution = None
        
        for attempt in range(10):
            if time.time() - start_time > time_limit:
                break
            
            # Random initialization with normalization
            U = np.random.randn(u_dim, rank)
            V = np.random.randn(v_dim, rank)
            W = np.random.randn(w_dim, rank)
            
            # Normalize columns
            for r in range(rank):
                norm_u = np.linalg.norm(U[:, r])
                if norm_u > 1e-9:
                    U[:, r] /= norm_u
                    V[:, r] *= norm_u
                norm_v = np.linalg.norm(V[:, r])
                if norm_v > 1e-9:
                    V[:, r] /= norm_v
                    W[:, r] *= norm_v
            
            def flatten_params(U, V, W):
                return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
                
            def unflatten_params(x):
                U_size = u_dim * rank
                V_size = v_dim * rank
                U = x[:U_size].reshape(u_dim, rank)
                V = x[U_size:U_size+V_size].reshape(v_dim, rank)
                W = x[U_size+V_size:].reshape(w_dim, rank)
                return U, V, W
                
            def residuals(x):
                U, V, W = unflatten_params(x)
                T_approx = np.einsum('ir,jr,kr->ijk', U, V, W)
                return (T - T_approx).flatten()
                
            x0 = flatten_params(U, V, W)
            
            try:
                res = least_squares(residuals, x0, method='lm', max_nfev=5000)
                cost = res.cost
                if cost < best_cost:
                    best_cost = cost
                    best_solution = res.x
                    if cost < 1e-10:
                        break
            except Exception:
                pass
        
        if best_solution is not None and best_cost < 1e-8:
            U, V, W = unflatten_params(best_solution)
            
            # Round to nearest half-integer
            U_int = np.round(U * 2) / 2
            V_int = np.round(V * 2) / 2
            W_int = np.round(W * 2) / 2
            
            # Verify exactness
            T_approx = np.einsum('ir,jr,kr->ijk', U_int, V_int, W_int)
            if np.allclose(T, T_approx, atol=1e-9):
                # Store the solution
                _rank13_solution = {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
                return True
                
    except ImportError:
        pass
    
    return False

def _build_rank13():
    """Return the cached rank-13 solution."""
    return _rank13_solution

def _build_rank14():
    """Build a rank-14 decomposition using two Strassen blocks."""
    n, m, p = DIMENSIONS
    u_dim = n * m
    v_dim = m * p
    w_dim = p * n
    rank = 14
    U = [[0] * rank for _ in range(u_dim)]
    V = [[0] * rank for _ in range(v_dim)]
    W = [[0] * rank for _ in range(w_dim)]
    
    # Strassen 7 terms for 2x2x2
    strassen_terms = [
        # (u_entries, v_entries, w_entries)
        # u: (i, j, val), v: (j, k, val), w: (k, i, val)
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(0,1,1), (1,1,-1)]),
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)]),
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)]),
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(1,0,1), (0,0,-1)]),
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)]),
    ]
    
    # Block 0: k = 0, 1
    for t in range(7):
        u_entries, v_entries, w_entries = strassen_terms[t]
        col = t
        for (i, j, val) in u_entries:
            U[i*m+j][col] += val
        for (j, k, val) in v_entries:
            V[j*p+k][col] += val
        for (k, i, val) in w_entries:
            W[k*n+i][col] += val
            
    # Block 1: k = 2, 3
    for t in range(7):
        u_entries, v_entries, w_entries = strassen_terms[t]
        col = 7 + t
        for (i, j, val) in u_entries:
            U[i*m+j][col] += val
        for (j, k, val) in v_entries:
            k_new = k + 2
            V[j*p+k_new][col] += val
        for (k, i, val) in w_entries:
            k_new = k + 2
            W[k_new*n+i][col] += val
            
    # Verify the decomposition
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    T = np.zeros((u_dim, v_dim, w_dim), dtype=float)
    for i in range(n):
        for j in range(m):
            u_idx = i * m + j
            for k in range(p):
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
    
    T_approx = np.einsum('ir,jr,kr->ijk', U_np, V_np, W_np)
    if not np.allclose(T, T_approx, atol=1e-9):
        # Fallback to naive rank 16 if verification fails
        return _build_rank16()
        
    return {"U": U, "V": V, "W": W}

def _build_rank16():
    """Naive rank-16 decomposition as ultimate fallback."""
    n, m, p = DIMENSIONS
    u_dim = n * m
    v_dim = m * p
    w_dim = p * n
    rank = 16
    U = [[0] * rank for _ in range(u_dim)]
    V = [[0] * rank for _ in range(v_dim)]
    W = [[0] * rank for _ in range(w_dim)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * m * p + j * p + k
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
