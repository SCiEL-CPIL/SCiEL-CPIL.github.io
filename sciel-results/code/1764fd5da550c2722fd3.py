"""Matrix multiplication 2x3 by 3x4: find rank-minimal exact tensor decomposition."""
import numpy as np
from fractions import Fraction
import itertools
import time

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    best_result = None
    start_time = time.time()
    
    # Try ranks from 19 to 23
    for rank in range(19, 24):
        if time.time() - start_time > 140:
            break
        try:
            result = _find_decomposition(n, m, p, rank)
            if result is not None:
                return result
        except Exception:
            continue
            
    # Fall back to schoolbook
    return _schoolbook(n, m, p)

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

def _find_decomposition(n, m, p, rank):
    """Try to find a decomposition of given rank using optimization and local search."""
    num_u = n * m
    num_v = m * p
    num_w = p * n
    total_vars = rank * (num_u + num_v + num_w)
    
    def unpack(params):
        u_flat = params[:rank * num_u]
        v_flat = params[rank * num_u : rank * num_u + rank * num_v]
        w_flat = params[rank * num_u + rank * num_v :]
        
        U = u_flat.reshape(rank, num_u).T # shape (num_u, rank)
        V = v_flat.reshape(rank, num_v).T # shape (num_v, rank)
        W = w_flat.reshape(rank, num_w).T # shape (num_w, rank)
        return U, V, W

    def residuals(params):
        U, V, W = unpack(params)
        res = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i * m + j, :]
                    v_val = V[j * p + k, :]
                    w_val = W[k * n + i, :]
                    s = np.dot(u_val, v_val * w_val)
                    res.append(s - 1.0)
        return np.array(res)

    # Optimization loop
    budget = 60 if rank == 19 else 30 if rank == 20 else 20 if rank == 21 else 10
    deadline = time.time() + budget
    
    np.random.seed(42)
    
    # Try multiple restarts
    for restart in range(100):
        if time.time() > deadline:
            break
            
        # Random initialization
        # Scale initialization to help convergence
        x0 = np.random.randn(total_vars) * 0.5
        
        try:
            result = least_squares(residuals, x0, method='trf', max_nfev=500)
        except Exception:
            continue
            
        if np.linalg.norm(result.fun) < 1e-8:
            # Check if we can snap to half-integers
            U, V, W = unpack(result.x)
            
            # Snap to nearest 0.5
            U_snapped = np.round(U * 2) / 2
            V_snapped = np.round(V * 2) / 2
            W_snapped = np.round(W * 2) / 2
            
            # Verify exactness with snapped values
            if _verify(n, m, p, U_snapped, V_snapped, W_snapped):
                return {
                    "U": U_snapped.tolist(),
                    "V": V_snapped.tolist(),
                    "W": W_snapped.tolist()
                }
            
            # If snapping failed, try local search around snapped values
            U_ls, V_ls, W_ls = _local_search(n, m, p, U_snapped, V_snapped, W_snapped, time_limit=5)
            if U_ls is not None:
                if _verify(n, m, p, U_ls, V_ls, W_ls):
                    return {
                        "U": U_ls.tolist(),
                        "V": V_ls.tolist(),
                        "W": W_ls.tolist()
                    }

    return None

def _local_search(n, m, p, U, V, W, time_limit=5):
    """Try to improve U, V, W (half-integers) to exact solution."""
    # Convert to integer representation (x * 2)
    U_int = (U * 2).astype(int)
    V_int = (V * 2).astype(int)
    W_int = (W * 2).astype(int)
    
    def residual_sum(U_f, V_f, W_f):
        res = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(U_f.shape[1]):
                        s += U_f[i*m+j, r] * V_f[j*p+k, r] * W_f[k*n+i, r]
                    res += (s - 1.0)**2
        return res

    current_cost = residual_sum(U, V, W)
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        # Pick a random matrix and a random element
        mat_choice = np.random.choice([0, 1, 2])
        if mat_choice == 0:
            idx = np.random.randint(U_int.size)
            r, c = np.unravel_index(idx, U_int.shape)
            old_val = U_int[r, c]
            # Try changing to neighbor
            new_val = old_val + np.random.choice([-1, 1])
            # Keep within reasonable bounds
            if -20 <= new_val <= 20:
                U_int[r, c] = new_val
                new_cost = residual_sum(U_int/2.0, V_int/2.0, W_int/2.0)
                if new_cost < current_cost:
                    current_cost = new_cost
                else:
                    U_int[r, c] = old_val
        elif mat_choice == 1:
            idx = np.random.randint(V_int.size)
            r, c = np.unravel_index(idx, V_int.shape)
            old_val = V_int[r, c]
            new_val = old_val + np.random.choice([-1, 1])
            if -20 <= new_val <= 20:
                V_int[r, c] = new_val
                new_cost = residual_sum(U_int/2.0, V_int/2.0, W_int/2.0)
                if new_cost < current_cost:
                    current_cost = new_cost
                else:
                    V_int[r, c] = old_val
        else:
            idx = np.random.randint(W_int.size)
            r, c = np.unravel_index(idx, W_int.shape)
            old_val = W_int[r, c]
            new_val = old_val + np.random.choice([-1, 1])
            if -20 <= new_val <= 20:
                W_int[r, c] = new_val
                new_cost = residual_sum(U_int/2.0, V_int/2.0, W_int/2.0)
                if new_cost < current_cost:
                    current_cost = new_cost
                else:
                    W_int[r, c] = old_val
                    
        if current_cost < 1e-12:
            return U_int/2.0, V_int/2.0, W_int/2.0
            
    return None, None, None

def _verify(n, m, p, U, V, W):
    """Verify that the decomposition is exact."""
    if not isinstance(U, np.ndarray):
        U = np.array(U)
    if not isinstance(V, np.ndarray):
        V = np.array(V)
    if not isinstance(W, np.ndarray):
        W = np.array(W)
        
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-10:
                    return False
    return True

# EVOLVE-BLOCK-END
