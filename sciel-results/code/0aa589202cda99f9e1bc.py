import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 6]

def _get_strassen_21():
    # Returns U, V, W for rank 21
    n, m, p = DIMENSIONS
    rank = 21
    U = np.zeros((4, rank))
    V = np.zeros((12, rank))
    W = np.zeros((12, rank))
    
    strassen_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {(0, 0): 1, (1, 1): 1}),
        ({2: 1, 3: 1}, {0: 1}, {(1, 0): 1, (1, 1): -1}),
        ({0: 1}, {1: 1, 3: -1}, {(0, 1): 1, (1, 1): 1}),
        ({3: 1}, {2: 1, 0: -1}, {(0, 0): 1, (1, 0): 1}),
        ({0: 1, 1: 1}, {3: 1}, {(0, 0): -1, (0, 1): 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {(1, 1): 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {(0, 0): 1})
    ]
    
    r = 0
    for b in range(3):
        for t in range(7):
            u_coeffs, v_coeffs, w_coeffs = strassen_terms[t]
            for idx, val in u_coeffs.items():
                U[idx, r] = val
            for l, val in v_coeffs.items():
                row = l // 2
                col = 2 * b + (l % 2)
                v_idx = row * p + col
                V[v_idx, r] = val
            for (i, k_local), val in w_coeffs.items():
                k_global = 2 * b + k_local
                w_idx = k_global * n + i
                W[w_idx, r] = val
            r += 1
    return U, V, W

def _construct_target_tensor():
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _try_rank_r(rank, T, U21, V21, W21, time_limit=50):
    """
    Attempts to find a decomposition of the given rank.
    Returns (U, V, W) if successful, else None.
    """
    if rank >= 21:
        # We already have 21
        if rank == 21:
            return U21, V21, W21
        return None

    # Variables: U (4 x rank), V (12 x rank), W (12 x rank)
    n_u = 4
    n_v = 12
    n_w = 12
    
    # Initialize from rank 21 (drop columns)
    # This might be a bad init if rank 21 is minimal, but let's try.
    U_init = U21[:, :rank]
    V_init = V21[:, :rank]
    W_init = W21[:, :rank]
    
    x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
    
    # Also try a random init?
    # Random init in high dim is hard.
    
    def loss_and_grad(x):
        U = x[:n_u*rank].reshape(n_u, rank)
        V = x[n_u*rank:n_u*rank+n_v*rank].reshape(n_v, rank)
        W = x[n_u*rank+n_v*rank:].reshape(n_w, rank)
        
        # R = sum_r U_r V_r W_r
        # Use einsum for speed
        R = np.einsum('ar,br,cr->abc', U, V, W)
        E = T - R
        loss = np.sum(E**2)
        
        # Gradients
        # dL/dU_{ar} = -2 * sum_{b,c} E_{abc} V_{br} W_{cr}
        # dL/dV_{br} = -2 * sum_{a,c} E_{abc} U_{ar} W_{cr}
        # dL/dW_{cr} = -2 * sum_{a,b} E_{abc} U_{ar} V_{br}
        
        # Efficient computation:
        # grad_U = -2 * np.einsum('abc,br,cr->ar', E, V, W)
        # grad_V = -2 * np.einsum('abc,ar,cr->br', E, U, W)
        # grad_W = -2 * np.einsum('abc,ar,br->cr', E, U, V)
        
        grad_U = -2.0 * np.einsum('abc,br,cr->ar', E, V, W)
        grad_V = -2.0 * np.einsum('abc,ar,cr->br', E, U, W)
        grad_W = -2.0 * np.einsum('abc,ar,br->cr', E, U, V)
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        
        return loss, grad

    start_time = time.time()
    
    # Use L-BFGS-B
    # maxiter should be limited by time
    # We can wrap the callback to check time
    state = {'stop': False}
    
    def callback(xk):
        if time.time() - start_time > time_limit:
            state['stop'] = True
            return True
        return False

    try:
        res = minimize(loss_and_grad, x0, jac=True, method='L-BFGS-B', 
                       options={'maxiter': 10000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 20},
                       callback=callback)
        
        if state['stop']:
             return None

        final_loss = res.fun
        if final_loss < 1e-8:
            # Extract solution
            U_sol = res.x[:n_u*rank].reshape(n_u, rank)
            V_sol = res.x[n_u*rank:n_u*rank+n_v*rank].reshape(n_v, rank)
            W_sol = res.x[n_u*rank+n_v*rank:].reshape(n_w, rank)
            
            # Verify exactness and snap to half-integers
            # Check if 2*x is close to integer
            U_snap = np.round(U_sol * 2) / 2
            V_snap = np.round(V_sol * 2) / 2
            W_snap = np.round(W_sol * 2) / 2
            
            # Check if snapped solution is still valid
            R_check = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap)
            if np.allclose(R_check, T, atol=1e-10):
                return U_snap, V_snap, W_snap
            else:
                # Maybe the solution is integer?
                U_snap_int = np.round(U_sol)
                V_snap_int = np.round(V_sol)
                W_snap_int = np.round(W_sol)
                R_check_int = np.einsum('ar,br,cr->abc', U_snap_int, V_snap_int, W_snap_int)
                if np.allclose(R_check_int, T, atol=1e-10):
                    return U_snap_int, V_snap_int, W_snap_int
                else:
                    return None
        else:
            return None
    except Exception:
        return None

def construct():
    T = _construct_target_tensor()
    U21, V21, W21 = _get_strassen_21()
    
    # Try rank 20
    # We have 120s total. Use 60s for search.
    result_20 = _try_rank_r(20, T, U21, V21, W21, time_limit=60)
    
    if result_20 is not None:
        U, V, W = result_20
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # Fallback to rank 21
        return {
            "U": U21.tolist(),
            "V": V21.tolist(),
            "W": W21.tolist()
        }
