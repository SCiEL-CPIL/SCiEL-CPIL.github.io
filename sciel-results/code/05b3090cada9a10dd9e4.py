import numpy as np
import time
import sys
from scipy.optimize import minimize

DIMENSIONS = [2, 5, 5]

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _build_target_tensor():
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _snap_and_verify(U, V, W, T_target, tol=1e-9):
    """
    Snaps entries to nearest half-integers and verifies exact decomposition.
    Returns (U_snap, V_snap, W_snap) if valid, else None.
    """
    # Snap to nearest 0.5
    U_snap = np.round(U * 2) / 2
    V_snap = np.round(V * 2) / 2
    W_snap = np.round(W * 2) / 2
    
    # Verify
    T_recon = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap)
    if np.allclose(T_recon, T_target, atol=tol):
        return U_snap, V_snap, W_snap
    return None

def _try_rank(R, T_target, time_limit=10.0, num_restarts=5):
    """
    Attempts to find a decomposition of rank R.
    """
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    start_time = time.time()
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Random initialization
        # Scale to avoid large gradients initially
        U = np.random.randn(dim_u, R) * 0.5
        V = np.random.randn(dim_v, R) * 0.5
        W = np.random.randn(dim_w, R) * 0.5
        
        # Flatten parameters for scipy
        def flatten(U, V, W):
            return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def unflatten(x):
            u_size = dim_u * R
            v_size = dim_v * R
            U = x[:u_size].reshape(dim_u, R)
            V = x[u_size:u_size+v_size].reshape(dim_v, R)
            W = x[u_size+v_size:].reshape(dim_w, R)
            return U, V, W
            
        def objective(x):
            U, V, W = unflatten(x)
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            return np.sum((T_recon - T_target) ** 2)
            
        x0 = flatten(U, V, W)
        
        # Use L-BFGS-B
        try:
            res = minimize(objective, x0, method='L-BFGS-B', 
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12})
            
            if res.success or res.fun < 1e-10:
                U, V, W = unflatten(res.x)
                snapped = _snap_and_verify(U, V, W, T_target)
                if snapped is not None:
                    return snapped
                    
        except Exception as e:
            pass
            
    return None

def construct():
    n, m, p = DIMENSIONS
    T_target = _build_target_tensor()
    
    # Get schoolbook solution as fallback
    U_fb, V_fb, W_fb = _schoolbook()
    best_rank = n * m * p
    best_solution = (U_fb, V_fb, W_fb)
    
    # Try to find lower rank decompositions
    # We prioritize higher reduction first? Or sequential?
    # Sequential from 49 down is safer to find the minimal rank if it exists close to 50.
    # However, if rank is 30, checking 49..31 is wasted.
    # Let's check a few key values: 49, 48, 45, 40, 35, 30.
    # But if the rank is 50, we return 50.
    
    # Given the time constraint (120s total), we can afford ~10s per rank check.
    # Let's try ranks 49 down to 45.
    ranks_to_try = [49, 48, 47, 46, 45]
    
    for R in ranks_to_try:
        if time.time() > 100: # Safety margin
            break
        sol = _try_rank(R, T_target, time_limit=15.0, num_restarts=3)
        if sol is not None:
            U, V, W = sol
            best_rank = R
            best_solution = (U, V, W)
            # If we find a solution, we could continue to see if we can go lower,
            # but given the difficulty, finding one is a huge win.
            # We break to ensure we return quickly with a valid low rank if found.
            # If we want the *minimum*, we should continue.
            # But if rank 49 is found, is rank 48 possible?
            # Let's continue searching for lower ranks if time permits.
            
    # If we found a solution, return it.
    # If not, return schoolbook.
    
    U, V, W = best_solution
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    T_target = _build_target_tensor()
    
    # Get schoolbook solution as fallback
    U_fb, V_fb, W_fb = _schoolbook()
    best_rank = n * m * p
    best_solution = (U_fb, V_fb, W_fb)
    
    # Try to find lower rank decompositions
    ranks_to_try = [49, 48, 47, 46, 45]
    
    start_time = time.time()
    for R in ranks_to_try:
        if time.time() - start_time > 100:
            break
        sol = _try_rank(R, T_target, time_limit=15.0, num_restarts=3)
        if sol is not None:
            U, V, W = sol
            best_rank = R
            best_solution = (U, V, W)
            
    U, V, W = best_solution
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
