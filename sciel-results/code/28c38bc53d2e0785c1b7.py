import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 4, 6]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W form a valid tensor decomposition."""
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
        return False
        
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _snap_to_half(arr):
    """Snap values to nearest half-integer."""
    return np.round(arr * 2) / 2.0

def _snap_to_quarter(arr):
    """Snap values to nearest quarter-integer."""
    return np.round(arr * 4) / 4.0

def _make_schoolbook(n, m, p):
    """Create the standard (schoolbook) tensor decomposition."""
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

def _build_index_arrays(n, m, p):
    """Precompute index arrays for vectorized operations."""
    u_idx = []
    v_idx = []
    w_idx = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx.append(i * m + j)
                v_idx.append(j * p + k)
                w_idx.append(k * n + i)
    return np.array(u_idx), np.array(v_idx), np.array(w_idx)

def _try_snap_and_verify(U, V, W, n, m, p, tol=1e-6):
    """Try various snapping strategies and verify."""
    # Try half-integers
    U_snap = _snap_to_half(U)
    V_snap = _snap_to_half(V)
    W_snap = _snap_to_half(W)
    if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p, tol):
        return U_snap, V_snap, W_snap, True
    
    # Try integers
    U_int = np.round(U)
    V_int = np.round(V)
    W_int = np.round(W)
    if _verify_decomposition(U_int, V_int, W_int, n, m, p, tol):
        return U_int, V_int, W_int, True
    
    # Try quarters
    U_q = _snap_to_quarter(U)
    V_q = _snap_to_quarter(V)
    W_q = _snap_to_quarter(W)
    if _verify_decomposition(U_q, V_q, W_q, n, m, p, tol):
        return U_q, V_q, W_q, True
    
    return None, None, None, False

def _optimize_direct(n, m, p, r, time_limit=20, num_inits=10):
    """
    Direct optimization for rank r decomposition of n×m×p matrix multiplication.
    Returns (U, V, W, success)
    """
    start_time = time.time()
    u_idx, v_idx, w_idx = _build_index_arrays(n, m, p)
    nm, mp, pn = n*m, m*p, p*n
    total_vars = nm*r + mp*r + pn*r
    
    def unpack(x):
        U = x[:nm*r].reshape(nm, r)
        V = x[nm*r : nm*r + mp*r].reshape(mp, r)
        W = x[nm*r + mp*r :].reshape(pn, r)
        return U, V, W

    def objective(x):
        U, V, W = unpack(x)
        U_sel = U[u_idx, :]
        V_sel = V[v_idx, :]
        W_sel = W[w_idx, :]
        vals = np.sum(U_sel * V_sel * W_sel, axis=1)
        err = np.sum((vals - 1.0)**2)
        return err

    def gradient(x):
        U, V, W = unpack(x)
        U_sel = U[u_idx, :]
        V_sel = V[v_idx, :]
        W_sel = W[w_idx, :]
        vals = np.sum(U_sel * V_sel * W_sel, axis=1)
        diff = 2.0 * (vals - 1.0)
        
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        np.add.at(grad_U, (u_idx, np.arange(r)), diff[:, None] * (V_sel * W_sel))
        np.add.at(grad_V, (v_idx, np.arange(r)), diff[:, None] * (U_sel * W_sel))
        np.add.at(grad_W, (w_idx, np.arange(r)), diff[:, None] * (U_sel * V_sel))
        
        return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])

    best_loss = float('inf')
    best_x = None
    
    # Build schoolbook-based initialization
    U_school, V_school, W_school = _make_schoolbook(n, m, p)
    
    for init_idx in range(num_inits):
        if time.time() - start_time > time_limit:
            break
        
        if init_idx == 0:
            # Schoolbook truncated with small perturbation
            U_init = U_school[:, :r].copy()
            V_init = V_school[:, :r].copy()
            W_init = W_school[:, :r].copy()
            x0 = np.concatenate([
                (U_init + 0.01 * np.random.randn(nm, r)).flatten(),
                (V_init + 0.01 * np.random.randn(mp, r)).flatten(),
                (W_init + 0.01 * np.random.randn(pn, r)).flatten()
            ])
        elif init_idx == 1:
            # Schoolbook truncated with larger perturbation
            U_init = U_school[:, :r].copy()
            V_init = V_school[:, :r].copy()
            W_init = W_school[:, :r].copy()
            x0 = np.concatenate([
                (U_init + 0.1 * np.random.randn(nm, r)).flatten(),
                (V_init + 0.1 * np.random.randn(mp, r)).flatten(),
                (W_init + 0.1 * np.random.randn(pn, r)).flatten()
            ])
        elif init_idx == 2:
            # Pure random small
            x0 = np.random.randn(total_vars) * 0.05
        elif init_idx == 3:
            # Sparse random initialization
            U_rand = np.zeros((nm, r))
            V_rand = np.zeros((mp, r))
            W_rand = np.zeros((pn, r))
            for col in range(r):
                U_rand[np.random.randint(nm), col] = np.random.choice([-1, 1])
                V_rand[np.random.randint(mp), col] = np.random.choice([-1, 1])
                W_rand[np.random.randint(pn), col] = np.random.choice([-1, 1])
            x0 = np.concatenate([U_rand.flatten(), V_rand.flatten(), W_rand.flatten()])
        elif init_idx == 4:
            # Structured random: group columns
            U_rand = np.zeros((nm, r))
            V_rand = np.zeros((mp, r))
            W_rand = np.zeros((pn, r))
            for col in range(r):
                i = col % n
                j = (col // n) % m
                k = (col // (n*m)) % p
                U_rand[i*m+j, col] = 1.0
                V_rand[j*p+k, col] = 1.0
                W_rand[k*n+i, col] = 1.0
            # Add some randomness
            U_rand += 0.05 * np.random.randn(nm, r)
            V_rand += 0.05 * np.random.randn(mp, r)
            W_rand += 0.05 * np.random.randn(pn, r)
            x0 = np.concatenate([U_rand.flatten(), V_rand.flatten(), W_rand.flatten()])
        else:
            # Mix of schoolbook and random
            alpha = 0.3 + 0.1 * init_idx
            U_init = U_school[:, :r].copy()
            V_init = V_school[:, :r].copy()
            W_init = W_school[:, :r].copy()
            x0 = np.concatenate([
                (alpha * U_init + (1-alpha) * np.random.randn(nm, r) * 0.1).flatten(),
                (alpha * V_init + (1-alpha) * np.random.randn(mp, r) * 0.1).flatten(),
                (alpha * W_init + (1-alpha) * np.random.randn(pn, r) * 0.1).flatten()
            ])

        try:
            from scipy.optimize import minimize
            res = minimize(objective, x0, jac=gradient, method='L-BFGS-B', 
                           options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-10})
            
            if res.fun < best_loss:
                best_loss = res.fun
                best_x = res.x.copy()
                
                U = best_x[:nm*r].reshape(nm, r)
                V = best_x[nm*r : nm*r + mp*r].reshape(mp, r)
                W = best_x[nm*r + mp*r :].reshape(pn, r)
                
                U_snap, V_snap, W_snap, success = _try_snap_and_verify(U, V, W, n, m, p, tol=1e-6)
                if success:
                    return U_snap, V_snap, W_snap, True
                    
        except Exception:
            pass
            
    # Polish pass
    if best_x is not None and best_loss < 1.0:
        try:
            from scipy.optimize import minimize
            res2 = minimize(objective, best_x, jac=gradient, method='L-BFGS-B', 
                           options={'maxiter': 10000, 'ftol': 1e-15, 'gtol': 1e-12})
            if res2.fun < best_loss:
                best_loss = res2.fun
                best_x = res2.x.copy()
                
                U = best_x[:nm*r].reshape(nm, r)
                V = best_x[nm*r : nm*r + mp*r].reshape(mp, r)
                W = best_x[nm*r + mp*r :].reshape(pn, r)
                
                U_snap, V_snap, W_snap, success = _try_snap_and_verify(U, V, W, n, m, p, tol=1e-5)
                if success:
                    return U_snap, V_snap, W_snap, True
        except Exception:
            pass
            
    if best_x is not None and best_loss < 1e-4:
        U = best_x[:nm*r].reshape(nm, r)
        V = best_x[nm*r : nm*r + mp*r].reshape(mp, r)
        W = best_x[nm*r + mp*r :].reshape(pn, r)
        U_snap, V_snap, W_snap, success = _try_snap_and_verify(U, V, W, n, m, p, tol=1e-4)
        if success:
            return U_snap, V_snap, W_snap, True
            
    return None, None, None, False

def _optimize_rank_343(r, time_limit=20):
    """Optimize for rank r decomposition of 3×4×3 matrix multiplication."""
    return _optimize_direct(3, 4, 3, r, time_limit=time_limit, num_inits=15)

def _construct_from_blocks(U_343, V_343, W_343, n, m, p):
    """
    Construct 3×4×6 solution from two 3×4×3 blocks.
    Block 1: k=0,1,2
    Block 2: k=3,4,5
    """
    r = U_343.shape[1]
    
    U_new = np.zeros((n*m, 2*r))
    V_new = np.zeros((m*p, 2*r))
    W_new = np.zeros((p*n, 2*r))
    
    # Block 1: k=0,1,2
    U_new[:, 0:r] = U_343
    V_new[0:12, 0:r] = V_343
    W_new[0:9, 0:r] = W_343
    
    # Block 2: k=3,4,5
    U_new[:, r:2*r] = U_343
    
    for j in range(m):
        for k_prime in range(3):
            V_new[j*6 + 3 + k_prime, r:2*r] = V_343[j*3 + k_prime, :]
            
    for k_prime in range(3):
        for i in range(n):
            W_new[(3+k_prime)*3 + i, r:2*r] = W_343[k_prime*3 + i, :]
    
    return U_new, V_new, W_new

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Strategy 1: Try direct optimization for 3×4×6 at various ranks
    # Schoolbook rank is 72. Try to find lower.
    target_ranks = [28, 30, 32, 34, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72]
    
    best_solution = None
    best_rank = float('inf')
    
    for r in target_ranks:
        if time.time() > deadline - 20:
            break
        
        remaining = deadline - time.time() - 10
        if remaining < 10:
            break
        
        # Try direct optimization
        U, V, W, success = _optimize_direct(n, m, p, r, time_limit=min(remaining, 25), num_inits=8)
        
        if success:
            best_solution = (U, V, W)
            best_rank = r
            break  # Found a valid solution at this rank, try to go lower next
    
    # Strategy 2: If direct didn't work well, try block-based approach
    if best_solution is None or best_rank > 40:
        # Try to find good 3×4×3 decomposition
        for r_343 in [14, 15, 16, 17, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36]:
            if time.time() > deadline - 15:
                break
            
            remaining = deadline - time.time() - 5
            if remaining < 5:
                break
            
            U_343, V_343, W_343, success = _optimize_rank_343(r_343, time_limit=min(remaining, 20))
            
            if success:
                # Construct 3×4×6 solution
                U_new, V_new, W_new = _construct_from_blocks(U_343, V_343, W_343, n, m, p)
                
                if _verify_decomposition(U_new, V_new, W_new, n, m, p, tol=1e-6):
                    rank_new = U_new.shape[1]
                    if rank_new < best_rank:
                        best_solution = (U_new, V_new, W_new)
                        best_rank = rank_new
                        break
    
    # If we found a solution, return it
    if best_solution is not None:
        U, V, W = best_solution
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    U_school, V_school, W_school = _make_schoolbook(n, m, p)
    U_list = [[float(x) for x in row] for row in U_school]
    V_list = [[float(x) for x in row] for row in V_school]
    W_list = [[float(x) for x in row] for row in W_school]
    return {"U": U_list, "V": V_list, "W": W_list}
