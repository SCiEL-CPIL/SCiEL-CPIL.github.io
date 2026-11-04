import time
import random
import numpy as np
from fractions import Fraction
from scipy.optimize import minimize

DIMENSIONS = [4, 4, 5]
n, m, p = DIMENSIONS
NM, MP, PN = n * m, m * p, p * n  # 16, 20, 20

def _verify(U, V, W, tol=1e-6):
    """
    Verify that U, V, W exactly decompose the all-ones multiplication tensor.
    U, V, W are lists of lists.
    """
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    r = U_np.shape[1]
    
    tensor = np.zeros((n, m, p), dtype=np.float64)
    for c in range(r):
        u_c = U_np[:, c]
        v_c = V_np[:, c]
        w_c = W_np[:, c]
        
        # Efficient tensor update: tensor[i, j, k] += u[i,j] * v[j,k] * w[k,i]
        # u_c is shape (n*m,), v_c is (m*p,), w_c is (p*n,)
        # We can use broadcasting or loops. For 16x20x20, loops are fine.
        
        for i in range(n):
            for j in range(m):
                u_val = u_c[i * m + j]
                if u_val == 0:
                    continue
                for k in range(p):
                    v_val = v_c[j * p + k]
                    w_val = w_c[k * n + i]
                    tensor[i, j, k] += u_val * v_val * w_val
    
    return np.allclose(tensor, np.ones((n, m, p)), atol=tol)

def _verify_exact(U, V, W):
    """
    Verify exact decomposition using rational arithmetic.
    """
    r = len(U[0])
    tensor = np.zeros((n, m, p), dtype=object)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                tensor[i, j, k] = Fraction(0)
    
    for c in range(r):
        for i in range(n):
            for j in range(m):
                u_val = Fraction(str(U[i * m + j][c]))
                if u_val == 0:
                    continue
                for k in range(p):
                    v_val = Fraction(str(V[j * p + k][c]))
                    w_val = Fraction(str(W[k * n + i][c]))
                    tensor[i, j, k] += u_val * v_val * w_val
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if tensor[i, j, k] != 1:
                    return False
    return True

def _build_schoolbook():
    """
    Build the schoolbook (rank 80) decomposition.
    """
    rank = n * m * p
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _objective(params, r):
    """
    Objective function for L-BFGS-B.
    params is a flat array of length 3 * r * (NM + MP + PN)
    Split into U (NM * r), V (MP * r), W (PN * r)
    """
    # Reshape
    u_flat = params[:NM * r].reshape((NM, r))
    v_flat = params[NM * r:NM * r + MP * r].reshape((MP, r))
    w_flat = params[NM * r + MP * r:].reshape((PN, r))
    
    # Compute tensor
    tensor = np.zeros((n, m, p), dtype=np.float64)
    for c in range(r):
        u_c = u_flat[:, c]
        v_c = v_flat[:, c]
        w_c = w_flat[:, c]
        
        for i in range(n):
            for j in range(m):
                u_val = u_c[i * m + j]
                if u_val == 0:
                    continue
                for k in range(p):
                    v_val = v_c[j * p + k]
                    w_val = w_c[k * n + i]
                    tensor[i, j, k] += u_val * v_val * w_val
    
    # Error
    error = tensor - np.ones((n, m, p))
    return np.sum(error * error)

def _gradient(params, r):
    """
    Gradient of the objective function.
    """
    u_flat = params[:NM * r].reshape((NM, r))
    v_flat = params[NM * r:NM * r + MP * r].reshape((MP, r))
    w_flat = params[NM * r + MP * r:].reshape((PN, r))
    
    tensor = np.zeros((n, m, p), dtype=np.float64)
    for c in range(r):
        u_c = u_flat[:, c]
        v_c = v_flat[:, c]
        w_c = w_flat[:, c]
        
        for i in range(n):
            for j in range(m):
                u_val = u_c[i * m + j]
                if u_val == 0:
                    continue
                for k in range(p):
                    v_val = v_c[j * p + k]
                    w_val = w_c[k * n + i]
                    tensor[i, j, k] += u_val * v_val * w_val
    
    error = tensor - np.ones((n, m, p))
    
    # Gradient w.r.t. U, V, W
    grad_u = np.zeros_like(u_flat)
    grad_v = np.zeros_like(v_flat)
    grad_w = np.zeros_like(w_flat)
    
    for c in range(r):
        u_c = u_flat[:, c]
        v_c = v_flat[:, c]
        w_c = w_flat[:, c]
        
        for i in range(n):
            for j in range(m):
                u_val = u_c[i * m + j]
                if u_val == 0:
                    continue
                for k in range(p):
                    v_val = v_c[j * p + k]
                    w_val = w_c[k * n + i]
                    err = error[i, j, k]
                    
                    # d(err^2)/d(u) = 2 * err * (v * w)
                    grad_u[i * m + j, c] += 2 * err * v_val * w_val
                    # d(err^2)/d(v) = 2 * err * (u * w)
                    grad_v[j * p + k, c] += 2 * err * u_val * w_val
                    # d(err^2)/d(w) = 2 * err * (u * v)
                    grad_w[k * n + i, c] += 2 * err * u_val * v_val
                    
    grad = np.concatenate([grad_u.flatten(), grad_v.flatten(), grad_w.flatten()])
    return grad

def _round_to_half_integer(x):
    """
    Round a float to the nearest half-integer (0, 0.5, 1, 1.5, etc.)
    """
    # Multiply by 2, round, divide by 2
    return np.round(x * 2) / 2

def _try_optimization(r, start_time, deadline, seed=0):
    """
    Try to find a decomposition of rank r using optimization.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Initial guess
    # Random initialization in [-1, 1]
    params = np.random.uniform(-1, 1, size=3 * r * (NM + MP + PN))
    
    # Optimization options
    options = {
        'maxiter': 50, # Keep low to allow multiple restarts
        'ftol': 1e-12,
        'gtol': 1e-8,
        'disp': False
    }
    
    best_obj = float('inf')
    best_params = None
    
    # Try a few random restarts
    num_restarts = 10
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Re-initialize for each restart
        params = np.random.uniform(-1, 1, size=3 * r * (NM + MP + PN))
        
        try:
            res = minimize(_objective, params, jac=_gradient, method='L-BFGS-B', 
                           args=(r,), options=options)
            
            if res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x
        except Exception:
            continue
        
        if best_obj < 1e-6:
            break
            
    if best_params is None or best_obj > 1e-4:
        return None
        
    # Extract and round
    u_flat = best_params[:NM * r].reshape((NM, r))
    v_flat = best_params[NM * r:NM * r + MP * r].reshape((MP, r))
    w_flat = best_params[NM * r + MP * r:].reshape((PN, r))
    
    # Round to half-integers
    u_rounded = _round_to_half_integer(u_flat)
    v_rounded = _round_to_half_integer(v_flat)
    w_rounded = _round_to_half_integer(w_flat)
    
    # Convert to list of lists
    U = [[float(val) for val in row] for row in u_rounded.T] # Note: U is (NM, r), we need (NM, r) list of lists where U[i][c]
    # Wait, U in the interface is (NM, rank).
    # u_flat is (NM, r).
    # U[i][c] = u_flat[i, c]
    U = [[u_rounded[i, c] for c in range(r)] for i in range(NM)]
    V = [[v_rounded[i, c] for c in range(r)] for i in range(MP)]
    W = [[w_rounded[i, c] for c in range(r)] for i in range(PN)]
    
    # Verify
    if _verify(U, V, W, tol=1e-6):
        if _verify_exact(U, V, W):
            return {"U": U, "V": V, "W": W}
            
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Start with schoolbook as guaranteed valid solution
    U_sb, V_sb, W_sb = _build_schoolbook()
    best_rank = n * m * p  # 80
    best_solution = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Try to find lower ranks using optimization
    # We know Strassen gives 38. Let's try to go lower.
    # The rank of 4x4*4x5 is likely around 25-30.
    # Let's try ranks from 37 down to 20.
    
    # To save time, we can skip ranges or use a heuristic.
    # Let's try a few key ranks.
    ranks_to_try = [37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25]
    
    for r in ranks_to_try:
        if time.time() > deadline - 10:
            break
            
        if r >= best_rank:
            continue
            
        # Try optimization
        solution = _try_optimization(r, start_time, deadline, seed=random.randint(0, 10000))
        
        if solution is not None:
            best_rank = r
            best_solution = solution
            # If we found a solution, we can try to go lower immediately
            # The loop structure handles this by continuing to next r
            
    return best_solution
