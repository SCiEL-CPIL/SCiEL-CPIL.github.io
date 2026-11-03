import time
import numpy as np
from scipy.optimize import minimize
import random

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n=2, m=3, p=7):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _residuals_fast(U, V, W, n, m, p, rank):
    """Compute residuals using vectorized operations."""
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Residual[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] - 1
    res = np.ones((n, m, p))
    for r in range(rank):
        u_r = U[:, r].reshape(n, m)
        v_r = V[:, r].reshape(m, p)
        w_r = W[:, r].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    res[i, j, k] -= u_r[i, j] * v_r[j, k] * w_r[k, i]
    return res

def _jacobian_fast(U, V, W, n, m, p, rank):
    """Compute Jacobian using vectorized operations. Returns J as (n*m*p, total_vars)."""
    num_u = n * m * rank
    num_v = m * p * rank
    num_w = p * n * rank
    total_vars = num_u + num_v + num_w
    n_eqs = n * m * p
    J = np.zeros((n_eqs, total_vars))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                eq_idx = i*m*p + j*p + k
                u_idx = i*m + j
                v_idx = j*p + k
                w_idx = k*n + i
                for r in range(rank):
                    u_val = U[u_idx, r]
                    v_val = V[v_idx, r]
                    w_val = W[w_idx, r]
                    J[eq_idx, u_idx*rank + r] = v_val * w_val
                    J[eq_idx, num_u + v_idx*rank + r] = u_val * w_val
                    J[eq_idx, num_u + num_v + w_idx*rank + r] = u_val * v_val
    return J

def _newton_search(rank, n, m, p, seed, max_iter=80, tol=1e-10):
    """
    Attempt to find a rank-R decomposition using Newton's method.
    Returns (U, V, W) as numpy arrays if successful, else None.
    """
    rng = np.random.RandomState(seed)
    
    # Multiple initialization strategies
    inits = []
    
    # Strategy 1: Small random
    U1 = rng.uniform(-0.3, 0.3, (n*m, rank))
    V1 = rng.uniform(-0.3, 0.3, (m*p, rank))
    W1 = rng.uniform(-0.3, 0.3, (p*n, rank))
    inits.append((U1, V1, W1))
    
    # Strategy 2: Sparse-like with some structure
    U2 = np.zeros((n*m, rank))
    V2 = np.zeros((m*p, rank))
    W2 = np.zeros((p*n, rank))
    for r in range(rank):
        # Randomly assign non-zero entries
        for _ in range(3):
            ui = rng.randint(0, n*m-1)
            U2[ui, r] = rng.choice([-1, 1]) * rng.uniform(0.5, 1.5)
        for _ in range(3):
            vi = rng.randint(0, m*p-1)
            V2[vi, r] = rng.choice([-1, 1]) * rng.uniform(0.5, 1.5)
        for _ in range(3):
            wi = rng.randint(0, p*n-1)
            W2[wi, r] = rng.choice([-1, 1]) * rng.uniform(0.5, 1.5)
    inits.append((U2, V2, W2))
    
    # Strategy 3: Block-structured
    U3 = np.zeros((n*m, rank))
    V3 = np.zeros((m*p, rank))
    W3 = np.zeros((p*n, rank))
    for r in range(rank):
        i0 = rng.randint(0, n-1)
        j0 = rng.randint(0, m-1)
        k0 = rng.randint(0, p-1)
        U3[i0*m+j0, r] = rng.uniform(0.8, 1.2)
        V3[j0*p+k0, r] = rng.uniform(0.8, 1.2)
        W3[k0*n+i0, r] = rng.uniform(0.8, 1.2)
        # Add some cross terms
        if rng.random() > 0.5:
            i1 = rng.randint(0, n-1)
            j1 = rng.randint(0, m-1)
            k1 = rng.randint(0, p-1)
            U3[i1*m+j1, r] = rng.uniform(-0.5, 0.5)
            V3[j1*p+k1, r] = rng.uniform(-0.5, 0.5)
            W3[k1*n+i1, r] = rng.uniform(-0.5, 0.5)
    inits.append((U3, V3, W3))
    
    best_result = None
    best_res_norm = float('inf')
    
    for U_init, V_init, W_init in inits:
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        prev_norm = float('inf')
        for it in range(max_iter):
            res = _residuals_fast(U, V, W, n, m, p, rank)
            res_norm = np.linalg.norm(res)
            
            if res_norm < tol:
                break
            if res_norm > prev_norm * 10:
                # Diverging, try to recover
                U = U_init.copy()
                V = V_init.copy()
                W = W_init.copy()
                prev_norm = float('inf')
                continue
            prev_norm = res_norm
            
            # Compute Jacobian
            J = _jacobian_fast(U, V, W, n, m, p, rank)
            num_u = n * m * rank
            num_v = m * p * rank
            num_w = p * n * rank
            
            try:
                # Use SVD for better conditioning
                delta = np.linalg.lstsq(J, -res.flatten(), rcond=1e-12)[0]
            except Exception:
                break
            
            # Adaptive step size
            step = 1.0
            for _ in range(10):
                U_new = U + step * delta[:num_u].reshape(n*m, rank)
                V_new = V + step * delta[num_u:num_u+num_v].reshape(m*p, rank)
                W_new = W + step * delta[num_u+num_v:].reshape(p*n, rank)
                res_new = _residuals_fast(U_new, V_new, W_new, n, m, p, rank)
                res_new_norm = np.linalg.norm(res_new)
                if res_new_norm < res_norm:
                    U, V, W = U_new, V_new, W_new
                    break
                step *= 0.5
            
        final_res = _residuals_fast(U, V, W, n, m, p, rank)
        final_norm = np.linalg.norm(final_res)
        if final_norm < best_res_norm:
            best_res_norm = final_norm
            best_result = (U.copy(), V.copy(), W.copy())
    
    if best_result and best_res_norm < 1e-6:
        return best_result
    return None

def _round_and_verify(U, V, W, n, m, p):
    """
    Round the float solution to nearest half-integers and verify.
    """
    # Try multiple rounding strategies
    candidates = []
    
    # Round to nearest 0.5
    U_r = [[round(float(v) * 2) / 2.0 for v in row] for row in U]
    V_r = [[round(float(v) * 2) / 2.0 for v in row] for row in V]
    W_r = [[round(float(v) * 2) / 2.0 for v in row] for row in W]
    candidates.append((U_r, V_r, W_r))
    
    # Round to integers
    U_i = [[round(float(v)) for v in row] for row in U]
    V_i = [[round(float(v)) for v in row] for row in V]
    W_i = [[round(float(v)) for v in row] for row in W]
    candidates.append((U_i, V_i, W_i))
    
    # Try floor/ceil combinations for borderline values
    U_fc = []
    V_fc = []
    W_fc = []
    for row in U:
        U_fc.append([round(float(v)*2)/2.0 if abs(float(v)*2 - round(float(v)*2)) > 0.1 else 
                     (int(float(v)*2)+1)/2.0 if float(v)*2 - int(float(v)*2) > 0.5 else int(float(v)*2)/2.0 
                     for v in row])
    for row in V:
        V_fc.append([round(float(v)*2)/2.0 if abs(float(v)*2 - round(float(v)*2)) > 0.1 else 
                     (int(float(v)*2)+1)/2.0 if float(v)*2 - int(float(v)*2) > 0.5 else int(float(v)*2)/2.0 
                     for v in row])
    for row in W:
        W_fc.append([round(float(v)*2)/2.0 if abs(float(v)*2 - round(float(v)*2)) > 0.1 else 
                     (int(float(v)*2)+1)/2.0 if float(v)*2 - int(float(v)*2) > 0.5 else int(float(v)*2)/2.0 
                     for v in row])
    candidates.append((U_fc, V_fc, W_fc))
    
    for U_c, V_c, W_c in candidates:
        if _verify(U_c, V_c, W_c, n, m, p):
            return U_c, V_c, W_c
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s limit
    
    # Schoolbook is always a valid fallback
    best = _schoolbook()
    best_rank = n * m * p
    
    # Try to find lower ranks
    # Lower bound is max(n*m, m*p, p*n) = 21
    # Try ranks from 22 upward
    for rank in range(22, best_rank):
        if time.time() > deadline:
            break
        
        found = False
        # Try more random seeds for Newton's method
        for seed in range(15):
            if time.time() > deadline:
                break
            
            result = _newton_search(rank, n, m, p, seed)
            if result:
                U, V, W = result
                verified = _round_and_verify(U, V, W, n, m, p)
                if verified:
                    U_r, V_r, W_r = verified
                    best = {"U": U_r, "V": V_r, "W": W_r}
                    best_rank = rank
                    found = True
                    break  # Found a valid decomposition for this rank, try lower rank
    
    return best
# EVOLVE-BLOCK-END
