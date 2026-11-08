import numpy as np
import time
import itertools
from fractions import Fraction

DIMENSIONS = [3, 3, 5]

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(R):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1) > 1e-9:
                    return False
    return True

def _is_half_integer_matrix(M, tol=1e-9):
    """Check if all entries are half-integers (integer or integer+0.5)."""
    for row in M:
        for x in row:
            if abs(x * 2 - round(x * 2)) > tol:
                return False
    return True

def _minimize_lattice_distance(W_particular, null_space, max_iter=100):
    """
    Given a particular solution W_particular and the null space of the linear system,
    find a vector c such that W_particular + null_space.T @ c is close to the half-integer lattice.
    """
    N = W_particular.shape[0]
    K = null_space.shape[0]
    if K == 0:
        return W_particular, 0.0

    best_c = np.zeros(K)
    best_dist = np.inf
    
    # Initial candidate: c=0
    W_curr = W_particular
    dist_curr = np.sum((W_curr * 2 - np.round(W_curr * 2))**2)
    if dist_curr < best_dist:
        best_dist = dist_curr
        best_c = np.zeros(K)
        
    # Try random small perturbations
    for _ in range(max_iter):
        step_dir = np.random.randn(K)
        for step_size in [0.1, 0.5, 1.0, 2.0]:
            c_trial = best_c + step_size * step_dir
            W_trial = W_particular + null_space.T @ c_trial
            dist_trial = np.sum((W_trial * 2 - np.round(W_trial * 2))**2)
            if dist_trial < best_dist:
                best_dist = dist_trial
                best_c = c_trial
                W_curr = W_trial
                if best_dist < 1e-10:
                    return W_curr, best_dist
                    
    return W_curr, best_dist

def _try_rank_r(R, n, m, p, deadline, max_restarts=2000):
    """Try to find a rank-R decomposition using the slice method."""
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        np.random.seed(restart * 12345 + R * 67890)
        
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        
        # Initialize U and V with sparse structure
        for i in range(n*m):
            num_nonzero = np.random.choice([1, 2], p=[0.5, 0.5])
            cols = np.random.choice(R, size=num_nonzero, replace=False)
            U[i, cols] = np.random.choice([-1, 1], size=num_nonzero)
            
        for j in range(m*p):
            num_nonzero = np.random.choice([1, 2], p=[0.5, 0.5])
            cols = np.random.choice(R, size=num_nonzero, replace=False)
            V[j, cols] = np.random.choice([-1, 1], size=num_nonzero)
            
        A = np.zeros((n*m*p, p*n*R), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq_idx = i*m*p + j*p + k
                    for r in range(R):
                        coeff = U[i*m+j, r] * V[j*p+k, r]
                        if coeff != 0:
                            var_idx = (k*n + i) * R + r
                            A[eq_idx, var_idx] = coeff
        
        b = np.ones(n*m*p)
        
        try:
            W_vec, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            continue
        
        error = np.linalg.norm(A @ W_vec - b)
        if error > 1e-7:
            continue
        
        W_mat = W_vec.reshape(p*n, R)
        W_scaled = W_mat * 2
        W_rounded = np.round(W_scaled)
        if np.allclose(W_scaled, W_rounded, atol=1e-6):
            W_final = W_rounded / 2.0
            if _verify(U, V, W_final, n, m, p):
                return U, V, W_final
        
        if rank < p*n*R:
            U_svd, s, Vt = np.linalg.svd(A, full_matrices=True)
            tol = np.max(s) * np.finfo(float).eps * max(A.shape)
            zero_indices = np.where(s <= tol)[0]
            
            if len(zero_indices) > 0:
                null_space = Vt[zero_indices]
                W_opt, dist = _minimize_lattice_distance(W_vec, null_space, max_iter=50)
                
                W_opt_mat = W_opt.reshape(p*n, R)
                W_opt_scaled = W_opt_mat * 2
                W_opt_rounded = np.round(W_opt_scaled)
                
                if np.allclose(W_opt_scaled, W_opt_rounded, atol=1e-6):
                    W_final = W_opt_rounded / 2.0
                    if _verify(U, V, W_final, n, m, p):
                        return U, V, W_final

    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    R_max = n * m * p  # 45
    
    U_sb = np.zeros((n*m, R_max))
    V_sb = np.zeros((m*p, R_max))
    W_sb = np.zeros((p*n, R_max))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
                
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    best_rank = R_max
    
    for R in range(R_max - 1, 14, -1):
        if time.time() > deadline - 5:
            break
        
        result = _try_rank_r(R, n, m, p, deadline - 5)
        if result is not None:
            U, V, W = result
            best_U = U
            best_V = V
            best_W = W
            best_rank = R
            # Do not break, try lower R
            
    U_out = [[float(x) for x in row] for row in best_U]
    V_out = [[float(x) for x in row] for row in best_V]
    W_out = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_out, "V": V_out, "W": W_out}
