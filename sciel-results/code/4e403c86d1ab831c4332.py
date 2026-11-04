import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 8]

def _build_target_tensor(n, m, p):
    """Build the target multiplication tensor."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for k in range(m):
            for j in range(p):
                u = i * m + k
                v = k * p + j
                w = j * n + i
                T[u, v, w] = 1.0
    return T

def _tensor_error(U, V, W, T_target):
    """Compute the max absolute error of the decomposition."""
    rank = U.shape[1]
    T_decomp = np.zeros_like(T_target)
    for r in range(rank):
        T_decomp += np.einsum('u,v,w->uvw', U[:, r], V[:, r], W[:, r])
    return np.max(np.abs(T_decomp - T_target))

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    T_target = _build_target_tensor(n, m, p)
    error = _tensor_error(U, V, W, T_target)
    return error < tol, error

def _construct_rank_88():
    """Construct the rank-88 decomposition using Strassen blocks."""
    n, m, p = DIMENSIONS  # 3, 4, 8
    rank = 88
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    def strassen_block(U, V, W, col, a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        u_a = a_rows[0] * m + a_cols[0]
        u_b = a_rows[0] * m + a_cols[1]
        u_c = a_rows[1] * m + a_cols[0]
        u_d = a_rows[1] * m + a_cols[1]
        
        v_e = b_rows[0] * p + b_cols[0]
        v_f = b_rows[0] * p + b_cols[1]
        v_g = b_rows[1] * p + b_cols[0]
        v_h = b_rows[1] * p + b_cols[1]
        
        w_00 = c_cols[0] * n + c_rows[0]
        w_01 = c_cols[1] * n + c_rows[0]
        w_10 = c_cols[0] * n + c_rows[1]
        w_11 = c_cols[1] * n + c_rows[1]
        
        # M1: (a+d)(e+h)
        U[u_a, col] += 1.0; U[u_d, col] += 1.0
        V[v_e, col] += 1.0; V[v_h, col] += 1.0
        W[w_00, col] += 1.0; W[w_11, col] += 1.0
        col += 1
        
        # M2: (c+d)e
        U[u_c, col] += 1.0; U[u_d, col] += 1.0
        V[v_e, col] += 1.0
        W[w_10, col] += 1.0; W[w_11, col] += -1.0
        col += 1
        
        # M3: a(f-h)
        U[u_a, col] += 1.0
        V[v_f, col] += 1.0; V[v_h, col] += -1.0
        W[w_01, col] += 1.0; W[w_11, col] += 1.0
        col += 1
        
        # M4: d(g-e)
        U[u_d, col] += 1.0
        V[v_g, col] += 1.0; V[v_e, col] += -1.0
        W[w_00, col] += 1.0; W[w_10, col] += 1.0
        col += 1
        
        # M5: (a+b)h
        U[u_a, col] += 1.0; U[u_b, col] += 1.0
        V[v_h, col] += 1.0
        W[w_00, col] += -1.0; W[w_01, col] += 1.0
        col += 1
        
        # M6: (c-a)(e+f)
        U[u_c, col] += 1.0; U[u_a, col] += -1.0
        V[v_e, col] += 1.0; V[v_f, col] += 1.0
        W[w_11, col] += 1.0
        col += 1
        
        # M7: (b-d)(g+h)
        U[u_b, col] += 1.0; U[u_d, col] += -1.0
        V[v_g, col] += 1.0; V[v_h, col] += 1.0
        W[w_00, col] += 1.0
        col += 1
        
        return col
    
    # Part 1a: A[0:2, 0:2] * B[0:2, :]
    for k_start in range(0, 8, 2):
        col = strassen_block(U, V, W, col,
            a_rows=[0, 1], a_cols=[0, 1],
            b_rows=[0, 1], b_cols=[k_start, k_start+1],
            c_rows=[0, 1], c_cols=[k_start, k_start+1])
    
    # Part 1b: A[0:2, 2:4] * B[2:4, :]
    for k_start in range(0, 8, 2):
        col = strassen_block(U, V, W, col,
            a_rows=[0, 1], a_cols=[2, 3],
            b_rows=[2, 3], b_cols=[k_start, k_start+1],
            c_rows=[0, 1], c_cols=[k_start, k_start+1])
    
    # Part 2: A[2:3, :] * B[:, :]
    for k in range(4):
        for j in range(8):
            U[2 * m + k, col] = 1.0
            V[k * p + j, col] = 1.0
            W[j * n + 2, col] = 1.0
            col += 1
    
    if col != rank:
        raise Exception(f"Rank mismatch: expected {rank}, got {col}")
    
    valid, error = _verify_decomposition(U, V, W, n, m, p)
    if not valid:
        raise Exception(f"Verification failed with error {error}")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _try_reduce_rank(U, V, W, target_rank, max_iters=500, seed=42):
    """Try to reduce the rank of the decomposition."""
    n, m, p = DIMENSIONS
    T_target = _build_target_tensor(n, m, p)
    
    # Flatten the problem
    # We have variables: U (12*r), V (32*r), W (24*r)
    # Total: (12+32+24)*r = 68*r variables
    
    def objective(params):
        r = target_rank
        U_flat = params[:12*r]
        V_flat = params[12*r:12*r+32*r]
        W_flat = params[12*r+32*r:]
        
        U = U_flat.reshape(12, r)
        V = V_flat.reshape(32, r)
        W = W_flat.reshape(24, r)
        
        T_decomp = np.zeros((12, 32, 24))
        for rr in range(r):
            T_decomp += np.einsum('u,v,w->uvw', U[:, rr], V[:, rr], W[:, rr])
        
        return np.sum((T_decomp - T_target)**2)
    
    # Initialize from the existing solution (truncate or project)
    current_rank = U.shape[1]
    if current_rank >= target_rank:
        # Use the first target_rank terms
        U_init = U[:, :target_rank]
        V_init = V[:, :target_rank]
        W_init = W[:, :target_rank]
    else:
        # Pad with random
        U_init = np.random.randn(12, target_rank)
        V_init = np.random.randn(32, target_rank)
        W_init = np.random.randn(24, target_rank)
    
    params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
    
    # Use L-BFGS-B
    result = minimize(objective, params_init, method='L-BFGS-B', 
                      options={'maxiter': max_iters, 'maxfun': 10000})
    
    if result.success or result.fun < 1e-6:
        params = result.x
        U_new = params[:12*target_rank].reshape(12, target_rank)
        V_new = params[12*target_rank:12*target_rank+32*target_rank].reshape(32, target_rank)
        W_new = params[12*target_rank+32*target_rank:].reshape(24, target_rank)
        
        valid, error = _verify_decomposition(U_new, V_new, W_new, n, m, p)
        if valid:
            return U_new, V_new, W_new
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Start with rank-88
    result_88 = _construct_rank_88()
    best_rank = 88
    best_result = result_88
    
    # Try to reduce rank
    U = np.array(result_88['U'])
    V = np.array(result_88['V'])
    W = np.array(result_88['W'])
    
    for target_rank in [84, 82, 80, 78]:
        if time.time() > deadline:
            break
        
        print(f"Trying rank {target_rank}...")
        reduced = _try_reduce_rank(U, V, W, target_rank, max_iters=200)
        
        if reduced is not None:
            U_new, V_new, W_new = reduced
            best_rank = target_rank
            best_result = {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
            print(f"Successfully reduced to rank {target_rank}")
            break
    
    if time.time() > deadline:
        return best_result
    
    return best_result
