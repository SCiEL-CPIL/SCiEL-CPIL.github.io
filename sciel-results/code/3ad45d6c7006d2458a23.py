import time
import sys
import numpy as np
from scipy.optimize import minimize, least_squares

START_TIME = time.time()
TIME_LIMIT = 140  # Safety margin before 150s kill

def time_up():
    return (time.time() - START_TIME) > TIME_LIMIT

def strassen_decomposition():
    """Return the Strassen-based decomposition (rank 28)."""
    n, m, p = 2, 4, 4
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    alpha = [
        {},
        { (0,0): 1, (1,1): 1 },
        { (1,0): 1, (1,1): 1 },
        { (0,0): 1 },
        { (1,1): 1 },
        { (0,0): 1, (0,1): 1 },
        { (1,0): 1, (0,0): -1 },
        { (0,1): 1, (1,1): -1 }
    ]
    
    beta = [
        {},
        { (0,0): 1, (1,1): 1 },
        { (0,0): 1 },
        { (0,1): 1, (1,1): -1 },
        { (1,0): 1, (0,0): -1 },
        { (1,1): 1 },
        { (0,0): 1, (0,1): 1 },
        { (1,0): 1, (1,1): 1 }
    ]
    
    w_coeff = [
        {},
        { (0,0): 1, (1,1): 1 },
        { (1,0): 1, (1,1): -1 },
        { (0,1): 1, (1,1): 1 },
        { (0,0): 1, (1,0): 1 },
        { (0,0): -1, (0,1): 1 },
        { (1,1): 1 },
        { (0,0): 1 }
    ]
    
    for s in range(1, 8):
        for t in range(4):
            c = 4 * (s - 1) + t
            k_a = 0 if t in [0, 2] else 1
            r_b = 1 if t in [1, 3] else 0
            c_b = 1 if t in [2, 3] else 0
            k_c = 0 if t in [0, 1] else 1
            
            for (i, j), val in alpha[s].items():
                if val != 0:
                    u_idx = i * m + 2 * j + k_a
                    U[u_idx][c] += val
            
            for (i, j), val in beta[s].items():
                if val != 0:
                    v_row = 2 * i + r_b
                    v_col = 2 * j + c_b
                    v_idx = v_row * p + v_col
                    V[v_idx][c] += val
            
            for (i, j), val in w_coeff[s].items():
                if val != 0:
                    w_col = 2 * j + k_c
                    w_row = i
                    w_idx = w_col * n + w_row
                    W[w_idx][c] += val
    
    return U, V, W

def verify_decomposition(U, V, W, n=2, m=4, p=4):
    """Verify that the decomposition is correct using numpy."""
    rank = U.shape[1]
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    T_recon = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_recon[i, j, k] = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
    
    T_target = np.ones((n, m, p), dtype=float)
    max_err = np.max(np.abs(T_recon - T_target))
    return max_err < 1e-9, max_err

def tensor_residual(x, n, m, p, rank):
    """Calculate residuals for the optimization problem."""
    size_U = n * m * rank
    size_V = m * p * rank
    U = x[:size_U].reshape(n*m, rank)
    V = x[size_U : size_U + size_V].reshape(m*p, rank)
    W = x[size_U + size_V : ].reshape(p*n, rank)
    
    T_recon = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_recon[i, j, k] = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
    
    return T_recon.flatten() - 1.0

def try_rank_reduction(target_rank, n=2, m=4, p=4, max_restarts=15):
    """Try to find a rank-target_rank decomposition using Strassen as base."""
    if time_up():
        return None
    
    size_U = n * m * target_rank
    size_V = m * p * target_rank
    size_W = p * n * target_rank
    total_size = size_U + size_V + size_W
    
    U_s, V_s, W_s = strassen_decomposition()
    
    best_result = None
    best_error = float('inf')
    
    for restart in range(max_restarts):
        if time_up():
            break
        
        # Strategy: create a random linear combination of Strassen terms
        # to get a target_rank-dimensional subspace
        if restart == 0:
            # Truncate directly
            U_init = U_s[:, :target_rank].copy()
            V_init = V_s[:, :target_rank].copy()
            W_init = W_s[:, :target_rank].copy()
        else:
            # Random mixing of Strassen columns
            np.random.seed(42 + restart)
            mix_matrix = np.random.randn(28, target_rank) * 0.5
            # Ensure first target_rank columns are somewhat preserved
            mix_matrix[:target_rank, :target_rank] += np.eye(target_rank)
            
            U_init = U_s @ mix_matrix
            V_init = V_s @ mix_matrix
            W_init = W_s @ mix_matrix
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            # Use least_squares with 'trf' method which handles this better
            result = least_squares(
                tensor_residual, x0, args=(n, m, p, target_rank),
                method='trf', max_nfev=3000, ftol=1e-12, xtol=1e-12, gtol=1e-12
            )
            err = np.linalg.norm(result.fun)
            
            if err < best_error:
                best_error = err
                best_result = result.x.copy()
                
            if err < 1e-9:
                U = best_result[:size_U].reshape(n*m, target_rank)
                V = best_result[size_U : size_U + size_V].reshape(m*p, target_rank)
                W = best_result[size_U + size_V : ].reshape(p*n, target_rank)
                
                valid, _ = verify_decomposition(U, V, W, n, m, p)
                if valid:
                    return U.tolist(), V.tolist(), W.tolist()
                    
        except Exception:
            pass
    
    return None

def try_random_search(target_rank, n=2, m=4, p=4, max_restarts=10):
    """Try random initialization for rank reduction."""
    if time_up():
        return None
    
    size_U = n * m * target_rank
    size_V = m * p * target_rank
    size_W = p * n * target_rank
    total_size = size_U + size_V + size_W
    
    best_result = None
    best_error = float('inf')
    
    for restart in range(max_restarts):
        if time_up():
            break
        
        np.random.seed(100 + restart)
        x0 = np.random.randn(total_size) * 0.3
        
        try:
            result = least_squares(
                tensor_residual, x0, args=(n, m, p, target_rank),
                method='trf', max_nfev=2000, ftol=1e-12, xtol=1e-12, gtol=1e-12
            )
            err = np.linalg.norm(result.fun)
            
            if err < best_error:
                best_error = err
                best_result = result.x.copy()
                
            if err < 1e-9:
                U = best_result[:size_U].reshape(n*m, target_rank)
                V = best_result[size_U : size_U + size_V].reshape(m*p, target_rank)
                W = best_result[size_U + size_V : ].reshape(p*n, target_rank)
                
                valid, _ = verify_decomposition(U, V, W, n, m, p)
                if valid:
                    return U.tolist(), V.tolist(), W.tolist()
                    
        except Exception:
            pass
    
    return None

def construct():
    n, m, p = 2, 4, 4
    
    # Start with Strassen (rank 28)
    U, V, W = strassen_decomposition()
    best_rank = 28
    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try rank 24 with Strassen-based approach
    if not time_up():
        res = try_rank_reduction(24, n, m, p, max_restarts=12)
        if res is not None:
            U2, V2, W2 = res
            best_rank = 24
            best_result = {"U": U2, "V": V2, "W": W2}
    
    # If rank 24 not found, try random search for rank 24
    if best_rank > 24 and not time_up():
        res = try_random_search(24, n, m, p, max_restarts=8)
        if res is not None:
            U2, V2, W2 = res
            best_rank = 24
            best_result = {"U": U2, "V": V2, "W": W2}
    
    # If rank 24 not found, try intermediate ranks
    if best_rank > 24 and not time_up():
        for r in [25, 26, 27]:
            if time_up():
                break
            res = try_rank_reduction(r, n, m, p, max_restarts=6)
            if res is not None:
                U2, V2, W2 = res
                best_rank = r
                best_result = {"U": U2, "V": V2, "W": W2}
                break
    
    return best_result

if __name__ == "__main__":
    result = construct()
    print(f"Best rank: {len(result['U'][0])}")
    valid, err = verify_decomposition(result['U'], result['V'], result['W'])
    print(f"Valid: {valid}, Max error: {err}")
