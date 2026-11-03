import time
import sys
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 4]
START_TIME = time.time()
TIME_LIMIT = 140  # Safety margin before 150s kill

def time_up():
    return (time.time() - START_TIME) > TIME_LIMIT

def verify_fast(U_arr, V_arr, W_arr, n=2, m=4, p=4, tol=1e-9):
    """Fast verification using vectorized numpy operations."""
    rank = U_arr.shape[1]
    T_recon = np.zeros((n, m, p), dtype=float)
    for r in range(rank):
        u_col = U_arr[:, r].reshape(n, m)
        v_col = V_arr[:, r].reshape(m, p)
        w_col = W_arr[:, r].reshape(p, n)
        for i in range(n):
            contrib = u_col[i] * (v_col * w_col[:, i].T)
            T_recon[i] += contrib
    max_err = np.max(np.abs(T_recon - 1.0))
    return max_err < tol, max_err

def strassen_decomposition():
    """Return the Strassen-based decomposition (rank 28)."""
    n, m, p = 2, 4, 4
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen coefficients for 2x2 matmul:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1+M4-M5-M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M7
    
    alpha = [
        {(0,0): 1, (1,1): 1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (0,0): -1},
        {(0,1): 1, (1,1): -1}
    ]
    
    beta = [
        {(0,0): 1, (1,1): 1},
        {(0,0): 1},
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (0,0): -1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (1,1): 1}
    ]
    
    w_coeff = [
        {(0,0): 1, (1,1): 1},
        {(1,0): 1, (1,1): -1},
        {(0,1): 1, (1,1): 1},
        {(0,0): 1, (1,0): 1},
        {(0,0): -1, (0,1): 1},
        {(1,1): 1},
        {(0,0): 1}
    ]
    
    for s in range(7):
        for t in range(4):
            c = 4 * s + t
            k_a = 0 if t in [0, 2] else 1
            r_b = 1 if t in [1, 3] else 0
            c_b = 1 if t in [2, 3] else 0
            k_c = 0 if t in [0, 1] else 1
            
            for (i, j), val in alpha[s].items():
                u_idx = i * m + 2 * j + k_a
                U[u_idx, c] += val
            
            for (i, j), val in beta[s].items():
                v_row = 2 * i + r_b
                v_col = 2 * j + c_b
                v_idx = v_row * p + v_col
                V[v_idx, c] += val
            
            for (i, j), val in w_coeff[s].items():
                w_col = 2 * j + k_c
                w_row = i
                w_idx = w_col * n + w_row
                W[w_idx, c] += val
    
    return U, V, W

def find_linear_dependence(U_arr, V_arr, W_arr, n=2, m=4, p=4):
    """Find linear dependencies among rank-1 tensors."""
    rank = U_arr.shape[1]
    nmp = n * m * p
    T_matrix = np.zeros((nmp, rank))
    
    for r in range(rank):
        u_col = U_arr[:, r].reshape(n, m)
        v_col = V_arr[:, r].reshape(m, p)
        w_col = W_arr[:, r].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_matrix[i * m * p + j * p + k, r] = u_col[i, j] * v_col[j, k] * w_col[k, i]
    
    # Find null space
    U_svd, s_vals, Vt_svd = np.linalg.svd(T_matrix, full_matrices=False)
    tol = 1e-10
    rank_svd = np.sum(s_vals > tol)
    
    if rank_svd < rank:
        # Null space vectors give dependencies
        null_vecs = Vt_svd[rank_svd:]
        return null_vecs, rank_svd
    return None, rank

def greedy_remove(U_arr, V_arr, W_arr, n=2, m=4, p=4):
    """Greedy removal of redundant terms."""
    current_rank = U_arr.shape[1]
    for _ in range(10):
        if time_up():
            break
        removed = False
        for r in range(current_rank):
            if time_up():
                break
            mask = np.ones(current_rank, dtype=bool)
            mask[r] = False
            new_U = U_arr[:, mask]
            new_V = V_arr[:, mask]
            new_W = W_arr[:, mask]
            valid, err = verify_fast(new_U, new_V, new_W, n, m, p)
            if valid:
                U_arr = new_U
                V_arr = new_V
                W_arr = new_W
                current_rank = len(mask)
                removed = True
                break
        if not removed:
            break
    return U_arr, V_arr, W_arr, current_rank

def construct():
    n, m, p = 2, 4, 4
    
    # Start with Strassen rank-28
    U, V, W = strassen_decomposition()
    valid, err = verify_fast(U, V, W, n, m, p)
    best_rank = 28 if valid else 32
    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    if not valid:
        # Fallback to schoolbook
        rank = 32
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, idx] = 1.0
                    V[j*p+k, idx] = 1.0
                    W[k*n+i, idx] = 1.0
                    idx += 1
        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        return best_result
    
    # Try greedy removal
    if not time_up():
        U2, V2, W2, rank2 = greedy_remove(U, V, W, n, m, p)
        if rank2 < best_rank:
            best_rank = rank2
            best_result = {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}
    
    # Try to find and exploit linear dependencies
    if not time_up() and best_rank > 14:
        null_vecs, actual_rank = find_linear_dependence(U, V, W, n, m, p)
        if null_vecs is not None and actual_rank < best_rank:
            # Use null space to reduce rank
            # For each null vector, we can eliminate one term
            # This is complex; just note the potential
            pass
    
    # Try random perturbation + local search
    if not time_up() and best_rank > 14:
        np.random.seed(42)
        for target_rank in range(min(best_rank - 1, 27), 13, -1):
            if time_up():
                break
            found = False
            for attempt in range(100):
                if time_up():
                    break
                # Start from current solution and perturb
                U_try = U.copy()
                V_try = V.copy()
                W_try = W.copy()
                
                # Try removing random terms and see if we can fix
                if target_rank < U_try.shape[1]:
                    # Zero out some random columns
                    cols_to_keep = np.random.choice(U_try.shape[1], target_rank, replace=False)
                    U_try = U_try[:, cols_to_keep]
                    V_try = V_try[:, cols_to_keep]
                    W_try = W_try[:, cols_to_keep]
                
                valid, err = verify_fast(U_try, V_try, W_try, n, m, p)
                if valid:
                    best_rank = target_rank
                    best_result = {"U": U_try.tolist(), "V": V_try.tolist(), "W": W_try.tolist()}
                    found = True
                    break
            
            if found:
                break
    
    return best_result

if __name__ == "__main__":
    result = construct()
    print(f"Best rank: {len(result['U'][0])}")
    valid, err = verify_fast(np.array(result['U']), np.array(result['V']), np.array(result['W']))
    print(f"Valid: {valid}, Max error: {err}")
