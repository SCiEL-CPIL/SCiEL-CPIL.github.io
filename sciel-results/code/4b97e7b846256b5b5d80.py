import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 4, 4]

def _build_tensor(n, m, p):
    """
    Build the multiplication tensor T of shape (n*m, m*p, p*n).
    T[i, j, k] = 1 if i, j, k correspond to a valid multiplication term contributing to output k.
    
    Indexing:
    i = r * m + c   (A row r, col c)
    j = c' * p + k' (B row c', col k')
    k = k'' * n + r'' (C row r'', col k'')
    
    For standard matrix multiplication:
    C_{r, k''} = sum_{c} A_{r, c} B_{c, k''}
    
    So, a term A_{r, c} B_{c, k''} contributes to C_{r, k''}.
    i = r*m + c
    j = c*p + k''
    k = k''*n + r
    
    T[i, j, k] = 1
    """
    T = np.zeros((n * m, m * p, p * n))
    for r in range(n):
        for c in range(m):
            for k in range(p):
                i = r * m + c
                j = c * p + k
                kk = k * n + r
                T[i, j, kk] = 1.0
    return T

def _strassen_decomposition(n, m, p):
    """
    Construct the known rank-28 Strassen-based decomposition for 2x4x4.
    """
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen 2x2 patterns
    # A = [[a11, a12], [a21, a22]] -> indices 0,1,2,3
    # B = [[b11, b12], [b21, b22]] -> indices 0,1,2,3
    # C = [[c11, c12], [c21, c22]] -> indices 0,1,2,3
    
    patterns = [
        # M1 = (a11+a22)(b11+b22) -> c11 += M1, c22 += M1
        {'a': {0: 1, 3: 1}, 'b': {0: 1, 3: 1}, 'c': {0: 1, 3: 1}},
        # M2 = (a21+a22)(b11) -> c21 += M2, c22 -= M2
        {'a': {2: 1, 3: 1}, 'b': {0: 1}, 'c': {2: 1, 3: -1}},
        # M3 = (a11)(b12-b22) -> c12 += M3, c22 += M3
        {'a': {0: 1}, 'b': {1: 1, 3: -1}, 'c': {1: 1, 3: 1}},
        # M4 = (a22)(b21-b11) -> c11 += M4, c21 += M4
        {'a': {3: 1}, 'b': {2: 1, 0: -1}, 'c': {0: 1, 2: 1}},
        # M5 = (a11+a12)(b22) -> c11 -= M5, c12 += M5
        {'a': {0: 1, 1: 1}, 'b': {3: 1}, 'c': {0: -1, 1: 1}},
        # M6 = (a21-a11)(b11+b12) -> c22 += M6
        {'a': {2: 1, 0: -1}, 'b': {0: 1, 1: 1}, 'c': {3: 1}},
        # M7 = (a12-a22)(b21+b22) -> c11 += M7
        {'a': {1: 1, 3: -1}, 'b': {2: 1, 3: 1}, 'c': {0: 1}},
    ]
    
    # Blocks for 2x4x4:
    # A is 2x4. Split A into A_left (2x2) and A_right (2x2).
    # B is 4x4. Split B into 4 blocks 2x2: B11, B12, B21, B22.
    # C is 2x4. Split C into C_left (2x2) and C_right (2x2).
    
    # C_left = A_left * B11 + A_right * B21
    # C_right = A_left * B12 + A_right * B22
    
    # We apply Strassen to each of the 4 products:
    # 1. A_left * B11 -> adds to C_left
    # 2. A_right * B21 -> adds to C_left
    # 3. A_left * B12 -> adds to C_right
    # 4. A_right * B22 -> adds to C_right
    
    blocks = [
        # (a_col_offset, b_row_offset, b_col_offset, c_col_offset)
        (0, 0, 0, 0), # A[:, 0:2] * B[0:2, 0:2] -> C[:, 0:2]
        (2, 2, 0, 0), # A[:, 2:4] * B[2:4, 0:2] -> C[:, 0:2]
        (0, 0, 2, 2), # A[:, 0:2] * B[0:2, 2:4] -> C[:, 2:4]
        (2, 2, 2, 2), # A[:, 2:4] * B[2:4, 2:4] -> C[:, 2:4]
    ]
    
    col_idx = 0
    for a_off, b_r_off, b_c_off, c_off in blocks:
        for pat in patterns:
            # Fill U
            for local_i, coeff in pat['a'].items():
                r = local_i // 2
                c = local_i % 2
                global_i = r * m + (a_off + c)
                U[global_i, col_idx] = coeff
            
            # Fill V
            for local_j, coeff in pat['b'].items():
                r = local_j // 2
                c = local_j % 2
                global_j = (b_r_off + r) * p + (b_c_off + c)
                V[global_j, col_idx] = coeff
            
            # Fill W
            for local_k, coeff in pat['c'].items():
                r = local_k // 2
                c = local_k % 2
                # W index: k*n + i  (Trace convention: W[k*n+i] where i is row of C)
                # C row is r, C col is c.
                # So global_k (col) = c_off + c
                # global_i (row) = r
                global_k = c_off + c
                global_i = r
                W[global_k * n + global_i, col_idx] = coeff
            
            col_idx += 1
            
    return U, V, W

def _try_optimize_rank(T, n, m, p, target_rank, seed=0):
    """
    Attempt to find a decomposition of rank 'target_rank' for tensor T.
    Returns (U, V, W) if found with low error, else None.
    """
    np.random.seed(seed)
    
    # Initial guess: Use the first 'target_rank' columns of the Strassen decomposition?
    # Or random initialization.
    # Let's try random initialization with Gaussian noise.
    U = np.random.randn(n * m, target_rank)
    V = np.random.randn(m * p, target_rank)
    W = np.random.randn(p * n, target_rank)
    
    # Flatten tensor
    T_flat = T.flatten()
    dim_u, dim_v, dim_w = n * m, m * p, p * n
    
    def residuals(x):
        U = x[:dim_u * target_rank].reshape(dim_u, target_rank)
        V = x[dim_u * target_rank : dim_u * target_rank + dim_v * target_rank].reshape(dim_v, target_rank)
        W = x[dim_u * target_rank + dim_v * target_rank :].reshape(dim_w, target_rank)
        
        # Compute tensor product sum
        # T_calc[i, j, k] = sum_t U[i, t] V[j, t] W[k, t]
        # This is an einsum: 'it, jt, kt -> ijk'
        T_calc = np.einsum('it,jt,kt->ijk', U, V, W)
        return (T_calc - T).flatten()
    
    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    try:
        # Use least_squares to minimize || T_calc - T ||_F
        # This is a non-linear least squares problem.
        result = least_squares(residuals, x0, method='lm', max_nfev=10000)
        
        if result.success and result.cost < 1e-12:
            U_sol = result.x[:dim_u * target_rank].reshape(dim_u, target_rank)
            V_sol = result.x[dim_u * target_rank : dim_u * target_rank + dim_v * target_rank].reshape(dim_v, target_rank)
            W_sol = result.x[dim_u * target_rank + dim_v * target_rank :].reshape(dim_w, target_rank)
            
            # Verify exactness
            T_check = np.einsum('it,jt,kt->ijk', U_sol, V_sol, W_sol)
            if np.allclose(T_check, T, atol=1e-9):
                return U_sol, V_sol, W_sol
    except Exception:
        pass
    return None

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # 1. Get the known rank-28 solution
    U28, V28, W28 = _strassen_decomposition(n, m, p)
    T = _build_tensor(n, m, p)
    
    best_U, best_V, best_W = U28, V28, W28
    best_rank = 28
    
    # 2. Try to reduce rank
    # Lower bound from catalecticant:
    # Flattening V (16) x U⊗W (8*8=64). Rank is likely 16.
    # Let's try ranks 16, 17, ..., 27
    for target_rank in range(16, 28):
        if time.time() - start_time > 100.0:
            break
            
        # Try a few seeds
        for seed in range(5):
            if time.time() - start_time > 100.0:
                break
                
            res = _try_optimize_rank(T, n, m, p, target_rank, seed=seed)
            if res is not None:
                U_sol, V_sol, W_sol = res
                # Convert to list of lists of floats
                # The interface expects integer/half-integer entries.
                # Optimization might produce floating point numbers.
                # We need to check if they are close to half-integers.
                # If not, we might need to round, but that risks breaking the decomposition.
                # The task says "integer/half-integer entries".
                # Strassen uses integers {-1, 0, 1}.
                # If optimization finds a solution, it might be rational.
                # Let's check if we can snap to half-integers.
                
                U_snap = np.round(U_sol * 2) / 2
                V_snap = np.round(V_sol * 2) / 2
                W_snap = np.round(W_sol * 2) / 2
                
                T_check = np.einsum('it,jt,kt->ijk', U_snap, V_snap, W_snap)
                if np.allclose(T_check, T, atol=1e-9):
                    best_U, best_V, best_W = U_snap, V_snap, W_snap
                    best_rank = target_rank
                    break # Found a valid decomposition for this rank, try lower next
        else:
            continue
        if best_rank == target_rank:
            # We found a solution for this rank. Do we want to try lower?
            # Yes, the loop continues to lower ranks.
            pass
        
    # If we found a better rank, return it.
    # Otherwise return the rank 28 Strassen solution.
    # Convert numpy arrays to lists
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}

if __name__ == "__main__":
    res = construct()
    print("Rank used:", len(res['U'][0]))
