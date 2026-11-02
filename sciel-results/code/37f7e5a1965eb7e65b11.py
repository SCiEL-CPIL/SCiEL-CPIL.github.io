import numpy as np
import time

DIMENSIONS = [2, 2, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Strassen parameters for 2x2x2
    U_strassen = [
        [1, 0, 0, 1], # P1
        [0, 0, 1, 1], # P2
        [1, 0, 0, 0], # P3
        [0, 0, 0, 1], # P4
        [1, 1, 0, 0], # P5
        [-1, 0, 1, 0],# P6
        [0, 1, 0, -1] # P7
    ]
    
    V_strassen = [
        [1, 0, 0, 1], # P1
        [1, 0, 0, 0], # P2
        [0, 1, 0, -1],# P3
        [-1, 0, 1, 0],# P4
        [0, 0, 0, 1], # P5
        [1, 1, 0, 0], # P6
        [0, 0, 1, 1]  # P7
    ]
    
    W_strassen = [
        [1, 0, 0, 1], # P1
        [0, 0, 1, -1],# P2
        [0, 1, 0, 1], # P3
        [1, 0, 1, 0], # P4
        [-1, 1, 0, 0],# P5
        [0, 0, 0, 1], # P6
        [1, 0, 0, 0]  # P7
    ]
    
    num_blocks = p // 2
    rank = num_blocks * 7
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for t in range(num_blocks):
        for r in range(7):
            for i in range(n * m):
                U[i][col] = U_strassen[r][i]
                
            v_indices = [2*t, 2*t+1, 8+2*t, 8+2*t+1]
            for l in range(4):
                V[v_indices[l]][col] = V_strassen[r][l]
                
            w_indices = [4*t, 4*t+2, 4*t+1, 4*t+3]
            for l in range(4):
                W[w_indices[l]][col] = W_strassen[r][l]
                
            col += 1
    
    # Verify the base decomposition
    def verify(U_mat, V_mat, W_mat):
        U_arr = np.array(U_mat, dtype=float)
        V_arr = np.array(V_mat, dtype=float)
        W_arr = np.array(W_mat, dtype=float)
        
        T = np.einsum('ir,jr,kr->ijk', U_arr, V_arr, W_arr)
        
        T_expected = np.zeros((n*m, m*p, p*n), dtype=float)
        for i in range(n*m):
            a_row = i // m
            a_col = i % m
            for j in range(m*p):
                b_row = j // p
                b_col = j % p
                if a_col != b_row:
                    continue
                for k in range(p*n):
                    c_col = k // n
                    c_row = k % n
                    if a_row == c_row and b_col == c_col:
                        T_expected[i,j,k] = 1.0
        
        if not np.allclose(T, T_expected, atol=1e-10):
            return False
        return True
    
    if not verify(U, V, W):
        raise ValueError("Base Strassen decomposition failed verification")
    
    # Build target tensor once
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    T_target = np.einsum('ir,jr,kr->ijk', U_arr, V_arr, W_arr)
    
    def try_rank_r(r, time_limit):
        """Try to find a rank-r decomposition using ALS with multiple restarts."""
        start_t = time.time()
        
        for attempt in range(30):
            if time.time() - start_t > time_limit:
                break
            
            # Use structured initialization based on Strassen
            np.random.seed(1000 + attempt)
            
            if attempt < 10:
                # Initialize close to Strassen but with random mixing across blocks
                U_try = np.zeros((n*m, r))
                V_try = np.zeros((m*p, r))
                W_try = np.zeros((p*n, r))
                
                # Assign each of the 28 Strassen terms to one of r slots
                # with some random permutation
                perm = np.random.permutation(28)
                for idx in range(min(28, r)):
                    s = perm[idx]
                    block = s // 7
                    strassen_idx = s % 7
                    
                    # U factor
                    for i in range(n*m):
                        U_try[i, idx] = U_strassen[strassen_idx][i]
                    
                    # V factor
                    v_indices = [2*block, 2*block+1, 8+2*block, 8+2*block+1]
                    for l in range(4):
                        V_try[v_indices[l], idx] = V_strassen[strassen_idx][l]
                    
                    # W factor
                    w_indices = [4*block, 4*block+2, 4*block+1, 4*block+3]
                    for l in range(4):
                        W_try[w_indices[l], idx] = W_strassen[strassen_idx][l]
                
                # If r < 28, we need to merge some terms
                if r < 28:
                    # Randomly scale and add to existing terms
                    for idx in range(28 - r):
                        target_idx = np.random.randint(0, r)
                        scale = np.random.choice([-1, 1]) * np.random.uniform(0.5, 1.5)
                        s = perm[r + idx]
                        block = s // 7
                        strassen_idx = s % 7
                        
                        for i in range(n*m):
                            U_try[i, target_idx] += scale * U_strassen[strassen_idx][i]
                        
                        v_indices = [2*block, 2*block+1, 8+2*block, 8+2*block+1]
                        for l in range(4):
                            V_try[v_indices[l], target_idx] += scale * V_strassen[strassen_idx][l]
                        
                        w_indices = [4*block, 4*block+2, 4*block+1, 4*block+3]
                        for l in range(4):
                            W_try[w_indices[l], target_idx] += scale * W_strassen[strassen_idx][l]
            else:
                # Random initialization
                U_try = np.random.randn(n*m, r)
                V_try = np.random.randn(m*p, r)
                W_try = np.random.randn(p*n, r)
            
            # ALS iterations
            best_err = np.inf
            best_U, best_V, best_W = None, None, None
            
            for iteration in range(300):
                if time.time() - start_t > time_limit:
                    break
                
                # Update U
                D = np.einsum('jr,kr->jkr', V_try, W_try)
                D_flat = D.reshape(m*p * p*n, r)
                for i in range(n*m):
                    target = T_target[i, :, :].flatten()
                    U_try[i, :], _, _, _ = np.linalg.lstsq(D_flat, target, rcond=None)
                
                # Update V
                D2 = np.einsum('ir,kr->ikr', U_try, W_try)
                D2_flat = D2.reshape(n*m * p*n, r)
                for j in range(m*p):
                    target = T_target[:, j, :].reshape(n*m * p*n)
                    V_try[j, :], _, _, _ = np.linalg.lstsq(D2_flat, target, rcond=None)
                
                # Update W
                D3 = np.einsum('ir,jr->ijr', U_try, V_try)
                D3_flat = D3.reshape(n*m * m*p, r)
                for k in range(p*n):
                    target = T_target[:, :, k].reshape(n*m * m*p)
                    W_try[k, :], _, _, _ = np.linalg.lstsq(D3_flat, target, rcond=None)
                
                # Check error
                T_recon = np.einsum('ir,jr,kr->ijk', U_try, V_try, W_try)
                err = np.linalg.norm(T_recon - T_target)
                
                if err < best_err:
                    best_err = err
                    best_U = U_try.copy()
                    best_V = V_try.copy()
                    best_W = W_try.copy()
                
                if err < 1e-12:
                    break
            
            # Check if we found a valid decomposition
            if best_err < 1e-10:
                # Try to round to integers or half-integers
                for mult in [1, 2, 4]:
                    U_rounded = np.round(best_U * mult) / mult
                    V_rounded = np.round(best_V * mult) / mult
                    W_rounded = np.round(best_W * mult) / mult
                    
                    T_rounded = np.einsum('ir,jr,kr->ijk', U_rounded, V_rounded, W_rounded)
                    if np.allclose(T_rounded, T_target, atol=1e-8):
                        return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), r
                
                # If rounding fails, check if values are already close to rational
                # Just return as-is if error is tiny
                if best_err < 1e-12:
                    return best_U.tolist(), best_V.tolist(), best_W.tolist(), r
        
        return None
    
    # Try ranks from 27 down to 16
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        # Allocate time: try higher ranks first with more time
        for r in range(27, 15, -1):
            if time.time() > deadline - 5:
                break
            
            # Give more time to higher ranks (more likely to succeed)
            time_for_rank = min(30, remaining_time * 0.3)
            result = try_rank_r(r, time_for_rank)
            
            if result is not None:
                U_list, V_list, W_list, rank_found = result
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to the original rank-28 decomposition
    return {"U": U, "V": V, "W": W}
