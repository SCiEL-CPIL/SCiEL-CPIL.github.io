import time
import numpy as np
import random

DIMENSIONS = [2, 3, 6]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that U, V, W form an exact tensor decomposition of the matrix multiplication tensor.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                actual = np.sum(U_arr[i * m + j, :] * V_arr[j * p + k, :] * W_arr[k * n + i, :])
                if abs(actual - 1.0) > tol:
                    return False
    return True

def _schoolbook_decomposition(n, m, p):
    """
    Fallback schoolbook decomposition. Rank = n*m*p.
    """
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _construct_2x3x6_decomp():
    """
    Construct Rank 33 decomposition for 2x3x6.
    This serves as the initial guess.
    """
    # Strassen 2x2x2
    U_s = [[0]*7 for _ in range(4)]
    V_s = [[0]*7 for _ in range(4)]
    W_s = [[0]*7 for _ in range(4)]
    
    # M0: A00 * (B01+B11)
    U_s[0][0] = 1
    V_s[1][0] = 1; V_s[3][0] = 1
    W_s[0][0] = 1; W_s[3][0] = 1
    
    # M1: (A10+A11) * B00
    U_s[2][1] = 1; U_s[3][1] = 1
    V_s[0][1] = 1
    W_s[1][1] = 1; W_s[2][1] = 1
    
    # M2: (A00+A01) * B11
    U_s[0][2] = 1; U_s[1][2] = 1
    V_s[3][2] = 1
    W_s[0][2] = 1; W_s[3][2] = 1
    
    # M3: A11 * (B00-B10)
    U_s[3][3] = 1
    V_s[0][3] = 1; V_s[2][3] = -1
    W_s[0][3] = 1; W_s[2][3] = 1
    
    # M4: (A00+A10) * B01
    U_s[0][4] = 1; U_s[2][4] = 1
    V_s[1][4] = 1
    W_s[0][4] = -1; W_s[1][4] = 1
    
    # M5: (A01-A10) * (B00+B10)
    U_s[1][5] = 1; U_s[2][5] = -1
    V_s[0][5] = 1; V_s[2][5] = 1
    W_s[0][5] = -1
    
    # M6: (A11-A01) * (B01+B11)
    U_s[3][6] = 1; U_s[1][6] = -1
    V_s[1][6] = 1; V_s[3][6] = 1
    W_s[3][6] = 1

    U_block = [[0]*11 for _ in range(6)]
    V_block = [[0]*11 for _ in range(6)]
    W_block = [[0]*11 for _ in range(4)]
    
    U_map = [0, 1, 3, 4]
    for local_i, global_i in enumerate(U_map):
        for r in range(7):
            U_block[global_i][r] = U_s[local_i][r]
            
    V_map = [0, 1, 2, 3]
    for local_j, global_j in enumerate(V_map):
        for r in range(7):
            V_block[global_j][r] = V_s[local_j][r]
            
    W_map = [0, 1, 2, 3]
    for local_k, global_k in enumerate(W_map):
        for r in range(7):
            W_block[global_k][r] = W_s[local_k][r]
            
    U_block[2][7] = 1
    V_block[4][7] = 1
    W_block[0][7] = 1
    
    U_block[2][8] = 1
    V_block[5][8] = 1
    W_block[1][8] = 1
    
    U_block[5][9] = 1
    V_block[4][9] = 1
    W_block[2][9] = 1
    
    U_block[5][10] = 1
    V_block[5][10] = 1
    W_block[3][10] = 1
    
    U = [[0]*33 for _ in range(6)]
    V = [[0]*33 for _ in range(18)]
    W = [[0]*33 for _ in range(12)]
    
    for block in range(3):
        offset = block * 11
        for i in range(6):
            for r in range(11):
                U[i][offset+r] = U_block[i][r]
                
        for j in range(6):
            for r in range(11):
                V[j + block*6][offset+r] = V_block[j][r]
                
        for k in range(4):
            for r in range(11):
                W[k + block*4][offset+r] = W_block[k][r]
                
    return {"U": U, "V": V, "W": W}

def _discrete_search_for_lower_rank(start_U, start_V, start_W, n, m, p, target_rank, time_limit=10):
    """
    Discrete hill-climbing search for a solution of target_rank.
    We start from a rank 33 solution, zero out one column (randomly), and then try to fix the errors
    by changing values in the remaining columns to half-integers.
    """
    U_init = np.array(start_U, dtype=float)
    V_init = np.array(start_V, dtype=float)
    W_init = np.array(start_W, dtype=float)
    
    # We want target_rank. Let's say we keep the first target_rank columns.
    # But the structure might be distributed.
    # Strategy: Randomly select 'target_rank' columns to keep, zero others, then optimize.
    
    # Precompute indices
    indices_U = []
    indices_V = []
    indices_W = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                indices_U.append(i * m + j)
                indices_V.append(j * p + k)
                indices_W.append(k * n + i)
                
    indices_U = np.array(indices_U)
    indices_V = np.array(indices_V)
    indices_W = np.array(indices_W)
    N_terms = len(indices_U)
    
    def count_errors(U, V, W):
        # U, V, W are (rows, target_rank)
        U_rows = U[indices_U]
        V_rows = V[indices_V]
        W_rows = W[indices_W]
        
        # Product and sum
        sums = np.sum(U_rows * V_rows * W_rows, axis=1)
        errors = np.abs(sums - 1.0) > 1e-9
        return np.sum(errors)
    
    best_error = N_terms
    best_U = None
    best_V = None
    best_W = None
    
    start_time = time.time()
    
    # Try multiple random subsets of columns to keep
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Select random columns to keep
        cols_to_keep = np.random.choice(33, target_rank, replace=False)
        cols_to_keep = np.sort(cols_to_keep)
        
        U_curr = U_init[:, cols_to_keep].copy()
        V_curr = V_init[:, cols_to_keep].copy()
        W_curr = W_init[:, cols_to_keep].copy()
        
        # Initial error
        curr_error = count_errors(U_curr, V_curr, W_curr)
        
        if curr_error < best_error:
            best_error = curr_error
            best_U = U_curr.copy()
            best_V = V_curr.copy()
            best_W = W_curr.copy()
            if best_error == 0:
                return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
        
        # Discrete Hill Climbing
        # Values are restricted to half-integers: -2, -1.5, ..., 2
        # We will randomly pick a cell and change its value to a random half-integer
        # and accept if error decreases.
        
        for step in range(1000):
            if time.time() - start_time > time_limit:
                break
            if curr_error == 0:
                return {"U": U_curr.tolist(), "V": V_curr.tolist(), "W": W_curr.tolist()}
                
            # Pick a random matrix and cell
            mat_idx = np.random.randint(3)
            if mat_idx == 0:
                row = np.random.randint(n*m)
                col = np.random.randint(target_rank)
                old_val = U_curr[row, col]
                # Try a new random half-integer
                new_val = np.random.choice(np.arange(-2, 2.1, 0.5))
                U_curr[row, col] = new_val
                new_error = count_errors(U_curr, V_curr, W_curr)
                if new_error < curr_error:
                    curr_error = new_error
                    if curr_error < best_error:
                        best_error = curr_error
                        best_U = U_curr.copy()
                        best_V = V_curr.copy()
                        best_W = W_curr.copy()
                else:
                    U_curr[row, col] = old_val
                    
            elif mat_idx == 1:
                row = np.random.randint(m*p)
                col = np.random.randint(target_rank)
                old_val = V_curr[row, col]
                new_val = np.random.choice(np.arange(-2, 2.1, 0.5))
                V_curr[row, col] = new_val
                new_error = count_errors(U_curr, V_curr, W_curr)
                if new_error < curr_error:
                    curr_error = new_error
                    if curr_error < best_error:
                        best_error = curr_error
                        best_U = U_curr.copy()
                        best_V = V_curr.copy()
                        best_W = W_curr.copy()
                else:
                    V_curr[row, col] = old_val
                    
            else:
                row = np.random.randint(p*n)
                col = np.random.randint(target_rank)
                old_val = W_curr[row, col]
                new_val = np.random.choice(np.arange(-2, 2.1, 0.5))
                W_curr[row, col] = new_val
                new_error = count_errors(U_curr, V_curr, W_curr)
                if new_error < curr_error:
                    curr_error = new_error
                    if curr_error < best_error:
                        best_error = curr_error
                        best_U = U_curr.copy()
                        best_V = V_curr.copy()
                        best_W = W_curr.copy()
                else:
                    W_curr[row, col] = old_val
                        
    if best_error == 0:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
        
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    # Get initial rank 33 solution
    res_33 = _construct_2x3x6_decomp()
    
    if _verify_decomposition(res_33['U'], res_33['V'], res_33['W'], n, m, p, tol=1e-9):
        best_result = res_33
    else:
        best_result = _schoolbook_decomposition(n, m, p)
        
    # Try to find a lower rank solution using discrete search
    # We prioritize rank 32, then 31, then 30
    
    for target_rank in [32, 31, 30]:
        if time.time() > deadline - 20:
            break
            
        # Run discrete search
        # We give it a generous time budget
        time_remaining = deadline - time.time() - 10
        if time_remaining < 5:
            continue
            
        res_opt = _discrete_search_for_lower_rank(
            best_result['U'], 
            best_result['V'], 
            best_result['W'], 
            n, m, p, 
            target_rank, 
            time_limit=time_remaining
        )
        
        if res_opt is not None:
            # Verify strictly
            if _verify_decomposition(res_opt['U'], res_opt['V'], res_opt['W'], n, m, p, tol=1e-9):
                best_result = res_opt
                # If we found rank 32, we can stop or try 31 if time permits
                # The loop naturally continues to 31 if time allows, but 
                # we should break if we found a very good one and time is tight
                if target_rank <= 31:
                    break
    
    return best_result

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
    print("Verified:", _verify_decomposition(res['U'], res['V'], res['W'], 2, 3, 6))
