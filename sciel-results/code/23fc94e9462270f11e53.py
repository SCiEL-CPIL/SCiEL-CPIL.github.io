import numpy as np
import time
import random

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    # Build target tensor T
    T = np.zeros((dim_U, dim_V, dim_W), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    # Schoolbook solution
    schoolbook_rank = n * m * p
    U_sb = np.zeros((dim_U, schoolbook_rank), dtype=int)
    V_sb = np.zeros((dim_V, schoolbook_rank), dtype=int)
    W_sb = np.zeros((dim_W, schoolbook_rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_rank = schoolbook_rank
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb

    start_time = time.time()
    time_limit = 140.0 # Safety margin
    
    # Try ranks from 19 up to schoolbook_rank - 1
    for target_rank in range(19, schoolbook_rank):
        if time.time() - start_time > time_limit:
            break
            
        # Try to find a solution of this rank
        found = False
        restarts = 0
        max_restarts = 50 # Adjust based on time
        
        while not found and (time.time() - start_time) < time_limit:
            restarts += 1
            
            # Initialize with random small integers
            # Using -1, 0, 1 is often effective for these small tensors
            U = np.random.randint(-1, 2, size=(dim_U, target_rank))
            V = np.random.randint(-1, 2, size=(dim_V, target_rank))
            W = np.random.randint(-1, 2, size=(dim_W, target_rank))
            
            # Local Search
            # Calculate initial error
            # Efficient calculation:
            # T_calc[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # We can compute this as:
            # Temp[a, r] = U[a, r]
            # Temp2[a, b, r] = Temp[a, r] * V[b, r] -> (dim_U, dim_V, rank)
            # T_calc = Temp2 @ W.T -> (dim_U, dim_V, dim_W)
            
            def calc_error(U, V, W):
                # (dim_U, rank) * (rank, dim_V) -> (dim_U, dim_V, rank) ? No.
                # U @ V.T is (dim_U, dim_V) per rank? No.
                # Sum_r U[a,r] V[b,r] W[c,r]
                # Let M_ab_r = U[a,r] V[b,r]. Shape (dim_U, dim_V, rank).
                # Then sum_r M_ab_r W[c,r].
                # This is a tensor contraction.
                # np.einsum('ar,br,cr->abc', U, V, W)
                T_calc = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
                return np.sum((T_calc - T) ** 2)

            err = calc_error(U, V, W)
            
            # Coordinate descent
            no_improve_count = 0
            max_no_improve = 200
            while no_improve_count < max_no_improve and (time.time() - start_time) < time_limit:
                # Pick random matrix and index
                mat_idx = random.randint(0, 2)
                if mat_idx == 0:
                    a, r = np.random.randint(dim_U, target_rank)
                    old_val = U[a, r]
                    for new_val in range(-2, 3):
                        if new_val == old_val: continue
                        U[a, r] = new_val
                        new_err = calc_error(U, V, W)
                        if new_err < err:
                            err = new_err
                            no_improve_count = 0
                            if err == 0:
                                found = True
                                break
                        else:
                            U[a, r] = old_val
                elif mat_idx == 1:
                    b, r = np.random.randint(dim_V, target_rank)
                    old_val = V[b, r]
                    for new_val in range(-2, 3):
                        if new_val == old_val: continue
                        V[b, r] = new_val
                        new_err = calc_error(U, V, W)
                        if new_err < err:
                            err = new_err
                            no_improve_count = 0
                            if err == 0:
                                found = True
                                break
                        else:
                            V[b, r] = old_val
                else:
                    c, r = np.random.randint(dim_W, target_rank)
                    old_val = W[c, r]
                    for new_val in range(-2, 3):
                        if new_val == old_val: continue
                        W[c, r] = new_val
                        new_err = calc_error(U, V, W)
                        if new_err < err:
                            err = new_err
                            no_improve_count = 0
                            if err == 0:
                                found = True
                                break
                        else:
                            W[c, r] = old_val
                            
                if found:
                    break
                no_improve_count += 1
                
            if found:
                best_rank = target_rank
                best_U = U
                best_V = V
                best_W = W
                break # Found minimal rank (since we go up from 19)

    # Convert to list of lists if required by interface? 
    # "factor matrices of shapes ... with integer/half-integer entries"
    # The seed returns lists of lists. I should return lists of lists to be safe, 
    # or numpy arrays if the evaluator accepts them. 
    # The seed: return {"U": U_sb, ...} where U_sb is list of lists.
    # I will convert numpy arrays to lists.
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
