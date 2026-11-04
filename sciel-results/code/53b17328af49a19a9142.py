"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook decomposition of rank n*m*p
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    # Target Rank 19
    target_rank = 19
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin (limit 120s)
    
    # Fast error calculation using NumPy
    # T_target is all ones (3,3,3)
    # T_decomp[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    def calc_error_fast(U, V, W):
        U_np = np.array(U, dtype=np.int64)
        V_np = np.array(V, dtype=np.int64)
        W_np = np.array(W, dtype=np.int64)
        
        # Reshape to (n, m, rank) etc
        U_3d = U_np.reshape(n, m, target_rank)
        V_3d = V_np.reshape(m, p, target_rank)
        W_3d = W_np.reshape(p, n, target_rank) # Note: index is k*n+i, so dims are p, n
        
        # We need sum over r of U[i,j,r] * V[j,k,r] * W[k,i,r]
        # This is a tensor network contraction.
        # We can do it in a loop over r for simplicity, or use einsum.
        # einsum('ijr,jkr,kr->ijk', U_3d, V_3d, W_3d)
        # Note: W is indexed by k, i. So W_3d[k, i, r].
        # My reshape of W_np (p*n, rank) -> (p, n, rank) gives W_3d[k, i, r].
        # This matches the index k*n+i.
        
        try:
            T_decomp = np.einsum('ijr,jkr,kr->ijk', U_3d, V_3d, W_3d, optimize=True)
        except Exception:
            # Fallback to loop
            T_decomp = np.zeros((n, m, p), dtype=np.int64)
            for r in range(target_rank):
                u_col = U_3d[:, :, r]
                v_col = V_3d[:, :, r]
                w_col = W_3d[:, :, r]
                # u_col[i,j], v_col[j,k], w_col[k,i]
                # Sum over j? No, T[i,j,k] is the coefficient.
                # T[i,j,k] += u_col[i,j] * v_col[j,k] * w_col[k,i]
                # This is element-wise multiplication and addition?
                # No, for a fixed r, the contribution to T[i,j,k] is
                # U_3d[i,j,r] * V_3d[j,k,r] * W_3d[k,i,r]
                # So T_decomp[i,j,k] = sum_r (U_3d[i,j,r] * V_3d[j,k,r] * W_3d[k,i,r])
                # This is exactly what einsum does.
                pass
        
        T_target = np.ones((n, m, p), dtype=np.int64)
        error = np.sum((T_decomp - T_target) ** 2)
        return error

    def local_search(U, V, W):
        current_error = calc_error_fast(U, V, W)
        if current_error == 0:
            return U, V, W, 0
        
        improved = True
        # Use a simple hill climbing with random restarts within the search
        while improved and time.time() < deadline:
            improved = False
            # Try a batch of random flips
            for _ in range(50):
                if time.time() > deadline: break
                
                mat_idx = random.randint(0, 2)
                row = random.randint(0, 8)
                col = random.randint(0, target_rank - 1)
                
                old_val = 0
                if mat_idx == 0:
                    old_val = U[row][col]
                    next_val = (old_val + 2) % 3 - 1
                    U[row][col] = next_val
                elif mat_idx == 1:
                    old_val = V[row][col]
                    next_val = (old_val + 2) % 3 - 1
                    V[row][col] = next_val
                else:
                    old_val = W[row][col]
                    next_val = (old_val + 2) % 3 - 1
                    W[row][col] = next_val
                    
                new_error = calc_error_fast(U, V, W)
                
                if new_error < current_error:
                    current_error = new_error
                    improved = True
                    if current_error == 0:
                        return U, V, W, 0
                else:
                    if mat_idx == 0:
                        U[row][col] = old_val
                    elif mat_idx == 1:
                        V[row][col] = old_val
                    else:
                        W[row][col] = old_val
        
        return U, V, W, current_error

    best_solution = None
    best_error = float('inf')
    
    # Run multiple restarts
    for restart in range(100):
        if time.time() > deadline:
            break
            
        # Initialize with random sparse matrices
        # High probability of 0
        def rand_int():
            r = random.random()
            if r < 0.7: return 0
            elif r < 0.85: return 1
            else: return -1
            
        U = [[rand_int() for _ in range(target_rank)] for _ in range(n * m)]
        V = [[rand_int() for _ in range(target_rank)] for _ in range(m * p)]
        W = [[rand_int() for _ in range(target_rank)] for _ in range(p * n)]
        
        U_sol, V_sol, W_sol, err = local_search(U, V, W)
        
        if err < best_error:
            best_error = err
            best_solution = (U_sol, V_sol, W_sol)
            
        if best_error == 0:
            break
            
    if best_solution and best_error == 0:
        U, V, W = best_solution
        return {"U": U, "V": V, "W": W}
    else:
        return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
