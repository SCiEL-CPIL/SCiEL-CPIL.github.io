import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with rank-56 Strassen decomposition
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_U = [
        {(0,0): 1, (1,1): 1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (0,0): -1},
        {(0,1): 1, (1,1): -1},
    ]
    strassen_V = [
        {(0,0): 1, (1,1): 1},
        {(0,0): 1},
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (0,0): -1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (1,1): 1},
    ]
    strassen_W = [
        {(0,0): 1, (1,1): 1},
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(0,0): -1, (1,0): 1},
        {(1,1): 1},
        {(0,0): 1},
    ]
    
    def add_block(base_rank, i_base, j_base, k_base):
        for r in range(7):
            cr = base_rank + r
            for (li, lj), val in strassen_U[r].items():
                gi = i_base + li
                gj = j_base + lj
                U[gi * 4 + gj][cr] += val
            for (lj, lk), val in strassen_V[r].items():
                gj = j_base + lj
                gk = k_base + lk
                V[gj * 8 + gk][cr] += val
            for (lk, li), val in strassen_W[r].items():
                gk = k_base + lk
                gi = i_base + li
                W[gk * 2 + gi][cr] += val
    
    add_block(0, 0, 0, 0)
    add_block(7, 0, 0, 2)
    add_block(14, 0, 0, 4)
    add_block(21, 0, 0, 6)
    add_block(28, 0, 2, 0)
    add_block(35, 0, 2, 2)
    add_block(42, 0, 2, 4)
    add_block(49, 0, 2, 6)
    
    # Convert to numpy for efficient tensor computation
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Precompute index mappings for vectorized tensor computation
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # Tensor[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    def compute_tensor_fast(U_arr, V_arr, W_arr, rank_val):
        """Vectorized tensor computation."""
        tensor = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_vals = U_arr[i*m+j, :rank_val]
                    v_vals = V_arr[j*p+k, :rank_val]
                    w_vals = W_arr[k*n+i, :rank_val]
                    tensor[i,j,k] = np.dot(u_vals, v_vals * w_vals)
        return tensor
    
    def is_valid(U_arr, V_arr, W_arr, rank_val):
        """Check if the decomposition exactly computes matrix multiplication."""
        tensor = compute_tensor_fast(U_arr, V_arr, W_arr, rank_val)
        return np.allclose(tensor, np.ones((n, m, p)), atol=1e-10)
    
    # Verify initial decomposition
    if not is_valid(U_np, V_np, W_np, rank):
        return {"U": U, "V": V, "W": W}
    
    best_rank = rank
    best_U = U_np.copy()
    best_V = V_np.copy()
    best_W = W_np.copy()
    
    # Greedy column removal with vectorized check
    current_rank = rank
    current_U = U_np.copy()
    current_V = V_np.copy()
    current_W = W_np.copy()
    
    improved = True
    while improved:
        improved = False
        for r in range(current_rank):
            if time.time() > deadline - 5:
                break
            
            test_U = current_U.copy()
            test_V = current_V.copy()
            test_W = current_W.copy()
            test_U[:, r] = 0
            test_V[:, r] = 0
            test_W[:, r] = 0
            
            if is_valid(test_U, test_V, test_W, current_rank):
                mask = [x for x in range(current_rank) if x != r]
                current_U = current_U[:, mask]
                current_V = current_V[:, mask]
                current_W = current_W[:, mask]
                current_rank -= 1
                improved = True
                if current_rank < best_rank:
                    best_rank = current_rank
                    best_U = current_U.copy()
                    best_V = current_V.copy()
                    best_W = current_W.copy()
                break
        
        if time.time() > deadline - 5:
            break
    
    # Local search: zero individual entries in ONE matrix at a time
    if time.time() < deadline - 10:
        import random
        random.seed(42)
        
        for trial in range(200):
            if time.time() > deadline - 5:
                break
            
            test_U = best_U.copy()
            test_V = best_V.copy()
            test_W = best_W.copy()
            test_rank = best_rank
            
            for _ in range(30):
                if time.time() > deadline - 5:
                    break
                
                # Choose which matrix to modify
                mat_choice = random.randint(0, 2)
                if mat_choice == 0:
                    row = random.randint(0, test_U.shape[0]-1)
                    col = random.randint(0, test_rank-1)
                    if test_U[row, col] == 0:
                        continue
                    old_val = test_U[row, col]
                    test_U[row, col] = 0
                    if not is_valid(test_U, test_V, test_W, test_rank):
                        test_U[row, col] = old_val
                elif mat_choice == 1:
                    row = random.randint(0, test_V.shape[0]-1)
                    col = random.randint(0, test_rank-1)
                    if test_V[row, col] == 0:
                        continue
                    old_val = test_V[row, col]
                    test_V[row, col] = 0
                    if not is_valid(test_U, test_V, test_W, test_rank):
                        test_V[row, col] = old_val
                else:
                    row = random.randint(0, test_W.shape[0]-1)
                    col = random.randint(0, test_rank-1)
                    if test_W[row, col] == 0:
                        continue
                    old_val = test_W[row, col]
                    test_W[row, col] = 0
                    if not is_valid(test_U, test_V, test_W, test_rank):
                        test_W[row, col] = old_val
            # After local search, try greedy column removal again
            cur_rank = test_rank
            cur_U = test_U.copy()
            cur_V = test_V.copy()
            cur_W = test_W.copy()
            imp = True
            while imp:
                imp = False
                for r in range(cur_rank):
                    if time.time() > deadline - 5:
                        break
                    tU = cur_U.copy()
                    tV = cur_V.copy()
                    tW = cur_W.copy()
                    tU[:, r] = 0
                    tV[:, r] = 0
                    tW[:, r] = 0
                    if is_valid(tU, tV, tW, cur_rank):
                        mask = [x for x in range(cur_rank) if x != r]
                        cur_U = cur_U[:, mask]
                        cur_V = cur_V[:, mask]
                        cur_W = cur_W[:, mask]
                        cur_rank -= 1
                        imp = True
                        break
                if time.time() > deadline - 5:
                    break
            
            if cur_rank < best_rank:
                best_rank = cur_rank
                best_U = cur_U.copy()
                best_V = cur_V.copy()
                best_W = cur_W.copy()
    
    # Final: try to round to integers/half-integers
    if best_rank < rank:
        U_final = best_U.tolist()
        V_final = best_V.tolist()
        W_final = best_W.tolist()
        
        for mat in [U_final, V_final, W_final]:
            for i in range(len(mat)):
                for j in range(len(mat[i])):
                    val = mat[i][j]
                    rounded = round(val * 2) / 2
                    if abs(val - rounded) < 1e-10:
                        mat[i][j] = rounded
        
        return {"U": U_final, "V": V_final, "W": W_final}
    else:
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
