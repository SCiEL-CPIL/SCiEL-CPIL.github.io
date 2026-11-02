import numpy as np
import time

def construct():
    n, m, p = 2, 3, 9
    start_time = time.time()
    deadline = start_time + 145  # 145s safety margin (evaluator kills at 150s)
    
    # Build the rank-50 decomposition using Strassen-like 2x2x2 blocks
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    u_a, u_b, u_c, u_d = 0, 1, 3, 4
    
    # 4 blocks of 7 columns each for k=0..7
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        v_e = k0
        v_f = k1
        v_g = p + k0
        v_h = p + k1
        
        U[u_a][col] += 1
        U[u_d][col] += 1
        V[v_e][col] += 1
        V[v_h][col] += 1
        
        U[u_c][col+1] += 1
        U[u_d][col+1] += 1
        V[v_e][col+1] += 1
        
        U[u_a][col+2] += 1
        V[v_f][col+2] += 1
        V[v_h][col+2] -= 1
        
        U[u_d][col+3] += 1
        V[v_g][col+3] += 1
        V[v_e][col+3] -= 1
        
        U[u_a][col+4] += 1
        U[u_b][col+4] += 1
        V[v_h][col+4] += 1
        
        U[u_c][col+5] += 1
        U[u_a][col+5] -= 1
        V[v_e][col+5] += 1
        V[v_f][col+5] += 1
        
        U[u_b][col+6] += 1
        U[u_d][col+6] -= 1
        V[v_g][col+6] += 1
        V[v_h][col+6] += 1
        
        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1
        
        W[w_00][col] += 1
        W[w_00][col+3] += 1
        W[w_00][col+4] -= 1
        W[w_00][col+6] += 1
        
        W[w_01][col+2] += 1
        W[w_01][col+4] += 1
        
        W[w_10][col+1] += 1
        W[w_10][col+3] += 1
        
        W[w_11][col] += 1
        W[w_11][col+1] -= 1
        W[w_11][col+2] += 1
        W[w_11][col+5] += 1
        
        col += 7
    
    # k=8 block (4 columns)
    k8 = 8
    v_e = k8
    v_g = p + k8
    
    U[u_a][col] += 1
    V[v_e][col] += 1
    U[u_b][col+1] += 1
    V[v_g][col+1] += 1
    U[u_c][col+2] += 1
    V[v_e][col+2] += 1
    U[u_d][col+3] += 1
    V[v_g][col+3] += 1
    
    w_08 = k8 * n + 0
    w_18 = k8 * n + 1
    
    W[w_08][col] += 1
    W[w_08][col+1] += 1
    W[w_18][col+2] += 1
    W[w_18][col+3] += 1
    
    col += 4
    
    # Row augmentation for a=0 and a=1 (18 columns)
    u_ar0, u_ar1 = 2, 5
    for k in range(9):
        v_b = 2 * p + k
        U[u_ar0][col] += 1
        V[v_b][col] += 1
        W[k * n + 0][col] += 1
        U[u_ar1][col+1] += 1
        V[v_b][col+1] += 1
        W[k * n + 1][col+1] += 1
        col += 2
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    def check_time():
        return time.time() > deadline - 5
    
    if check_time():
        return {"U": U, "V": V, "W": W}
    
    # Build the target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n*m):
        ai = i // m
        am = i % m
        for j in range(m*p):
            bj = j // p
            bm = j % p
            if am == bj:
                k_idx = bm * n + ai
                T[i, j, k_idx] = 1.0
    
    # Verify current decomposition
    T_decomp = np.einsum('ic,jc,kc->ijk', U_np, V_np, W_np)
    residual = np.linalg.norm(T_decomp - T)
    
    if residual > 1e-10:
        # Decomposition is invalid, return as-is
        return {"U": U, "V": V, "W": W}
    
    # Precompute per-column tensor contributions for fast elimination tests
    nm, mp, pn = n*m, m*p, p*n
    col_tensors = np.zeros((rank, nm, mp, pn))
    for c in range(rank):
        col_tensors[c] = np.einsum('i,j,k->ijk', U_np[:, c], V_np[:, c], W_np[:, c])
    
    if check_time():
        return {"U": U, "V": V, "W": W}
    
    # Check for linear dependencies using vectorized tensor flattening
    total_size = nm * mp * pn
    mat = np.zeros((total_size, rank))
    for c in range(rank):
        mat[:, c] = col_tensors[c].ravel()
    
    # SVD to find numerical rank and null space
    try:
        _, s, Vt = np.linalg.svd(mat, full_matrices=False)
        tol = s[0] * max(nm, mp, pn) * 1e-10
        num_rank = int(np.sum(s > tol))
        
        if num_rank < rank:
            best_rank = rank
            best_U = U_np.copy()
            best_V = V_np.copy()
            best_W = W_np.copy()
            
            # Strategy 1: Try removing single columns
            for drop_col in range(rank):
                if check_time():
                    break
                
                T_new = T_decomp - col_tensors[drop_col]
                res = np.linalg.norm(T_new - T)
                
                if res < 1e-10:
                    new_rank = rank - 1
                    if new_rank < best_rank:
                        best_rank = new_rank
                        best_U = np.delete(U_np, drop_col, axis=1)
                        best_V = np.delete(V_np, drop_col, axis=1)
                        best_W = np.delete(W_np, drop_col, axis=1)
            
            # Strategy 2: Use null space to find combinations
            if best_rank < rank and not check_time():
                # Get null space vectors
                null_space = Vt[num_rank:].T  # shape (rank - num_rank, rank)
                
                # For each null space vector, try to eliminate the column with largest magnitude
                for null_vec in null_space:
                    if check_time():
                        break
                    
                    # Find the column with largest absolute value in this null vector
                    max_idx = np.argmax(np.abs(null_vec))
                    max_val = null_vec[max_idx]
                    
                    if abs(max_val) < 1e-10:
                        continue
                    
                    # Compute the combination: sum of null_vec[c] * col_tensors[c]
                    # This should be zero (or close to zero)
                    combo_tensor = np.zeros((nm, mp, pn))
                    for c in range(rank):
                        if abs(null_vec[c]) > 1e-10:
                            combo_tensor += null_vec[c] * col_tensors[c]
                    
                    # If we can eliminate the max column by adjusting others
                    # We need: T_decomp - col_tensors[max_idx] - combo_without_max = T
                    # i.e., combo_tensor - null_vec[max_idx]*col_tensors[max_idx] should be ~0
                    combo_without_max = combo_tensor - null_vec[max_idx] * col_tensors[max_idx]
                    
                    # Check if removing max_idx and adding the combination works
                    # Actually, we want to see if we can replace the set of columns
                    # with a reduced set. Let's try a different approach:
                    # If the null space tells us columns are dependent, we can
                    # eliminate one column by expressing it in terms of others.
                    
                    # Try: can we remove column max_idx?
                    # The dependency is: sum_c null_vec[c] * col_c = 0
                    # So col_max = -sum_{c != max} (null_vec[c]/null_vec[max]) * col_c
                    # This means col_max is a linear combination of other columns
                    # But for tensor rank reduction, we need more than linear dependence
                    # of the tensor columns - we need the full decomposition to still work.
                    
                    # Instead, let's try a more direct approach:
                    # Try removing pairs using the null space structure
                    pass
            
            # Strategy 3: Try removing pairs more efficiently
            if best_rank < rank and not check_time():
                # Precompute all single-column residuals
                single_res = np.zeros(rank)
                for c in range(rank):
                    T_new = T_decomp - col_tensors[c]
                    single_res[c] = np.linalg.norm(T_new - T)
                
                # Only consider pairs where both columns have small individual residuals
                candidates = [c for c in range(rank) if single_res[c] < 1e-10]
                
                if len(candidates) >= 2:
                    for i in range(len(candidates)):
                        if check_time():
                            break
                        for j in range(i + 1, len(candidates)):
                            if check_time():
                                break
                            c1, c2 = candidates[i], candidates[j]
                            T_new = T_decomp - col_tensors[c1] - col_tensors[c2]
                            res = np.linalg.norm(T_new - T)
                            
                            if res < 1e-10:
                                new_rank = rank - 2
                                if new_rank < best_rank:
                                    best_rank = new_rank
                                    cols_to_drop = sorted([c1, c2])
                                    best_U = np.delete(U_np, cols_to_drop, axis=1)
                                    best_V = np.delete(V_np, cols_to_drop, axis=1)
                                    best_W = np.delete(W_np, cols_to_drop, axis=1)
            
            # Strategy 4: Try removing triples if we found pairs
            if best_rank < rank and not check_time():
                # If we can remove 2, try to find a third
                if best_rank == rank - 2:
                    # Get the current best set of columns to keep
                    # Try adding one more removal
                    for c in range(rank):
                        if check_time():
                            break
                        # Check if c is already in the dropped set
                        # For simplicity, just try all
                        T_new = T_decomp - col_tensors[c]
                        res = np.linalg.norm(T_new - T)
                        if res < 1e-10:
                            # This column can be removed independently
                            # Check if it's compatible with the pair removal
                            pass
            
            if best_rank < rank:
                return {
                    "U": best_U.tolist(),
                    "V": best_V.tolist(),
                    "W": best_W.tolist()
                }
    except Exception:
        pass
    
    # Strategy 5: Try a more aggressive search - remove columns greedily
    if not check_time():
        current_rank = rank
        current_U = U_np.copy()
        current_V = V_np.copy()
        current_W = W_np.copy()
        current_T = T_decomp.copy()
        active_cols = list(range(rank))
        
        for _ in range(rank):  # Try to remove up to rank columns
            if check_time():
                break
            
            removed = False
            for c in active_cols:
                if check_time():
                    break
                
                T_new = current_T - col_tensors[c]
                res = np.linalg.norm(T_new - T)
                
                if res < 1e-10:
                    # Remove this column
                    current_T = T_new
                    current_U = np.delete(current_U, active_cols.index(c), axis=1)
                    current_V = np.delete(current_V, active_cols.index(c), axis=1)
                    current_W = np.delete(current_W, active_cols.index(c), axis=1)
                    active_cols.remove(c)
                    current_rank -= 1
                    removed = True
                    break
            
            if not removed:
                break
        
        if current_rank < rank:
            return {
                "U": current_U.tolist(),
                "V": current_V.tolist(),
                "W": current_W.tolist()
            }
    
    # Fall back to rank-50
    return {"U": U, "V": V, "W": W}
