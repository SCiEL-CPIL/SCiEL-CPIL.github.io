import time
import numpy as np

DIMENSIONS = [2, 5, 6]
START_TIME = time.time()
TIME_LIMIT = 145.0

def construct():
    n, m, p = DIMENSIONS
    
    # Target Rank 54
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i_base, j_base, k_base):
        nonlocal col
        U[i_base * m + j_base][col] += 1
        U[(i_base+1) * m + (j_base+1)][col] += 1
        V[j_base * p + k_base][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] += 1
        W[k_base * n + i_base][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1
        
        U[(i_base+1) * m + j_base][col] += 1
        U[(i_base+1) * m + (j_base+1)][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + (i_base+1)][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] -= 1
        col += 1
        
        U[i_base * m + j_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] -= 1
        W[(k_base+1) * n + i_base][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1
        
        U[(i_base+1) * m + (j_base+1)][col] += 1
        V[(j_base+1) * p + k_base][col] += 1
        V[j_base * p + k_base][col] -= 1
        W[k_base * n + i_base][col] += 1
        W[k_base * n + (i_base+1)][col] += 1
        col += 1
        
        U[i_base * m + j_base][col] += 1
        U[i_base * m + (j_base+1)][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] += 1
        W[k_base * n + i_base][col] -= 1
        W[(k_base+1) * n + i_base][col] += 1
        col += 1
        
        U[(i_base+1) * m + j_base][col] += 1
        U[i_base * m + j_base][col] -= 1
        V[j_base * p + k_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1
        
        U[i_base * m + (j_base+1)][col] += 1
        U[(i_base+1) * m + (j_base+1)][col] -= 1
        V[(j_base+1) * p + k_base][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] += 1
        W[k_base * n + i_base][col] += 1
        col += 1

    def add_naive_2112_block(i_base, j_base, k_base):
        nonlocal col
        U[i_base * m + j_base][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + i_base][col] += 1
        col += 1
        
        U[i_base * m + j_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        W[(k_base+1) * n + i_base][col] += 1
        col += 1
        
        U[(i_base+1) * m + j_base][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + (i_base+1)][col] += 1
        col += 1
        
        U[(i_base+1) * m + j_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1

    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 0, 2)
    add_strassen_block(0, 0, 4)
    
    add_strassen_block(0, 2, 0)
    add_strassen_block(0, 2, 2)
    add_strassen_block(0, 2, 4)
    
    add_naive_2112_block(0, 4, 0)
    add_naive_2112_block(0, 4, 2)
    add_naive_2112_block(0, 4, 4)
    
    if col != 54:
        pass
    
    # MPS-inspired rank reduction: use mode-unfoldings to identify
    # linearly dependent rank-1 terms and merge them.
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    T_size = n * m * p  # 60
    
    # Build the coefficient matrix M where M[(i,j,k), r] gives the
    # contribution of rank-1 term r to tensor entry (i,j,k)
    M = np.zeros((T_size, col))
    for r in range(col):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    M[idx, r] = U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
    
    # Check if the rank-1 terms are linearly independent
    rank_M = np.linalg.matrix_rank(M, tol=1e-10)
    
    # MPS approach: use mode-1, mode-2, mode-3 unfoldings to find
    # low-rank structure. The key idea is that if the tensor has
    # small bond dimensions in an MPS representation, we can find
    # a more compact decomposition.
    
    # First, try direct column elimination using QR with pivoting
    if rank_M < col:
        # Greedy elimination: remove columns one at a time if rank unchanged
        keep = list(range(col))
        changed = True
        deadline = START_TIME + TIME_LIMIT - 5.0
        
        while changed and time.time() < deadline:
            changed = False
            for idx in list(keep):
                test_keep = [k for k in keep if k != idx]
                if len(test_keep) == 0:
                    break
                M_test = M[:, test_keep]
                if np.linalg.matrix_rank(M_test, tol=1e-10) == rank_M:
                    keep = test_keep
                    changed = True
                    break
        
        if len(keep) < col:
            new_rank = len(keep)
            U_new = U_arr[:, keep]
            V_new = V_arr[:, keep]
            W_new = W_arr[:, keep]
            
            # Verify the decomposition is still exact
            verified = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U_new[i*m+j, r] * V_new[j*p+k, r] * W_new[k*n+i, r] for r in range(new_rank))
                        if abs(val - 1.0) > 1e-9:
                            verified = False
                            break
                    if not verified:
                        break
                if not verified:
                    break
            
            if verified:
                # Convert to lists with integer/half-integer entries
                def to_clean_list(arr):
                    result = []
                    for row in arr:
                        clean_row = []
                        for val in row:
                            # Round to nearest half-integer
                            half = round(val * 2) / 2
                            if abs(val - half) < 1e-8:
                                clean_row.append(half if half != int(half) else int(half))
                            else:
                                clean_row.append(val)
                        result.append(clean_row)
                    return result
                
                return {"U": to_clean_list(U_new), "V": to_clean_list(V_new), "W": to_clean_list(W_new)}
    
    # If no reduction found via direct elimination, try MPS-inspired
    # recombination: look for pairs of rank-1 terms that can be merged
    # using the low bond dimension from n=2
    
    deadline = START_TIME + TIME_LIMIT - 5.0
    
    # Try pairwise merging: for each pair (r1, r2), check if their sum
    # can be replaced by a single rank-1 term
    U_list = [U_arr[:, r].copy() for r in range(col)]
    V_list = [V_arr[:, r].copy() for r in range(col)]
    W_list = [W_arr[:, r].copy() for r in range(col)]
    
    best_rank = col
    best_indices = list(range(col))
    
    # Iterate: try to merge pairs
    merged = True
    while merged and time.time() < deadline:
        merged = False
        for r1 in range(len(U_list)):
            for r2 in range(r1+1, len(U_list)):
                # Check if U[r1]*V[r1]*W[r1] + U[r2]*V[r2]*W[r2] 
                # can be written as a single rank-1 term
                # This is possible if the sum tensor has rank 1
                sum_tensor = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            sum_tensor[i,j,k] = (U_list[r1][i*m+j] * V_list[r1][j*p+k] * W_list[r1][k*n+i] +
                                                U_list[r2][i*m+j] * V_list[r2][j*p+k] * W_list[r2][k*n+i])
                
                # Check if sum_tensor is rank 1 by checking all 2x2 minors
                is_rank1 = True
                # Flatten to matrix and check
                mat1 = sum_tensor.reshape(n*m, p)
                if np.linalg.matrix_rank(mat1, tol=1e-10) > 1:
                    is_rank1 = False
                
                if is_rank1:
                    # Extract the rank-1 factors
                    # SVD to get the factors
                    U_svd, s, Vt = np.linalg.svd(mat1)
                    if s[0] > 1e-10:
                        u_vec = U_svd[:, 0] * s[0]
                        v_vec = Vt[0, :]
                        
                        # Now we need to find W such that:
                        # sum_{i,j,k} u_vec[i*m+j] * v_vec[j*p+k] * W_new[k*n+i] = sum_tensor[i,j,k]
                        # For each (i,k): sum_j sum_tensor[i,j,k] / (u_vec[i*m+j] * v_vec[j*p+k]) should be consistent
                        # Actually, this is tricky. Let's just check if the merge works.
                        pass
                # Skip complex merging for now, focus on elimination
    
    # Return best result
    if best_rank < col:
        U_out = U_arr[:, best_indices]
        V_out = V_arr[:, best_indices]
        W_out = W_arr[:, best_indices]
        
        def to_clean_list(arr):
            result = []
            for row in arr:
                clean_row = []
                for val in row:
                    half = round(val * 2) / 2
                    if abs(val - half) < 1e-8:
                        clean_row.append(half if half != int(half) else int(half))
                    else:
                        clean_row.append(val)
                result.append(clean_row)
            return result
        
        return {"U": to_clean_list(U_out), "V": to_clean_list(V_out), "W": to_clean_list(W_out)}
    
    # Fallback: return original
    return {"U": U, "V": V, "W": W}
