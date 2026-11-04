import time
import numpy as np

DIMENSIONS = [2, 5, 6]
START_TIME = time.time()
TIME_LIMIT = 145.0

def construct():
    n, m, p = DIMENSIONS
    
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
    
    rank_M = np.linalg.matrix_rank(M, tol=1e-10)
    
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
    
    def verify_decomposition(U_new, V_new, W_new):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U_new[i*m+j, r] * V_new[j*p+k, r] * W_new[k*n+i, r] 
                             for r in range(U_new.shape[1]))
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    if rank_M < col:
        # More aggressive elimination using SVD to prioritize which columns to remove
        keep = list(range(col))
        deadline = START_TIME + TIME_LIMIT - 5.0
        
        # Compute column norms to prioritize removal of small-norm columns
        col_norms = np.linalg.norm(M, axis=0)
        
        for _pass in range(20):
            if time.time() > deadline:
                break
            changed = True
            while changed and time.time() < deadline:
                changed = False
                # Sort remaining columns by norm (ascending) to try removing small ones first
                sorted_keep = sorted(keep, key=lambda x: col_norms[x])
                for idx in sorted_keep:
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
            if verify_decomposition(U_new, V_new, W_new):
                return {"U": to_clean_list(U_new), "V": to_clean_list(V_new), "W": to_clean_list(W_new)}
    
    # Fallback: return original
    return {"U": to_clean_list(U_arr), "V": to_clean_list(V_arr), "W": to_clean_list(W_arr)}
