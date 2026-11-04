"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(i_start, j_start, k_start, offset):
        col = offset
        U[(i_start+0)*m + (j_start+0)][col] += 1
        U[(i_start+1)*m + (j_start+1)][col] += 1
        V[(j_start+0)*p + (k_start+0)][col] += 1
        V[(j_start+1)*p + (k_start+1)][col] += 1
        W[(k_start+0)*n + (i_start+0)][col] += 1
        W[(k_start+1)*n + (i_start+1)][col] += 1
        
        col = offset + 1
        U[(i_start+1)*m + (j_start+0)][col] += 1
        U[(i_start+1)*m + (j_start+1)][col] += 1
        V[(j_start+0)*p + (k_start+0)][col] += 1
        W[(k_start+0)*n + (i_start+1)][col] += 1
        W[(k_start+1)*n + (i_start+1)][col] -= 1
        
        col = offset + 2
        U[(i_start+0)*m + (j_start+0)][col] += 1
        V[(j_start+0)*p + (k_start+1)][col] += 1
        V[(j_start+1)*p + (k_start+1)][col] -= 1
        W[(k_start+1)*n + (i_start+0)][col] += 1
        W[(k_start+1)*n + (i_start+1)][col] += 1
        
        col = offset + 3
        U[(i_start+1)*m + (j_start+1)][col] += 1
        V[(j_start+1)*p + (k_start+0)][col] += 1
        V[(j_start+0)*p + (k_start+0)][col] -= 1
        W[(k_start+0)*n + (i_start+0)][col] += 1
        W[(k_start+0)*n + (i_start+1)][col] += 1
        
        col = offset + 4
        U[(i_start+0)*m + (j_start+0)][col] += 1
        U[(i_start+0)*m + (j_start+1)][col] += 1
        V[(j_start+1)*p + (k_start+1)][col] += 1
        W[(k_start+0)*n + (i_start+0)][col] -= 1
        W[(k_start+1)*n + (i_start+0)][col] += 1
        
        col = offset + 5
        U[(i_start+1)*m + (j_start+0)][col] += 1
        U[(i_start+0)*m + (j_start+0)][col] -= 1
        V[(j_start+0)*p + (k_start+0)][col] += 1
        V[(j_start+0)*p + (k_start+1)][col] += 1
        W[(k_start+1)*n + (i_start+1)][col] += 1
        
        col = offset + 6
        U[(i_start+0)*m + (j_start+1)][col] += 1
        U[(i_start+1)*m + (j_start+1)][col] -= 1
        V[(j_start+1)*p + (k_start+0)][col] += 1
        V[(j_start+1)*p + (k_start+1)][col] += 1
        W[(k_start+0)*n + (i_start+0)][col] += 1

    for j_block in range(2):
        for k_block in range(3):
            add_strassen_block(0, j_block*2, k_block*2, (j_block*3 + k_block)*7)
    
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Verify rank-42
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                if abs(s - 1.0) > 1e-9:
                    break
    
    # Try to reduce rank by finding linear dependencies
    cur_U, cur_V, cur_W = U, V, W
    cur_rank = rank
    
    for _ in range(10):
        if time.time() > deadline - 5:
            break
        
        U_c = np.array(cur_U, dtype=np.float64)
        V_c = np.array(cur_V, dtype=np.float64)
        W_c = np.array(cur_W, dtype=np.float64)
        
        # Build tensor matrix M (48 x cur_rank)
        M = np.zeros((n*m*p, cur_rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    row_idx = (i*m + j) * p + k
                    for r in range(cur_rank):
                        M[row_idx, r] = U_c[i*m+j, r] * V_c[j*p+k, r] * W_c[k*n+i, r]
        
        try:
            s_vals = np.linalg.svd(M, compute_uv=False)
            tol = 1e-10
            eff_rank = np.sum(s_vals > tol)
        except np.linalg.LinAlgError:
            break
        
        if eff_rank >= cur_rank:
            break
        
        # Find null vector
        _, _, Vt = np.linalg.svd(M, full_matrices=False)
        null_vec = Vt[-1]
        
        # Check if any entry is close to zero
        drop_idx = -1
        for r in range(cur_rank):
            if abs(null_vec[r]) < 1e-10:
                drop_idx = r
                break
        
        if drop_idx == -1:
            # No exact zero, try to find a combination
            # Use the null vector to express one term in terms of others
            # Pick the term with smallest |null_vec[r]| to drop
            drop_idx = int(np.argmin(np.abs(null_vec)))
            
            # We need: T_drop = sum_{r != drop} c_r * T_r
            # where c_r = -null_vec[r] / null_vec[drop_idx]
            # But this changes the tensor, not just the representation.
            # The total tensor is unchanged if we replace T_drop with its expression.
            # But we can't express (1+c_r)*T_r as a single rank-1 term unless
            # we scale one of the factors.
            
            # Scale U: new_U_s = (1 + c_s) * U_s for s != drop
            # This gives: sum_{s != drop} (1+c_s) * T_s = sum_{s != drop} T_s + sum_{s != drop} c_s * T_s
            # = sum_{s != drop} T_s + T_drop (since sum c_s * T_s = T_drop from null vec)
            # = Total tensor ✓
            
            new_U = []
            new_V = []
            new_W = []
            for s in range(cur_rank):
                if s == drop_idx:
                    continue
                # c_s = -null_vec[s] / null_vec[drop_idx]
                c_s = -null_vec[s] / null_vec[drop_idx]
                scale = 1.0 + c_s
                new_U.append([U_c[i, s] * scale for i in range(n*m)])
                new_V.append([V_c[i, s] for i in range(m*p)])
                new_W.append([W_c[i, s] for i in range(p*n)])
            
            # Verify
            U_new = np.array(new_U, dtype=np.float64).T
            V_new = np.array(new_V, dtype=np.float64).T
            W_new = np.array(new_W, dtype=np.float64).T
            
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s_val = np.dot(U_new[i*m+j], V_new[j*p+k] * W_new[k*n+i])
                        if abs(s_val - 1.0) > 1e-8:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                cur_U = new_U
                cur_V = new_V
                cur_W = new_W
                cur_rank -= 1
            else:
                break
        else:
            # Exact zero entry, just drop the term
            cur_U = [row[:drop_idx] + row[drop_idx+1:] for row in cur_U]
            cur_V = [row[:drop_idx] + row[drop_idx+1:] for row in cur_V]
            cur_W = [row[:drop_idx] + row[drop_idx+1:] for row in cur_W]
            cur_rank -= 1
    
    if cur_rank < rank:
        return {"U": cur_U, "V": cur_V, "W": cur_W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
