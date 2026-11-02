import numpy as np
import time
import random

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1

    # Try to find a better decomposition for 3x3x3
    # If found, use it for the first 3 columns of B and C
    # And use schoolbook for the last 2 columns
    
    # Define target tensor for 3x3x3
    n2, m2, p2 = 3, 3, 3
    T2 = np.zeros((n2*m2, m2*p2, p2*n2))
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                T2[i*m2+j, j*p2+k, k*n2+i] = 1.0
                
    # Search for rank 23 solution for 3x3x3
    R2 = 23
    found = False
    U2 = None
    V2 = None
    W2 = None
    
    start_time = time.time()
    deadline = start_time + 100 # 100 seconds for search
    
    # Random search
    best_err = float('inf')
    best_U2 = None
    best_V2 = None
    best_W2 = None
    
    for _ in range(1000):
        if time.time() > deadline:
            break
            
        # Initialize with random small integers/halves
        U2_rand = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(n2*m2, R2))
        V2_rand = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(m2*p2, R2))
        W2_rand = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(p2*n2, R2))
        
        # Local search
        U2_cur = U2_rand.copy()
        V2_cur = V2_rand.copy()
        W2_cur = W2_rand.copy()
        
        for step in range(100):
            if time.time() > deadline:
                break
                
            # Calculate error
            # T_calc = np.einsum('ir,jr,kr->ijk', U2_cur, V2_cur, W2_cur)
            # This is slow. Let's use a faster method.
            # We only need to check the non-zero entries of T2.
            # T2 has 27 non-zero entries.
            
            err = 0.0
            for i in range(n2):
                for j in range(m2):
                    for k in range(p2):
                        idx_u = i*m2+j
                        idx_v = j*p2+k
                        idx_w = k*n2+i
                        val = 0.0
                        for r in range(R2):
                            val += U2_cur[idx_u, r] * V2_cur[idx_v, r] * W2_cur[idx_w, r]
                        err += (val - 1.0)**2
                        
            if err < best_err:
                best_err = err
                best_U2 = U2_cur.copy()
                best_V2 = V2_cur.copy()
                best_W2 = W2_cur.copy()
                
            if err < 1e-9:
                found = True
                break
                
            # Perturb one element
            mat = random.choice([U2_cur, V2_cur, W2_cur])
            row = random.randint(0, mat.shape[0]-1)
            col = random.randint(0, mat.shape[1]-1)
            old_val = mat[row, col]
            # Try new values
            new_vals = [v for v in [-1.5, -1, -0.5, 0, 0.5, 1, 1.5] if v != old_val]
            random.shuffle(new_vals)
            improved = False
            for new_val in new_vals:
                mat[row, col] = new_val
                # Quick error check
                err_new = 0.0
                # Only need to check entries affected by this change
                # This is complex. Let's just recompute full error for simplicity in this short search.
                # But 100 steps * 1000 restarts * 27 entries * 23 mults is 6e6 ops. Fast.
                for i in range(n2):
                    for j in range(m2):
                        for k in range(p2):
                            idx_u = i*m2+j
                            idx_v = j*p2+k
                            idx_w = k*n2+i
                            val = 0.0
                            for r in range(R2):
                                val += U2_cur[idx_u, r] * V2_cur[idx_v, r] * W2_cur[idx_w, r]
                            err_new += (val - 1.0)**2
                            
                if err_new < err:
                    err = err_new
                    improved = True
                    break
                else:
                    mat[row, col] = old_val
                    
            if not improved:
                break
                
        if found:
            break

    if found:
        # Construct full solution
        # U2, V2, W2 are for 3x3x3
        # We need to map them to the full 3x3x5 indices
        
        R_total = R2 + (m2 * (p - p2)) # 23 + 3*2 = 29?
        # Wait, the 3x3x2 part uses schoolbook rank 18.
        # So R_total = 23 + 18 = 41.
        
        U_full = [[0] * R_total for _ in range(n * m)]
        V_full = [[0] * R_total for _ in range(m * p)]
        W_full = [[0] * R_total for _ in range(p * n)]
        
        # Fill U, V, W for the 3x3x3 part (columns 0-2 of B and C)
        for r in range(R2):
            for i in range(n2):
                for j in range(m2):
                    U_full[i*m + j][r] = best_U2[i*m2+j, r]
            for j in range(m2):
                for k in range(p2):
                    V_full[j*p + k][r] = best_V2[j*p2+k, r]
            for k in range(p2):
                for i in range(n2):
                    W_full[k*n + i][r] = best_W2[k*n2+i, r]
                    
        # Fill U, V, W for the 3x3x2 part (columns 3-4 of B and C)
        # Schoolbook for 3x3x2
        col_offset = R2
        for i in range(n2):
            for j in range(m2):
                for k in range(p2, p):
                    U_full[i*m + j][col_offset] = 1
                    V_full[j*p + k][col_offset] = 1
                    W_full[k*n + i][col_offset] = 1
                    col_offset += 1
                    
        return {"U": U_full, "V": V_full, "W": W_full}
    else:
        return {"U": U_school, "V": V_school, "W": W_school}
