import numpy as np
import time
import random

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_school
    best_decomp = {"U": U_school, "V": V_school, "W": W_school}
    
    # Try to find lower rank decompositions
    # Target ranks: 21, 22, ... down to 18
    for R in range(21, 17, -1):
        if time.time() > deadline:
            break
            
        # Try multiple random initializations for U and V
        for attempt in range(50):
            if time.time() > deadline:
                break
                
            # Random sparse U and V
            U = np.zeros((n * m, R), dtype=float)
            V = np.zeros((m * p, R), dtype=float)
            
            # Set some entries to 1 or -1
            for c in range(R):
                # Choose random rows for U
                idx_u = random.sample(range(n * m), min(n * m, 2))
                for idx in idx_u:
                    U[idx, c] = random.choice([-1, 1])
                
                # Choose random rows for V
                idx_v = random.sample(range(m * p), min(m * p, 2))
                for idx in idx_v:
                    V[idx, c] = random.choice([-1, 1])
            
            # Build linear system for W
            # T_ijk = sum_c U_im+j,c V_jp+k,c W_kn+i,c
            # For fixed U, V, this is linear in W.
            # We have n*m*p equations.
            # Variables are W_kn+i,c for k in p, i in n, c in R.
            # Total variables: p * n * R.
            
            A = np.zeros((n * m * p, p * n * R))
            b = np.zeros(n * m * p)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Equation for T_ijk
                        # Target: 1 if (i,j,k) is part of the identity? 
                        # No, the tensor for matrix multiplication is:
                        # (A B)_{i,k} = sum_j A_{i,j} B_{j,k}
                        # The tensor T has T_{i,j,k} = 1.
                        # The decomposition is T_{i,j,k} = sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c}
                        # Note the indices in W: W[k*n + i, c]
                        
                        b[eq_idx] = 1.0
                        
                        for c in range(R):
                            u_val = U[i * m + j, c]
                            v_val = V[j * p + k, c]
                            if u_val != 0 and v_val != 0:
                                # Coefficient for W[k*n + i, c]
                                w_idx = k * n + i
                                var_idx = w_idx * R + c
                                A[eq_idx, var_idx] = u_val * v_val
                        eq_idx += 1
            
            # Solve A W_vec = b
            # Use least squares or pinv
            try:
                W_vec, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
                
            # Check if solution is close to integer or half-integer
            # And if residuals are small
            if residuals and residuals[0] < 1e-9:
                # Check integrality
                is_valid = True
                W_mat = np.zeros((p * n, R), dtype=float)
                for w_idx in range(p * n):
                    for c in range(R):
                        val = W_vec[w_idx * R + c]
                        # Check if close to integer or half-integer
                        nearest_int = round(val)
                        nearest_half = round(val * 2) / 2
                        if abs(val - nearest_int) < 1e-6:
                            W_mat[w_idx, c] = nearest_int
                        elif abs(val - nearest_half) < 1e-6:
                            W_mat[w_idx, c] = nearest_half
                        else:
                            is_valid = False
                            break
                    if not is_valid:
                        break
                
                if is_valid:
                    # Verify exact decomposition
                    U_int = U.astype(int)
                    V_int = V.astype(int)
                    W_int = W_mat
                    
                    valid = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = 0
                                for c in range(R):
                                    s += U_int[i * m + j, c] * V_int[j * p + k, c] * W_int[k * n + i, c]
                                if s != 1:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    
                    if valid:
                        # Convert to lists of ints/floats
                        U_list = U_int.tolist()
                        V_list = V_int.tolist()
                        W_list = W_int.tolist()
                        # Ensure types are correct (int or float)
                        U_out = [[int(x) if x == int(x) else float(x) for x in row] for row in U_list]
                        V_out = [[int(x) if x == int(x) else float(x) for x in row] for row in V_list]
                        W_out = [[int(x) if x == int(x) else float(x) for x in row] for row in W_list]
                        
                        return {"U": U_out, "V": V_out, "W": W_out}
                        
    return best_decomp
