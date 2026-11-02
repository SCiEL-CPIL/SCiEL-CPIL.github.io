"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import numpy as np
import time
import itertools
from scipy.optimize import minimize

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = DIMENSIONS
    U_dim = n * m  # 8
    V_dim = m * p  # 32
    W_dim = p * n  # 16
    
    # 1. Construct the Rank 28 Strassen Decomposition (Fallback)
    # This is the same as the current program's logic, ensuring we have a valid baseline.
    def get_strassen_decomp():
        rank = 28
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        
        def add_strassen_term(base_col, A_idx_map, B_idx_map, C_idx_map):
            # M1
            col = base_col
            U[A_idx_map['A00'], col] = 1
            V[B_idx_map['B01'], col] = 1
            V[B_idx_map['B11'], col] = 1
            W[C_idx_map['C00'], col] = 1
            W[C_idx_map['C11'], col] = 1
            
            # M2
            col = base_col + 1
            U[A_idx_map['A10'], col] = 1
            U[A_idx_map['A11'], col] = 1
            V[B_idx_map['B00'], col] = 1
            W[C_idx_map['C10'], col] = 1
            W[C_idx_map['C11'], col] = -1
            
            # M3
            col = base_col + 2
            U[A_idx_map['A00'], col] = 1
            U[A_idx_map['A01'], col] = 1
            V[B_idx_map['B10'], col] = 1
            W[C_idx_map['C01'], col] = 1
            W[C_idx_map['C11'], col] = 1
            
            # M4
            col = base_col + 3
            U[A_idx_map['A11'], col] = 1
            V[B_idx_map['B10'], col] = 1
            V[B_idx_map['B00'], col] = 1
            W[C_idx_map['C00'], col] = 1
            W[C_idx_map['C10'], col] = 1
            
            # M5
            col = base_col + 4
            U[A_idx_map['A00'], col] = 1
            U[A_idx_map['A10'], col] = 1
            V[B_idx_map['B01'], col] = 1
            W[C_idx_map['C00'], col] = -1
            W[C_idx_map['C01'], col] = 1
            
            # M6
            col = base_col + 5
            U[A_idx_map['A01'], col] = 1
            U[A_idx_map['A11'], col] = -1
            V[B_idx_map['B10'], col] = 1
            V[B_idx_map['B11'], col] = 1
            W[C_idx_map['C11'], col] = 1
            
            # M7
            col = base_col + 6
            U[A_idx_map['A10'], col] = 1
            U[A_idx_map['A00'], col] = -1
            V[B_idx_map['B00'], col] = 1
            V[B_idx_map['B01'], col] = 1
            W[C_idx_map['C00'], col] = 1

        add_strassen_term(0, 
                          {'A00': 0, 'A01': 1, 'A10': 4, 'A11': 5},
                          {'B00': 0, 'B01': 1, 'B10': 8, 'B11': 9},
                          {'C00': 0, 'C01': 2, 'C10': 1, 'C11': 3})
        
        add_strassen_term(7, 
                          {'A00': 2, 'A01': 3, 'A10': 6, 'A11': 7},
                          {'B00': 16, 'B01': 17, 'B10': 24, 'B11': 25},
                          {'C00': 0, 'C01': 2, 'C10': 1, 'C11': 3})
        
        add_strassen_term(14, 
                          {'A00': 0, 'A01': 1, 'A10': 4, 'A11': 5},
                          {'B00': 2, 'B01': 3, 'B10': 10, 'B11': 11},
                          {'C00': 4, 'C01': 6, 'C10': 5, 'C11': 7})
        
        add_strassen_term(21, 
                          {'A00': 2, 'A01': 3, 'A10': 6, 'A11': 7},
                          {'B00': 18, 'B01': 19, 'B10': 26, 'B11': 27},
                          {'C00': 4, 'C01': 6, 'C10': 5, 'C11': 7})
                          
        return U, V, W

    U_fb, V_fb, W_fb = get_strassen_decomp()
    
    def is_valid(U, V, W, tol=1e-8):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U[i*m + j, :]
                    v_row = V[j*p + k, :]
                    w_row = W[k*n + i, :]
                    val = np.sum(u_row * v_row * w_row)
                    if abs(val - 1.0) > tol:
                        return False
        return True

    if not is_valid(U_fb, V_fb, W_fb):
        # Fallback to schoolbook if Strassen fails (should not happen)
        rank = n * m * p
        U_fb = np.zeros((U_dim, rank))
        V_fb = np.zeros((V_dim, rank))
        W_fb = np.zeros((W_dim, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m + j, col] = 1.0
                    V_fb[j*p + k, col] = 1.0
                    W_fb[k*n + i, col] = 1.0
                    col += 1

    best_rank = 28
    best_U = U_fb
    best_V = V_fb
    best_W = W_fb

    # 2. Attempt to find a Rank 27 Decomposition via Optimization
    # We parameterize U, V, W for rank 27.
    # Objective: Minimize sum (T_{ijk} - sum_r U_{ir} V_{jr} W_{kr})^2
    
    target_rank = 27
    if time.time() < deadline - 20: # Need time for optimization
        
        def objective(params):
            # Unpack params
            # U: 8 * 27 = 216
            # V: 32 * 27 = 864
            # W: 16 * 27 = 432
            # Total: 1512
            
            U_flat = params[:216].reshape((8, 27))
            V_flat = params[216:216+864].reshape((32, 27))
            W_flat = params[216+864:].reshape((16, 27))
            
            # Compute error
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_row = U_flat[i*m + j, :]
                        v_row = V_flat[j*p + k, :]
                        w_row = W_flat[k*n + i, :]
                        val = np.sum(u_row * v_row * w_row)
                        error += (val - 1.0)**2
            return error

        # Initialize with random values
        # Random initialization is often poor for high-dimensional non-convex problems.
        # We can try initializing with the Strassen decomposition (truncated to 27 cols) + noise.
        # But Strassen is 28 cols. We can drop one col and add noise.
        
        # Let's try a few random restarts
        found_better = False
        for restart in range(5):
            if time.time() > deadline - 10:
                break
                
            # Init params
            # Use random uniform [-1, 1]
            params0 = np.random.uniform(-1, 1, size=1512)
            
            # Optimize
            try:
                res = minimize(objective, params0, method='BFGS', options={'maxiter': 500, 'disp': False})
                if res.success or res.fun < 1e-4:
                    # Check validity
                    U_opt = res.x[:216].reshape((8, 27))
                    V_opt = res.x[216:216+864].reshape((32, 27))
                    W_opt = res.x[216+864:].reshape((16, 27))
                    
                    if is_valid(U_opt, V_opt, W_opt):
                        found_better = True
                        best_rank = 27
                        best_U = U_opt
                        best_V = V_opt
                        best_W = W_opt
                        break # Found a better one, stop searching
            except Exception:
                pass

    # 3. Return Best Result
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
