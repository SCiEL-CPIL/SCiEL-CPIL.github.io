import numpy as np
import random
import time

DIMENSIONS = [2, 3, 6]

def verify_decomp(U, V, W, n, m, p, rank):
    """
    Verifies if U, V, W form a valid decomposition for M_{n,m,p}.
    Equation: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k.
    """
    # Check dimensions
    if U.shape != (n*m, rank): return False
    if V.shape != (m*p, rank): return False
    if W.shape != (p*n, rank): return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
                if not np.isclose(val, 1.0):
                    return False
    return True

def solve_m232(rank):
    """
    Try to find a rank decomposition for M_{2,3,2}.
    n=2, m=3, p=2.
    U: 6xR, V: 6xR, W: 4xR.
    Equations: 12 equations.
    """
    n, m, p = 2, 3, 2
    # Indices
    # i in 0,1; j in 0,1,2; k in 0,1
    # Eq index e
    # val(e) = sum_c U[3*i+j, c] V[2*j+k, c] W[2*k+i, c] - 1
    
    # Try to find integer solution
    # Use alternating least squares with restarts
    for _ in range(200): # restarts
        # Initialize with random integers
        U = np.random.randint(-1, 2, (6, rank))
        V = np.random.randint(-1, 2, (6, rank))
        W = np.random.randint(-1, 2, (4, rank))
        
        # Ensure some are non-zero to avoid trivial 0=1
        if np.all(U==0) or np.all(V==0) or np.all(W==0):
            continue
            
        converged = False
        for step in range(100):
            # Update U: fix V, W. Linear system for U
            # sum_c V[...] W[...] U[...] = 1
            # For each (i,j,k), we have an equation.
            # The variable is U_{i m + j, c}.
            # The coefficient for U_{idx, c} in equation (i,j,k) is V_{j p + k, c} * W_{k n + i, c}.
            # Note: idx = i*m + j.
            # The equation is linear in U.
            # A * vec(U) = b?
            # A is sparse. Each equation involves R variables (one for each c).
            # Specifically, for a given (i,j,k), the term is sum_c A_{c, c} U_{idx, c}.
            # Wait, U index depends on i,j.
            # So for a fixed (i,j), the variable U_{idx, c} appears in equations where this (i,j) is the index.
            # Equations are indexed by (i,j,k).
            # For a specific i,j, the variables are U_{3i+j, 0..R-1}.
            # They appear in equations for k=0 and k=1.
            # So we have 2 equations for each (i,j).
            # 2 * 2 * 3 = 12 equations total.
            # 6 sets of (i,j). Each has 2 equations.
            # System for U_{3i+j, c}:
            # sum_c V_{2j, c} W_{i, c} U_{3i+j, c} = 1  (for k=0)
            # sum_c V_{2j+1, c} W_{1*2+i, c} U_{3i+j, c} = 1 (for k=1, W index is 2(1)+i = 2+i)
            
            # Let's solve for U for each (i,j) independently?
            # No, U_{3i+j, c} is the variable. The sum is over c.
            # So for fixed i,j, we have:
            # vec(V_j * W_i^T) dot vec(U_{3i+j}) = 1
            # vec(V_{j+1} * W_{i+1}^T) dot vec(U_{3i+j}) = 1  (Wait, W index for k=1 is 2*1+i. V index for k=1 is 2*j+1)
            
            # This is a 2xR system: A_ij * u_ij = [1, 1]^T
            # A_ij is 2xR.
            # A_ij[0, c] = V[2*j, c] * W[i, c]  (Note: W index for k=0 is 0*2+i = i)
            # A_ij[1, c] = V[2*j+1, c] * W[2+i, c] (Note: W index for k=1 is 1*2+i = 2+i)
            
            # We want integer solution for u_ij.
            # Use least squares to get an approximate integer solution, then round?
            # Or use sympy? Sympy is slow.
            # Let's use numpy lstsq and check if close to integers.
            
            solved_U = np.zeros_like(U)
            valid = True
            for i in range(2):
                for j in range(3):
                    idx_u = 3*i + j
                    # Build A
                    row0 = V[2*j, :] * W[i, :]
                    row1 = V[2*j+1, :] * W[2+i, :]
                    A = np.array([row0, row1])
                    b = np.array([1.0, 1.0])
                    # Solve A x = b
                    try:
                        x, residuals, rank_a, s = np.linalg.lstsq(A, b, rcond=None)
                        # Check if x is close to integers
                        x_int = np.round(x)
                        if np.linalg.norm(A.dot(x_int) - b) < 1e-6:
                            solved_U[idx_u, :] = x_int
                        else:
                            # Try small perturbation or just mark invalid for this restart
                            valid = False
                            break
                    except:
                        valid = False
                        break
                if not valid: break
                
            if not valid:
                continue
            U = solved_U
            
            # Update V similarly
            # Equations for V[2*j+k, c]. Variable V_idx.
            # For fixed j, k varies? No, V index is 2*j+k.
            # For fixed j, we have variables V_{2j, c} and V_{2j+1, c}.
            # Equations involving V_{2j, c} are for k=0.
            # Eq (i, j, 0): sum_c U_{3i+j, c} V_{2j, c} W_{i, c} = 1
            # Eq (i, j, 1): sum_c U_{3i+j, c} V_{2j+1, c} W_{2+i, c} = 1
            # So for fixed j, we have a system for V_{2j, c} and V_{2j+1, c}.
            # Variables: v0 = V[2j, :], v1 = V[2j+1, :]
            # Eqs for i=0:
            # 0: sum_c U_{3*0+j, c} v0_c W_{0, c} = 1
            # 1: sum_c U_{3*0+j, c} v1_c W_{2, c} = 1
            # Eqs for i=1:
            # 2: sum_c U_{3*1+j, c} v0_c W_{1, c} = 1
            # 3: sum_c U_{3*1+j, c} v1_c W_{3, c} = 1
            # This is a 4x2R system? No, v0 and v1 are separate vectors of length R.
            # They are coupled in the sum over c, but appear in different terms?
            # No, each equation involves either v0 or v1, not both.
            # Eq 0 involves v0. Eq 1 involves v1. Eq 2 involves v0. Eq 3 involves v1.
            # So we can solve for v0 and v1 independently!
            
            solved_V = np.zeros_like(V)
            valid = True
            for j in range(3):
                # Solve for v0 = V[2j, :]
                # Eqs from i=0 and i=1 with k=0.
                # A0 = U[3*0+j, :] * W[0, :]
                # A1 = U[3*1+j, :] * W[1, :]
                # A * v0 = [1, 1]^T
                A = np.array([U[j, :] * W[0, :], U[3+j, :] * W[1, :]])
                b = np.array([1.0, 1.0])
                try:
                    x, residuals, rank_a, s = np.linalg.lstsq(A, b, rcond=None)
                    x_int = np.round(x)
                    if np.linalg.norm(A.dot(x_int) - b) < 1e-6:
                        solved_V[2*j, :] = x_int
                    else:
                        valid = False
                        break
                except:
                    valid = False
                    break
                
                # Solve for v1 = V[2j+1, :]
                # Eqs from i=0 and i=1 with k=1.
                # A0 = U[3*0+j, :] * W[2, :]
                # A1 = U[3*1+j, :] * W[3, :]
                A = np.array([U[j, :] * W[2, :], U[3+j, :] * W[3, :]])
                b = np.array([1.0, 1.0])
                try:
                    x, residuals, rank_a, s = np.linalg.lstsq(A, b, rcond=None)
                    x_int = np.round(x)
                    if np.linalg.norm(A.dot(x_int) - b) < 1e-6:
                        solved_V[2*j+1, :] = x_int
                    else:
                        valid = False
                        break
                except:
                    valid = False
                    break
            if not valid: continue
            V = solved_V
            
            # Update W
            # Variables W[2k+i, c].
            # For fixed k, we have W_{2k} and W_{2k+1} (if i=0,1).
            # Equations for k=0 (W_0, W_1):
            # i=0, j=0: U[0, c] V[0, c] W[0, c] ... sum
            # This couples W_0 and W_1?
            # Eq (i,j,0) involves W_i.
            # So for fixed k, the variable is W_{k}.
            # But W index is 2k+i.
            # For k=0, we have W_0 (i=0) and W_1 (i=1).
            # Eq (0, j, 0) involves W_0.
            # Eq (1, j, 0) involves W_1.
            # So W_0 and W_1 are decoupled in the sense that each appears in different equations?
            # Eq (0,j,0): sum_c U_{3*0+j, c} V_{2j, c} W_{0, c} = 1
            # Eq (1,j,0): sum_c U_{3*1+j, c} V_{2j, c} W_{1, c} = 1
            # Yes, decoupled.
            
            solved_W = np.zeros_like(W)
            valid = True
            for k in range(2):
                for i in range(2):
                    idx_w = 2*k + i
                    # Equations for this W_idx come from all j, with this i, k.
                    # i is fixed. k is fixed. j varies.
                    # Eq (i, j, k): sum_c U_{3i+j, c} V_{2j+k, c} W_{2k+i, c} = 1
                    # A_j = U[3*i+j, :] * V[2*j+k, :]
                    # We have 3 equations (j=0,1,2) for the vector W_idx.
                    A = np.array([U[3*i + j, :] * V[2*j + k, :] for j in range(3)])
                    b = np.array([1.0, 1.0, 1.0])
                    try:
                        x, residuals, rank_a, s = np.linalg.lstsq(A, b, rcond=None)
                        x_int = np.round(x)
                        if np.linalg.norm(A.dot(x_int) - b) < 1e-6:
                            solved_W[idx_w, :] = x_int
                        else:
                            valid = False
                            break
                    except:
                        valid = False
                        break
                if not valid: break
            if not valid: continue
            W = solved_W
            
            # Check convergence
            # If U, V, W satisfy all equations, we found it.
            if verify_decomp(U, V, W, 2, 3, 2, rank):
                return U, V, W
    return None

def construct():
    n, m, p = DIMENSIONS
    # Try to find low rank for M_{2,3,2}
    target_ranks = [7, 8, 9, 10]
    
    deadline = time.time() + 100 # 100s limit for search
    
    U_232 = None
    V_232 = None
    W_232 = None
    best_rank = 36
    
    for r in target_ranks:
        if time.time() > deadline:
            break
        res = solve_m232(r)
        if res is not None:
            U_232, V_232, W_232 = res
            best_rank = 3 * r
            break # Found minimal for M_232, use it
            
    if U_232 is None:
        # Fallback to schoolbook
        rank = 36
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
        
    # Construct M_{2,3,6} from three M_{2,3,2}
    r_232 = best_rank // 3
    rank_total = best_rank
    
    # U for M_{2,3,6} is same as U_232?
    # U size 12 x rank.
    # U_{i m + j} depends on i, j.
    # In M_{2,3,6}, i in 0,1; j in 0,1,2.
    # U_{3i+j, c} is the coefficient.
    # We have 3 copies of the decomposition.
    # For c in 0..r-1 (copy 1), use U_232.
    # For c in r..2r-1 (copy 2), use U_232.
    # For c in 2r..3r-1 (copy 3), use U_232.
    
    U = np.zeros((12, rank_total), dtype=int)
    V = np.zeros((18, rank_total), dtype=int)
    W = np.zeros((12, rank_total), dtype=int)
    
    for copy in range(3):
        start_c = copy * r_232
        end_c = start_c + r_232
        k_offset = copy * 2
        
        # U: same for all copies
        U[:, start_c:end_c] = U_232[:, :]
        
        # V: V_{j p + k, c}. p=6.
        # For this copy, k in 2*copy, 2*copy+1.
        # V_{j*6 + k, c} = V_232_{j*2 + (k%2), c-start_c}
        for j in range(3):
            for k_local in range(2):
                k_global = k_offset + k_local
                V[j*6 + k_global, start_c:end_c] = V_232[j*2 + k_local, :]
                
        # W: W_{k n + i, c}. n=2.
        # W_{k*2 + i, c} = W_232_{k_local*2 + i, c-start_c}
        for k_local in range(2):
            for i in range(2):
                k_global = k_offset + k_local
                W[k_global*2 + i, start_c:end_c] = W_232[k_local*2 + i, :]
                
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

