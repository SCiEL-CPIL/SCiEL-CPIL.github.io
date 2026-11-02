import numpy as np
import time
import random

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Initialize Schoolbook
    U = np.zeros((n * m, rank_schoolbook), dtype=float)
    V = np.zeros((m * p, rank_schoolbook), dtype=float)
    W = np.zeros((p * n, rank_schoolbook), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    current_rank = rank_schoolbook
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    best_rank = current_rank
    
    start_time = time.time()
    time_limit = 100
    
    def is_valid(U, V, W, r):
        # Check if sum U*V*W == 1 for all i,j,k
        # Use vectorized ops for speed?
        # T_ijk = sum_c U_ij_c V_jk_c W_ki_c
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(r):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > 1e-6:
                        return False
        return True

    def snap_to_half_int(matrix):
        return np.round(matrix * 2) / 2

    # Helper to try eliminating term c by solving for variable 'var' in U, V, W
    # var: 0 for U, 1 for V, 2 for W
    def try_eliminate(U, V, W, r, c, var):
        # Create copies
        U_new = U.copy()
        V_new = V.copy()
        W_new = W.copy()
        
        # Zero out term c
        if var == 0: # Solve for U
            # We fix V and W.
            # For each j, k:
            #   Variables: U_{i j}^{c'} for i=0..n-1, c' != c
            #   Equations: sum_{c'} U_{i j}^{c'} V_{j k}^{c'} W_{k i}^{c'} = 1 for i=0..n-1
            #   Matrix A: n x (r-1)
            #   b: n x 1 (all ones)
            #   We solve A x = b
            #   x is vector of U_{i j}^{c'}? No.
            #   Let's flatten the variables.
            #   For a fixed j, k, the variables are U_{0 j}^{c'}, ..., U_{n-1 j}^{c'} for c' != c.
            #   Total variables: n * (r-1).
            #   Equations: for each i, sum_{c'} U_{i j}^{c'} V_{j k}^{c'} W_{k i}^{c'} = 1.
            #   This is n equations.
            #   Wait, the equation is for a specific i.
            #   So for each i, we have: sum_{c' != c} U_{i j}^{c'} (V_{j k}^{c'} W_{k i}^{c'}) = 1.
            #   This is a system of n equations with n*(r-1) unknowns?
            #   No, the unknowns for equation i are U_{i j}^{c'} for c' != c.
            #   So for each i, we have r-1 unknowns.
            #   And we have n equations.
            #   But the unknowns in eq i are disjoint from eq i'?
            #   U_{i j}^{c'} only appears in eq i.
            #   So we have n independent systems of r-1 unknowns and 1 equation?
            #   No, that's not right.
            #   The tensor condition is:
            #   sum_{c} U_{i j}^{c} V_{j k}^{c} W_{k i}^{c} = 1
            #   If we fix V, W, and want to find U:
            #   For each (i, j, k), we have a constraint.
            #   But U_{i j}^{c} appears in constraints for all k.
            #   So for fixed i, j, the variables are U_{i j}^{c'} (c' != c).
            #   The constraints are:
            #   For each k: sum_{c' != c} U_{i j}^{c'} V_{j k}^{c'} W_{k i}^{c'} = 1.
            #   This is p equations (k=0..p-1) with r-1 unknowns.
            #   We can solve this system.
            
            success = True
            for i in range(n):
                for j in range(m):
                    # Build system for U_{i j}^{c'}
                    # Variables: x_{c'} = U_{i j}^{c'} for c' != c
                    # Equations: for k in 0..p-1:
                    #   sum_{c' != c} x_{c'} V_{j k}^{c'} W_{k i}^{c'} = 1
                    A = np.zeros((p, r-1))
                    b = np.ones(p)
                    idx = 0
                    for c_prime in range(r):
                        if c_prime == c:
                            continue
                        for k in range(p):
                            A[k, idx] = V[j*p+k, c_prime] * W[k*n+i, c_prime]
                        idx += 1
                    # Solve A x = b
                    # If p > r-1, least squares. If p <= r-1, might have many solutions.
                    # We want a solution that is close to half-integers.
                    # Use lstsq for minimum norm.
                    x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    
                    # Check if residual is small
                    if np.linalg.norm(A.dot(x) - b) > 1e-6:
                        success = False
                        break
                    
                    # Check if x is close to half-integers
                    x_snapped = snap_to_half_int(x)
                    if np.linalg.norm(A.dot(x_snapped) - b) > 1e-6:
                        # Try to find a better solution?
                        # For now, reject
                        success = False
                        break
                        
                    # Assign back
                    idx = 0
                    for c_prime in range(r):
                        if c_prime == c:
                            continue
                        U_new[i*m+j, c_prime] = x_snapped[idx]
                        idx += 1
                        
            if not success:
                return None, None, None, False
                
            # Zero out term c in U_new
            U_new[:, c] = 0
            
            return U_new, V_new, W_new, True

        elif var == 1: # Solve for V
            success = True
            for j in range(m):
                for k in range(p):
                    # Variables: V_{j k}^{c'} for c' != c
                    # Equations: for i in 0..n-1:
                    #   sum_{c' != c} U_{i j}^{c'} V_{j k}^{c'} W_{k i}^{c'} = 1
                    A = np.zeros((n, r-1))
                    b = np.ones(n)
                    idx = 0
                    for c_prime in range(r):
                        if c_prime == c:
                            continue
                        for i in range(n):
                            A[i, idx] = U[i*m+j, c_prime] * W[k*n+i, c_prime]
                        idx += 1
                    x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    if np.linalg.norm(A.dot(x) - b) > 1e-6:
                        success = False
                        break
                    x_snapped = snap_to_half_int(x)
                    if np.linalg.norm(A.dot(x_snapped) - b) > 1e-6:
                        success = False
                        break
                    idx = 0
                    for c_prime in range(r):
                        if c_prime == c:
                            continue
                        V_new[j*p+k, c_prime] = x_snapped[idx]
                        idx += 1
            if not success:
                return None, None, None, False
            V_new[:, c] = 0
            return U_new, V_new, W_new, True

        else: # Solve for W
            success = True
            for k in range(p):
                for i in range(n):
                    # Variables: W_{k i}^{c'} for c' != c
                    # Equations: for j in 0..m-1:
                    #   sum_{c' != c} U_{i j}^{c'} V_{j k}^{c'} W_{k i}^{c'} = 1
                    A = np.zeros((m, r-1))
                    b = np.ones(m)
                    idx = 0
                    for c_prime in range(r):
                        if c_prime == c:
                            continue
                        for j in range(m):
                            A[j, idx] = U[i*m+j, c_prime] * V[j*p+k, c_prime]
                        idx += 1
                    x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    if np.linalg.norm(A.dot(x) - b) > 1e-6:
                        success = False
                        break
                    x_snapped = snap_to_half_int(x)
                    if np.linalg.norm(A.dot(x_snapped) - b) > 1e-6:
                        success = False
                        break
                    idx = 0
                    for c_prime in range(r):
                        if c_prime == c:
                            continue
                        W_new[k*n+i, c_prime] = x_snapped[idx]
                        idx += 1
            if not success:
                return None, None, None, False
            W_new[:, c] = 0
            return U_new, V_new, W_new, True

    # Main Loop
    while time.time() - start_time < time_limit:
        progress = False
        # Try to eliminate a random term
        for c in range(current_rank - 1, -1, -1): # Try from end to start? Or random?
            if time.time() - start_time > time_limit:
                break
            # Try solving for U, V, W
            for var in [0, 1, 2]:
                U_new, V_new, W_new, success = try_eliminate(U, V, W, current_rank, c, var)
                if success:
                    # Verify full decomposition
                    if is_valid(U_new, V_new, W_new, current_rank - 1):
                        U, V, W = U_new, V_new, W_new
                        current_rank -= 1
                        best_U = U.copy()
                        best_V = V.copy()
                        best_W = W.copy()
                        best_rank = current_rank
                        progress = True
                        break
            if progress:
                break
                
        if not progress:
            # If no progress, maybe we are stuck.
            # We can try to perturb and repair?
            # Or just break?
            # Let's try a few random restarts of the elimination process with different term orders.
            # Or simply break to save time, as schoolbook is the baseline.
            # Given the difficulty, it might be hard to reduce below 105.
            # But let's try a bit more.
            # Shuffle terms?
            # Actually, if we can't eliminate, we are at a local minimum.
            # We can try to "swap" terms to create redundancy.
            # This is complex.
            # Let's just break.
            break

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
