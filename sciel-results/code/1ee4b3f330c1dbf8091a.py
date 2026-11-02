import numpy as np
import time
import random

DIMENSIONS = [2, 3, 4]

def verify_decomp(U, V, W, n, m, p, tol=1e-9):
    """
    Verifies if U, V, W form an exact decomposition of the matrix multiplication tensor.
    U: (n*m, R)
    V: (m*p, R)
    W: (p*n, R)
    Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
    """
    R = U.shape[1]
    # Pre-flatten for speed if needed, but loops are fine for small sizes
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(R):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if not np.isclose(s, 1.0, atol=tol):
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Schoolbook decomposition (Rank 24)
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook), dtype=float)
    V_school = np.zeros((m * p, rank_schoolbook), dtype=float)
    W_school = np.zeros((p * n, rank_schoolbook), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_school, V_school, W_school

    # Try to find a lower rank decomposition
    # Theoretical lower bounds suggest rank could be lower. 
    # Let's try ranks from 20 up to 23.
    # We use a random search with local optimization (gradient-free perturbation)
    # targeting integer/half-integer solutions.
    
    for R in range(20, rank_schoolbook):
        if time.time() > deadline:
            break
            
        found = False
        # Number of restarts depends on remaining time
        max_restarts = 200
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Initialize with small random integers/half-integers
            # Using {-1, 0, 1} or {-0.5, 0, 0.5, 1}
            # Let's try integers first, then half-integers if needed
            vals = [-1, 0, 1]
            U = np.random.choice(vals, size=(n * m, R))
            V = np.random.choice(vals, size=(m * p, R))
            W = np.random.choice(vals, size=(p * n, R))
            
            # Local search to minimize error
            # Error function: sum (sum_c U_ijk V_jk W_ki - 1)^2
            best_err = float('inf')
            best_U_loc, best_V_loc, best_W_loc = U.copy(), V.copy(), W.copy()
            
            # Simple iterative improvement: pick a random column, random row, change value
            # This is very slow. Better: use a few steps of gradient descent on continuous values, then snap.
            
            U_c = U.astype(float)
            V_c = V.astype(float)
            W_c = W.astype(float)
            
            # Gradient Descent Steps
            lr = 0.01
            for step in range(50):
                # Compute gradients numerically or analytically?
                # Analytical is better.
                # E = sum_{i,j,k} (S_{ijk} - 1)^2
                # S_{ijk} = sum_c U_{ij,c} V_{jk,c} W_{ki,c}
                
                # This is expensive to compute gradients manually in Python loops for 50 steps * 200 restarts.
                # Let's try a different approach: Random search with "snap to grid"
                # If we find a real solution with small error, snap it.
                
                # Actually, for such small problems, a pure random search for integers is often insufficient.
                # Let's try a hybrid: 
                # 1. Random init
                # 2. Run a few steps of coordinate descent on the continuous domain
                # 3. Snap to nearest 0.5
                # 4. Verify
                # 5. If fail, perturb and retry
                
                # Given the time limit, let's stick to a more structured search if possible, 
                # or a very fast local search.
                
                # Let's try to solve the linear system for one matrix given the other two?
                # If we fix V and W, the equations for U are linear:
                # sum_c U_{ij,c} (V_{jk,c} W_{ki,c}) = 1 for all i,j,k.
                # This is a system of 24 equations for U (which has 6*R variables).
                # We can use least squares to find U that minimizes error, given V, W.
                
                # Strategy:
                # 1. Randomly initialize V and W (integers/half-integers).
                # 2. Solve for U using least squares (continuous).
                # 3. Snap U to nearest 0.5.
                # 4. Verify.
                # 5. If fail, try swapping roles (solve for V given U, W, etc.) or random restart.
                
                # Let's implement this "fix two, solve one" strategy.
                
                # Initialize V and W with small integers
                V_try = np.random.choice([-1, 0, 1], size=(m * p, R))
                W_try = np.random.choice([-1, 0, 1], size=(p * n, R))
                
                # Construct A matrix and b vector for U
                # Variables: U_{ij,c} for i in [0,n), j in [0,m), c in [0,R)
                # Total variables: n*m*R
                # Equations: for each i,j,k: sum_c U_{ij,c} * V_{jk,c} * W_{ki,c} = 1
                
                n_vars = n * m * R
                n_eqs = n * m * p
                
                A_mat = np.zeros((n_eqs, n_vars))
                b_vec = np.ones(n_eqs)
                
                eq_idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # Coefficients for U_{i, j, c}
                            # Variable index: (i * m + j) * R + c
                            for c in range(R):
                                coeff = V_try[j * p + k, c] * W_try[k * n + i, c]
                                if coeff != 0:
                                    A_mat[eq_idx, (i * m + j) * R + c] = coeff
                            eq_idx += 1
                
                # Solve least squares
                try:
                    U_sol, residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                except:
                    continue
                
                # Reshape U_sol
                U_sol = U_sol.reshape((n * m, R))
                
                # Snap to nearest 0.5
                U_snap = np.round(U_sol * 2) / 2
                
                # Verify
                if verify_decomp(U_snap, V_try, W_try, n, m, p):
                    best_rank = R
                    best_U, best_V, best_W = U_snap, V_try, W_try
                    found = True
                    break
            
            if found:
                break
                
            # Also try solving for V given U, W and W given U, V to cover more space
            # But time is limited. The "fix V, W" approach is a good start.
            # If we don't find it in the first few restarts, it might be hard.
            
            # Optimization: If R is close to schoolbook, it's easier.
            # Let's add a second pass where we fix U and W, solve for V.
            
            # Re-initialize U and W
            U_try = np.random.choice([-1, 0, 1], size=(n * m, R))
            W_try = np.random.choice([-1, 0, 1], size=(p * n, R))
            
            # Solve for V
            # Eq: sum_c U_{ij,c} V_{jk,c} W_{ki,c} = 1
            # Variable: V_{jk,c}
            # Coeff: U_{ij,c} * W_{ki,c}
            
            A_mat = np.zeros((n_eqs, n_vars)) # n_vars is m*p*R
            b_vec = np.ones(n_eqs)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for c in range(R):
                            coeff = U_try[i * m + j, c] * W_try[k * n + i, c]
                            if coeff != 0:
                                A_mat[eq_idx, (j * p + k) * R + c] = coeff
                        eq_idx += 1
                        
            try:
                V_sol, residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            except:
                continue
                
            V_sol = V_sol.reshape((m * p, R))
            V_snap = np.round(V_sol * 2) / 2
            
            if verify_decomp(U_try, V_snap, W_try, n, m, p):
                best_rank = R
                best_U, best_V, best_W = U_try, V_snap, W_try
                found = True
                break
                
            if found:
                break

    # Convert to lists
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
