# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    
    # Schoolbook decomposition
    U_sb = np.zeros((n * m, rank_school))
    V_sb = np.zeros((m * p, rank_school))
    W_sb = np.zeros((p * n, rank_school))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * m * p + j * p + k
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1

    # Target Tensor
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1

    def is_valid(U, V, W, tol=1e-9):
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    for r in range(R):
                        val += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                    if abs(val - 1) > tol:
                        return False
        return True

    start_time = time.time()
    deadline = 110.0  # Safety margin for 120s limit

    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_school

    # Helper to solve for one matrix given the other two
    def solve_for_matrix(fixed1, fixed2, solve_for, R):
        """
        fixed1, fixed2: The two matrices we keep fixed (U, V, or W)
        solve_for: 'U', 'V', or 'W'
        R: Rank
        Returns the solved matrix or None if no integer solution found
        """
        # Construct linear system A @ X_flat = b
        # Equations: sum_r A1[i,j,r] * A2[k,l,r] * A3[x,y,r] = 1
        # We need to map the indices correctly.
        
        # Dimensions:
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        
        # We will build A_mat and b_vec for the unknown matrix.
        # Number of equations: n*m*p
        # Number of variables: (size of unknown matrix) * R
        
        num_vars = 0
        A_mat = None
        
        if solve_for == 'U':
            # Unknown: U (n*m, R)
            # Fixed: V, W
            num_vars = (n * m) * R
            A_mat = np.zeros((n * m * p, num_vars))
            b_vec = np.ones(n * m * p)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Term: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        u_row = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        for r in range(R):
                            var_idx = u_row * R + r
                            A_mat[eq_idx, var_idx] = fixed2[v_idx, r] * fixed1[w_idx, r] # Note: fixed1=V, fixed2=W in this call? 
                            # Let's standardize arguments: solve_for_matrix(V, W, 'U', R)
                            # So fixed1 is V, fixed2 is W.
                            # Equation: sum_r U[u,r] * V[v,r] * W[w,r] = 1
                            # Coeff for U[u,r] is V[v,r] * W[w,r]
                        eq_idx += 1
                            
        elif solve_for == 'V':
            # Unknown: V (m*p, R)
            # Fixed: U, W
            num_vars = (m * p) * R
            A_mat = np.zeros((n * m * p, num_vars))
            b_vec = np.ones(n * m * p)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_row = j * p + k
                        w_idx = k * n + i
                        for r in range(R):
                            var_idx = v_row * R + r
                            A_mat[eq_idx, var_idx] = fixed1[u_idx, r] * fixed2[w_idx, r]
                        eq_idx += 1

        elif solve_for == 'W':
            # Unknown: W (p*n, R)
            # Fixed: U, V
            num_vars = (p * n) * R
            A_mat = np.zeros((n * m * p, num_vars))
            b_vec = np.ones(n * m * p)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_row = k * n + i
                        for r in range(R):
                            var_idx = w_row * R + r
                            A_mat[eq_idx, var_idx] = fixed1[u_idx, r] * fixed2[v_idx, r]
                        eq_idx += 1
        else:
            return None

        # Solve A_mat @ X_flat = b_vec
        try:
            if num_vars >= n * m * p:
                X_flat, residuals, rank, s = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            else:
                X_flat = np.linalg.solve(A_mat, b_vec)
            
            # Check integer solution
            X_int = np.round(X_flat)
            if np.allclose(A_mat @ X_int, b_vec, atol=1e-9):
                return X_int
            # Check half-integer solution
            X_half = np.round(X_flat * 2) / 2
            if np.allclose(A_mat @ X_half, b_vec, atol=1e-9):
                return X_half
                
        except np.linalg.LinAlgError:
            pass
            
        return None

    # Search for lower ranks
    for R in range(rank_school - 1, 15, -1):
        if time.time() > deadline:
            break
            
        found = False
        # Increase attempts for lower ranks as they are harder? Or just fixed high number.
        # 1000 attempts is reasonable if fast.
        max_attempts = 1000
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
                
            # Randomly decide which matrix to solve for
            choice = np.random.randint(0, 3)
            
            # Initialize random matrices with small integers
            # Using -1, 0, 1 often yields better conditioned systems for small ranks
            U_rand = np.random.randint(-1, 2, size=(n * m, R))
            V_rand = np.random.randint(-1, 2, size=(m * p, R))
            W_rand = np.random.randint(-1, 2, size=(p * n, R))
            
            if choice == 0:
                # Solve for U
                U_sol = solve_for_matrix(V_rand, W_rand, 'U', R)
                if U_sol is not None:
                    U_sol = U_sol.reshape((n * m, R))
                    if is_valid(U_sol, V_rand, W_rand):
                        best_U, best_V, best_W = U_sol, V_rand, W_rand
                        best_rank = R
                        found = True
                        break
            elif choice == 1:
                # Solve for V
                V_sol = solve_for_matrix(U_rand, W_rand, 'V', R)
                if V_sol is not None:
                    V_sol = V_sol.reshape((m * p, R))
                    if is_valid(U_rand, V_sol, W_rand):
                        best_U, best_V, best_W = U_rand, V_sol, W_rand
                        best_rank = R
                        found = True
                        break
            else:
                # Solve for W
                W_sol = solve_for_matrix(U_rand, V_rand, 'W', R)
                if W_sol is not None:
                    W_sol = W_sol.reshape((p * n, R))
                    if is_valid(U_rand, V_rand, W_sol):
                        best_U, best_V, best_W = U_rand, V_rand, W_sol
                        best_rank = R
                        found = True
                        break
                        
        if found:
            # Found a valid decomposition for this rank, try lower
            continue
        else:
            # If we didn't find it, we might still have a valid one from previous iteration?
            # No, we only update best_rank when found.
            # If we exhaust attempts for this R, we move to R-1.
            pass

    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
