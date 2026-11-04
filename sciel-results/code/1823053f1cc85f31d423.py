<<<<<<< SEARCH
    start_time = time.time()
    time_limit = 110.0

    best_result = None
    best_rank = n * m * p
    
    # Schoolbook
    U_sb = np.zeros((dim_U, best_rank))
    V_sb = np.zeros((dim_V, best_rank))
    W_sb = np.zeros((dim_W, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
                
    if verify(U_sb, V_sb, W_sb):
        best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    # Search for lower ranks using ALS with snapping
    # Try ranks from 40 upwards
    for R in range(40, best_rank):
        if time.time() - start_time > time_limit:
            break
            
        found = False
        # Multiple restarts
        for restart in range(50):
            if time.time() - start_time > time_limit:
                break
                
            # Initialize U, V, W randomly with half-integers
            U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(dim_U, R))
            V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(dim_V, R))
            W = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(dim_W, R))
            
            # ALS Loop
            for it in range(20):
                if time.time() - start_time > time_limit:
                    break
                    
                # Solve for U
                U_new = solve_for_U(V, W)
                U_new = snap_to_half_int(U_new)
                
                # Solve for V
                V_new = solve_for_V(U_new, W)
                V_new = snap_to_half_int(V_new)
                
                # Solve for W
                W_new = solve_for_W(U_new, V_new)
                W_new = snap_to_half_int(W_new)
                
                # Check convergence
                res = calculate_residual(U_new, V_new, W_new)
                
                if res < 1e-9:
                    if verify(U_new, V_new, W_new):
                        best_result = {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
                        best_rank = R
                        found = True
                        break
                
                U, V, W = U_new, V_new, W_new
                
            if found:
                break

        if found:
            break

    if best_result is None:
        best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    return best_result
=======
    def polish(U, V, W, max_iter=50, tol=1e-12):
        """
        Final high-precision polish pass.
        """
        for it in range(max_iter):
            U_new = solve_for_U(V, W)
            V_new = solve_for_V(U_new, W)
            W_new = solve_for_W(U_new, V_new)
            
            # Check convergence
            res = calculate_residual(U_new, V_new, W_new)
            if res < tol:
                U, V, W = U_new, V_new, W_new
                break
                
            U, V, W = U_new, V_new, W_new
            
        return U, V, W

    start_time = time.time()
    time_limit = 140.0  # Leave 10s margin for 150s limit

    best_result = None
    best_rank = n * m * p  # 75
    
    # Schoolbook decomposition
    U_sb = np.zeros((dim_U, best_rank))
    V_sb = np.zeros((dim_V, best_rank))
    W_sb = np.zeros((dim_W, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
                
    if verify(U_sb, V_sb, W_sb):
        best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    # Search for lower ranks using ALS with snapping
    # Try ranks from lower bound upwards
    # Theoretical lower bound: max(n*m, m*p, p*n) = 25, but likely higher
    # Let's try from 30 to 74
    ranks_to_try = list(range(30, best_rank))
    
    for R in ranks_to_try:
        if time.time() - start_time > time_limit:
            break
            
        found = False
        # Multiple restarts - fewer for higher ranks
        n_restarts = max(10, 50 - (R - 30))
        
        for restart in range(n_restarts):
            if time.time() - start_time > time_limit:
                break
                
            # Initialize U, V, W randomly with half-integers
            # Use a mix of small values to help convergence
            U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(dim_U, R))
            V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(dim_V, R))
            W = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(dim_W, R))
            
            # ALS Loop
            for it in range(30):
                if time.time() - start_time > time_limit:
                    break
                    
                # Solve for U
                U_new = solve_for_U(V, W)
                U_new = snap_to_half_int(U_new)
                
                # Solve for V
                V_new = solve_for_V(U_new, W)
                V_new = snap_to_half_int(V_new)
                
                # Solve for W
                W_new = solve_for_W(U_new, V_new)
                W_new = snap_to_half_int(W_new)
                
                # Check convergence
                res = calculate_residual(U_new, V_new, W_new)
                
                if res < 1e-9:
                    if verify(U_new, V_new, W_new):
                        # Try to polish for better precision
                        U_polished, V_polished, W_polished = polish(U_new, V_new, W_new)
                        if verify(U_polished, V_polished, W_polished):
                            U_new, V_new, W_new = U_polished, V_polished, W_polished
                        
                        best_result = {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
                        best_rank = R
                        found = True
                        break
                
                U, V, W = U_new, V_new, W_new
                
            if found:
                break

        if found:
            break

    if best_result is None:
        best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    return best_result
>>>>>>> REPLACE
