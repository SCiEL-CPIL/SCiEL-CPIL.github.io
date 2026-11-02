<<<<<<< SEARCH
    # Try to find lower rank decompositions
    # Known bounds: rank of 2x3*3x4 matrix multiplication
    # Lower bound: max(n*m, m*p, p*n) = max(6, 12, 8) = 12
    # Upper bound: n*m*p = 24
    # Let's try ranks from 12 upward
    
    # Use randomized search / optimization
    # The tensor equation is: for all (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # This is a system of equations in the entries of U, V, W.
    # We can use a random initialization and gradient descent, or just try random matrices.
    
    # Actually, let's think about this differently.
    # The matrix multiplication tensor is a 3-tensor. We want a rank-r decomposition.
    # 
    # Approach: For a given rank r, we have r*r*(nm + mp + pn) unknowns and n*m*p equations.
    # For r=12: 12*12*(6+12+8) = 144*26 = 3744 unknowns, 24 equations. Very underdetermined.
    # 
    # Let's try a structured approach. Fix W to be simple, then solve for U and V.
    
    # Strategy: Try small ranks with random initialization and verify.
    # Since the system is highly underdetermined for small r, random solutions should exist.
    
    # Let's try ranks from max(nm, mp, pn) = 12 up to schoolbook rank - 1
    
    min_rank = max(nm, mp, pn)  # 12
    
    for rank in range(min_rank, best_rank):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random initializations
        max_attempts = 50 if rank <= 14 else 20
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            U = np.random.randn(nm, rank)
            V = np.random.randn(mp, rank)
            W = np.random.randn(pn, rank)
            
            # Verify
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(rank):
                            s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        if abs(s - 1.0) > 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Try to snap to half-integers
                U_s, V_s, W_s = _snap_to_half_int(U, V, W)
                # Re-verify after snapping
                U_a = np.array(U_s)
                V_a = np.array(V_s)
                W_a = np.array(W_s)
                valid_snapped = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(rank):
                                s += U_a[i*m+j, c] * V_a[j*p+k, c] * W_a[k*n+i, c]
                            if abs(s - 1.0) > 1e-6:
                                valid_snapped = False
                                break
                        if not valid_snapped:
                            break
                    if not valid_snapped:
                        break
                
                if valid_snapped:
                    best_rank = rank
                    best_result = {"U": U_s, "V": V_s, "W": W_s}
                else:
                    # Use the non-snapped version if it's valid
                    best_rank = rank
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
                break  # Found a valid decomposition for this rank
    
    return best_result
=======
    # Try to find lower rank decompositions
    # Lower bound: max(n*m, m*p, p*n) = 12
    # Upper bound: 24
    min_rank = max(nm, mp, pn)  # 12
    
    for rank in range(min_rank, best_rank):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        max_attempts = 5
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Random initialization
            U = np.random.randn(nm, rank)
            V = np.random.randn(mp, rank)
            W = np.random.randn(pn, rank)
            
            params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            # Optimize
            try:
                res = least_squares(_residuals, params0, args=((nm, rank), (mp, rank), (pn, rank), n, m, p), method='trf', max_nfev=2000)
            except Exception:
                continue
                
            if res.cost < 1e-10:
                # Extract solution
                U_opt = res.x[:nm*rank].reshape(nm, rank)
                V_opt = res.x[nm*rank:nm*rank + mp*rank].reshape(mp, rank)
                W_opt = res.x[nm*rank + mp*rank:].reshape(pn, rank)
                
                # Verify with tight tolerance
                if _verify(U_opt, V_opt, W_opt, n, m, p, tol=1e-6):
                    # Try to snap
                    U_s, V_s, W_s = _snap_to_half_int(U_opt, V_opt, W_opt)
                    if _verify(U_s, V_s, W_s, n, m, p, tol=1e-6):
                        best_rank = rank
                        best_result = {"U": U_s, "V": V_s, "W": W_s}
                        break
                    else:
                        # If snapping fails, the solution might be irrational. 
                        # We cannot use it if integer/half-integer is required.
                        pass

    return best_result
>>>>>>> REPLACE
