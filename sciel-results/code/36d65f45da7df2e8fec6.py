    def try_bini_decomposition(rank, max_trials=1000):
        """
        Attempts to find a rank-`rank` decomposition using Bini's algorithm idea.
        Returns (U, V, W) if found, else None.
        """
        dim_u, dim_v, dim_w = n*m, m*p, p*n
        # Target tensor T is identity-like: T[i*m+j, j*p+k, k*n+i] = 1
        # We want sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        for trial in range(max_trials):
            if time.time() > deadline - 5:
                return None
            
            # Generate random rank-1 components
            # Using small integers to keep values small
            U_rand = np.random.randint(-2, 3, size=(dim_u, rank))
            V_rand = np.random.randint(-2, 3, size=(dim_v, rank))
            W_rand = np.random.randint(-2, 3, size=(dim_w, rank))
            
            # Construct the linear system A * x = b
            # A[i*m+j, j*p+k, r] -> A_flat[idx, r]
            # b[i*m+j, j*p+k] -> b_flat[idx]
            
            # We need to be careful with memory and speed.
            # N = 60, Rank = 46.
            # A is 60 x 46.
            
            A = np.zeros((n*m*p, rank))
            b = np.zeros(n*m*p)
            
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx_u = i*m + j
                        idx_v = j*p + k
                        idx_w = k*n + i
                        # A[idx, r] = U_rand[idx_u, r] * V_rand[idx_v, r] * W_rand[idx_w, r]
                        A[idx, :] = U_rand[idx_u, :] * V_rand[idx_v, :] * W_rand[idx_w, :]
                        b[idx] = 1.0
                        idx += 1
                        
            # Solve A * x = b
            # If a solution exists, the rank is sufficient.
            # We want to check if the residual is 0.
            try:
                sol, residuals, rank_sol, sv = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
                
            # Check residual
            if residuals and residuals[0] < 1e-9:
                # We have a solution.
                # The decomposition is:
                # U_final[:, r] = U_rand[:, r] * sol[r]
                # V_final[:, r] = V_rand[:, r]
                # W_final[:, r] = W_rand[:, r]
                # Wait, this is not symmetric.
                # Bini's algorithm usually constructs U, V, W directly.
                # The "solution" x gives the coefficients for the linear combination.
                # But we need the factors to be integer/half-integer.
                # If sol[r] is not integer, we can absorb it into U_rand.
                # But U_rand * sol[r] might not be integer.
                
                # Better approach:
                # The condition is that T is in the span of {T_r}.
                # This means there exist scalars c_r such that T = sum c_r T_r.
                # This is not a decomposition in the form sum U_r V_r W_r = T with U_r, V_r, W_r being the factors.
                # The factors ARE the vectors u_r, v_r, w_r.
                # If T = sum c_r (u_r \otimes v_r \otimes w_r), we can define
                # U'_r = c_r u_r, V'_r = v_r, W'_r = w_r.
                # Then T = sum U'_r \otimes V'_r \otimes W'_r.
                # So we just need to check if the system has a solution.
                
                # However, we need U', V', W' to be integer/half-integer.
                # Since U_rand, V_rand, W_rand are integers, if c_r is integer/half-integer, we are good.
                # c_r might be any real number.
                # We can try to round c_r and re-verify.
                
                c = sol
                U_final = U_rand * c
                V_final = V_rand
                W_final = W_rand
                
                # Try rounding U_final to half-integers
                U_rounded = np.round(U_final * 2) / 2
                # Verify
                if verify(U_rounded, V_final, W_final, tol=1e-9) <= 1e-9:
                    return U_rounded, V_final, W_final
                    
                # Try rounding V_final
                V_rounded = np.round(V_final * 2) / 2
                if verify(U_final, V_rounded, W_final, tol=1e-9) <= 1e-9:
                    return U_final, V_rounded, W_final
                    
                # Try rounding W_final
                W_rounded = np.round(W_final * 2) / 2
                if verify(U_final, V_final, W_rounded, tol=1e-9) <= 1e-9:
                    return U_final, V_final, W_rounded
                    
                # Try rounding all
                U_r = np.round(U_final * 2) / 2
                V_r = np.round(V_final * 2) / 2
                W_r = np.round(W_final * 2) / 2
                if verify(U_r, V_r, W_r, tol=1e-9) <= 1e-9:
                    return U_r, V_r, W_r
                
        return None
