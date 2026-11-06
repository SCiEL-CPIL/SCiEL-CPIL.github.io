<<<<<<< SEARCH
        if rel_err < 1e-10 and rel_err_test < 1e-10:
            # Model is valid. Use exact solution via matrix exponential.
            dt = t1 - t0
            
            # Check if b is effectively zero to use simpler form
            b_norm = np.linalg.norm(b)
            a_norm = np.linalg.norm(A, ord=1)
            
            if b_norm < 1e-12 * (a_norm + 1.0):
                # Purely linear: y' = A y
                # y_end = expm(A * dt) @ y0
                # Use dense expm for maximum accuracy (n=100 is small)
                try:
                    E_A = expm(A * dt)
                    y_end = E_A @ y0_arr
                    if np.all(np.isfinite(y_end)):
                        return y_end
                except Exception:
                    pass
                
                # Fallback to sparse expm_multiply
                try:
                    A_sparse = diags([lower_val, diag_val, upper_val], [-1, 0, 1], shape=(n, n))
                    A_dt_sparse = A_sparse * dt
                    y_end = expm_multiply(A_dt_sparse, y0_arr)
                    if np.all(np.isfinite(y_end)):
                        return y_end
                except Exception:
                    pass
            else:
                # Affine: y' = A y + b
                # Closed form: y(t) = e^(At) y0 + ∫₀ᵗ e^(A(t-s)) b ds
                # Use augmented matrix with dense expm for accuracy
                try:
                    M = np.zeros((n+1, n+1))
                    M[:n, :n] = A
                    M[:n, n] = b
                    E_M = expm(M * dt)
                    y_end = E_M[:n, :n] @ y0_arr + E_M[:n, n]
                    
                    if np.all(np.isfinite(y_end)):
                        return y_end
                except Exception:
                    pass
                
                # Fallback: use the closed form with dense matrices
                try:
                    E_A = expm(A * dt)
                    try:
                        rhs = (E_A - np.eye(n)) @ b
                        X = np.linalg.solve(A, rhs)
                        y_end = E_A @ y0_arr + X
                    except np.linalg.LinAlgError:
                        M = np.zeros((n+1, n+1))
                        M[:n, :n] = A
                        M[:n, n] = b
                        E_M = expm(M * dt)
                        y_end = E_M[:n, :n] @ y0_arr + E_M[:n, n]
                    
                    if np.all(np.isfinite(y_end)):
                        return y_end
                except Exception:
                    pass
=======
        if rel_err < 1e-10 and rel_err_test < 1e-10:
            # Model is valid. Use spectral decomposition for exact solution.
            dt = t1 - t0
            
            # Check if b is effectively zero to use simpler form
            b_norm = np.linalg.norm(b)
            a_norm = np.linalg.norm(A, ord=1)
            
            if b_norm < 1e-12 * (a_norm + 1.0):
                # Purely linear: y' = A y
                # Use spectral decomposition of tridiagonal Toeplitz matrix
                # For A with diag=d, upper=u, lower=l:
                # Eigenvalues: lambda_k = d + 2*sqrt(u*l)*cos(k*pi/(n+1)), k=1..n
                # Eigenvectors: V[j,k] = sin((j+1)*k*pi/(n+1)) / norm
                try:
                    if abs(upper_val * lower_val) > 1e-30:
                        # General case: asymmetric tridiagonal Toeplitz
                        sqrt_ul = np.sqrt(upper_val * lower_val)
                        # Eigenvalues
                        k_vals = np.arange(1, n+1)
                        eigvals = diag_val + 2.0 * sqrt_ul * np.cos(k_vals * np.pi / (n + 1))
                        # Eigenvectors (columns of V)
                        j_vals = np.arange(0, n)
                        # V[j,k] = sin((j+1)*(k+1)*pi/(n+1))
                        V = np.sin(np.outer(j_vals + 1, k_vals) * np.pi / (n + 1))
                        # Normalize columns
                        V_norm = np.linalg.norm(V, axis=0)
                        V = V / V_norm
                        # For symmetric case (u==l), V is orthogonal. For asymmetric, need V^{-1}
                        # Compute V^{-1} using the fact that for this structure, V^{-1} has similar form
                        # Actually, let's just compute it numerically for n=100
                        V_inv = np.linalg.inv(V)
                        
                        # Compute expm(A*dt) @ y0 via diagonalization
                        # expm(A*dt) = V * diag(exp(eigvals*dt)) * V^{-1}
                        exp_eig = np.exp(eigvals * dt)
                        # y_end = V @ (exp_eig * (V_inv @ y0))
                        V_inv_y0 = V_inv @ y0_arr
                        scaled = exp_eig * V_inv_y0
                        y_end = V @ scaled
                        
                        if np.all(np.isfinite(y_end)):
                            return y_end
                    else:
                        # Degenerate case: one of upper/lower is zero
                        # Upper triangular or lower triangular
                        if abs(upper_val) < 1e-30:
                            # Lower triangular Toeplitz: eigenvalues are all diag_val
                            # This is a Jordan block case if lower != 0
                            # For simplicity, fall back to expm
                            E_A = expm(A * dt)
                            y_end = E_A @ y0_arr
                        else:
                            # Upper triangular Toeplitz
                            E_A = expm(A * dt)
                            y_end = E_A @ y0_arr
                        
                        if np.all(np.isfinite(y_end)):
                            return y_end
                except Exception:
                    pass
                
                # Fallback to sparse expm_multiply
                try:
                    A_sparse = diags([lower_val, diag_val, upper_val], [-1, 0, 1], shape=(n, n))
                    A_dt_sparse = A_sparse * dt
                    y_end = expm_multiply(A_dt_sparse, y0_arr)
                    if np.all(np.isfinite(y_end)):
                        return y_end
                except Exception:
                    pass
            else:
                # Affine: y' = A y + b
                # Closed form: y(t) = e^(At) y0 + ∫₀ᵗ e^(A(t-s)) b ds
                # First compute expm(A*dt) @ y0, then handle the integral
                try:
                    # Compute E_A @ y0 using spectral decomposition
                    if abs(upper_val * lower_val) > 1e-30:
                        sqrt_ul = np.sqrt(upper_val * lower_val)
                        k_vals = np.arange(1, n+1)
                        eigvals = diag_val + 2.0 * sqrt_ul * np.cos(k_vals * np.pi / (n + 1))
                        j_vals = np.arange(0, n)
                        V = np.sin(np.outer(j_vals + 1, k_vals) * np.pi / (n + 1))
                        V_norm = np.linalg.norm(V, axis=0)
                        V = V / V_norm
                        V_inv = np.linalg.inv(V)
                        
                        exp_eig = np.exp(eigvals * dt)
                        V_inv_y0 = V_inv @ y0_arr
                        scaled = exp_eig * V_inv_y0
                        E_A_y0 = V @ scaled
                    else:
                        E_A = expm(A * dt)
                        E_A_y0 = E_A @ y0_arr
                    
                    # Now compute the integral term: ∫₀ᵗ e^(A(t-s)) b ds
                    # This equals A^{-1} * (e^(A*t) - I) * b if A is invertible
                    # Or use augmented matrix approach
                    try:
                        # Try A^{-1} * (E_A - I) * b
                        if abs(upper_val * lower_val) > 1e-30:
                            # We have E_A in spectral form
                            # E_A - I = V * diag(exp_eig - 1) * V^{-1}
                            exp_eig_minus_1 = np.exp(eigvals * dt) - 1.0
                            V_inv_b = V_inv @ b
                            scaled_b = exp_eig_minus_1 * V_inv_b
                            E_A_minus_I_b = V @ scaled_b
                            
                            # Now need A^{-1} * E_A_minus_I_b
                            # A^{-1} for tridiagonal Toeplitz can be computed efficiently
                            # But for n=100, just use np.linalg.solve
                            try:
                                A_inv_EA_minus_I_b = np.linalg.solve(A, E_A_minus_I_b)
                            except np.linalg.LinAlgError:
                                # A is singular, use augmented matrix
                                M = np.zeros((n+1, n+1))
                                M[:n, :n] = A
                                M[:n, n] = b
                                E_M = expm(M * dt)
                                y_end = E_M[:n, :n] @ y0_arr + E_M[:n, n]
                                if np.all(np.isfinite(y_end)):
                                    return y_end
                                else:
                                    raise
                            
                            y_end = E_A_y0 + A_inv_EA_minus_I_b
                        else:
                            # Degenerate case, use augmented matrix
                            M = np.zeros((n+1, n+1))
                            M[:n, :n] = A
                            M[:n, n] = b
                            E_M = expm(M * dt)
                            y_end = E_M[:n, :n] @ y0_arr + E_M[:n, n]
                        
                        if np.all(np.isfinite(y_end)):
                            return y_end
                    except Exception:
                        pass
                    
                    # Fallback: augmented matrix
                    M = np.zeros((n+1, n+1))
                    M[:n, :n] = A
                    M[:n, n] = b
                    E_M = expm(M * dt)
                    y_end = E_M[:n, :n] @ y0_arr + E_M[:n, n]
                    
                    if np.all(np.isfinite(y_end)):
                        return y_end
                except Exception:
                    pass
>>>>>>> REPLACE
