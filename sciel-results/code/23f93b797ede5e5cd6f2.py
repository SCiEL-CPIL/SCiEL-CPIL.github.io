    def objective(params):
        a = params[:12]
        b = params[12:]
        
        total_error = 0.0
        for t, U_exact in zip(times, U_exacts):
            U_approx = np.eye(64, dtype=complex)
            for k in range(12):
                # coeff for B is -1j * b[k] * t
                term_B = fast_expm_herm(-1j * b[k] * t, eigvals_B, eigvecs_B, inv_eigvecs_B)
                # coeff for A is -1j * a[k] * t
                term_A = fast_expm_herm(-1j * a[k] * t, eigvals_A, eigvecs_A, inv_eigvecs_A)
                U_approx = term_B @ term_A @ U_approx
                
            diff = U_approx - U_exact
            s = np.linalg.svd(diff, compute_uv=False)
            err = s[0] if len(s) > 0 else 0.0
            total_error += err
            
        return total_error / len(times)
