<<<<<<< SEARCH
    # === Phase 5: Fine-grained coordinate refinement on best found ===
    if time.time() < deadline - 10.0:
=======
    # === Phase 4d: DPSS spectral coefficient optimization ===
    # DPSS (Slepian) sequences have optimal time-frequency concentration.
    # Represent h as nonnegative projection of linear combination of DPSS.
    # Optimize spectral coefficients via L-BFGS-B.
    
    def dpss_basis(N_basis, W_prod):
        """Generate DPSS basis functions on X grid."""
        basis = np.zeros((N_basis, N))
        for k in range(N_basis):
            # dpss(n, W, k) gives the k-th DPSS for n samples with bandwidth W
            # We want to map this to our X grid
            dpss_vals = dpss(N, W_prod, k)
            # Interpolate/resize to our N points if needed
            # Actually dpss(N, W, k) already gives N values
            basis[k] = dpss_vals
        return basis
    
    def dpss_model_to_h(coeffs, N_basis, W_prod):
        """h = max(0, sum_k coeffs[k] * dpss_k(x))"""
        basis = dpss_basis(N_basis, W_prod)
        h = basis.T @ coeffs
        return np.clip(h, 0.0, None)
    
    def dpss_ratio(params, N_basis, W_prod):
        h = dpss_model_to_h(params, N_basis, W_prod)
        return ratio_from_h(h)
    
    # Try different time-bandwidth products and basis sizes
    dpss_configs = [
        (10, 0.1), (15, 0.15), (20, 0.2), (10, 0.2), (20, 0.3), (30, 0.4)
    ]
    
    for N_basis, W_prod in dpss_configs:
        if time.time() > deadline - 15.0:
            break
        
        n_starts = 8
        for start in range(n_starts):
            if time.time() > deadline - 12.0:
                break
            
            # Initialize with random coefficients, biased toward positive
            coeffs = np.zeros(N_basis)
            coeffs[0] = float(rng.uniform(0.5, 2.0))
            for k in range(1, N_basis):
                coeffs[k] = float(rng.uniform(-0.5, 0.5))
            
            try:
                result = opt.minimize(
                    lambda c: dpss_ratio(c, N_basis, W_prod),
                    coeffs,
                    method='L-BFGS-B',
                    bounds=[(-10.0, 10.0)] * N_basis,
                    options={'maxiter': 200, 'gtol': 1e-6, 'ftol': 1e-8}
                )
                if result.fun < best_val:
                    h_opt = dpss_model_to_h(result.x, N_basis, W_prod)
                    update_best(h_opt, result.fun)
            except Exception:
                pass
    
    # === Phase 4e: DPSS with soft-max smooth surrogate for better gradients ===
    def dpss_soft_ratio(params, N_basis, W_prod, beta=0.05):
        h = dpss_model_to_h(params, N_basis, W_prod)
        h = np.clip(h, 1e-10, None)
        f = h[CELL]
        F = np.fft.rfft(f, N_FFT)
        conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1] * DX
        l2 = np.sum(f * f) * DX
        if l2 < 1e-15:
            return 1e10
        max_conv = np.max(conv)
        soft_max_val = max_conv + beta * np.log(np.sum(np.exp((conv - max_conv) / beta)))
        return float(soft_max_val / l2)
    
    for N_basis, W_prod in [(15, 0.15), (20, 0.2)]:
        if time.time() > deadline - 10.0:
            break
        
        n_starts = 5
        for start in range(n_starts):
            if time.time() > deadline - 8.0:
                break
            
            coeffs = np.zeros(N_basis)
            coeffs[0] = float(rng.uniform(0.5, 2.0))
            for k in range(1, N_basis):
                coeffs[k] = float(rng.uniform(-0.5, 0.5))
            
            try:
                result = opt.minimize(
                    lambda c: dpss_soft_ratio(c, N_basis, W_prod, 0.05),
                    coeffs,
                    method='L-BFGS-B',
                    bounds=[(-10.0, 10.0)] * N_basis,
                    options={'maxiter': 300, 'gtol': 1e-7, 'ftol': 1e-9}
                )
                if result.fun < best_val:
                    h_opt = dpss_model_to_h(result.x, N_basis, W_prod)
                    v = ratio_from_h(h_opt)
                    update_best(h_opt, v)
            except Exception:
                pass
    
    # === Phase 5: Fine-grained coordinate refinement on best found ===
    if time.time() < deadline - 10.0:
>>>>>>> REPLACE
