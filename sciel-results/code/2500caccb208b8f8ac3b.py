<<<<<<< SEARCH
    # --- Phase 2: Refinement ---
    
    # We refine the top candidates.
    # We will use SLSQP on the normalized vector to handle the non-smooth max better.
    
    top_candidates = candidates[:20]
    
    for sc, h_init in top_candidates:
        if elapsed() > deadline - 2:
            break
            
        # Normalize h_init
        h_norm_init = h_init / np.sum(h_init)
        
        # Constraints for SLSQP: sum(h) = 1, h >= 0
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
        bounds = [(0, 1)] * N
        
        try:
            # Use SLSQP with finite difference gradient
            # This is robust to the kink in the max function
            res = minimize(neg_ratio_normalized, h_norm_init, method='SLSQP',
                          bounds=bounds, constraints=constraints,
                          options={'maxiter': 200, 'ftol': 1e-12})
            
            if elapsed() > deadline - 1:
                break
                
            h_opt = res.x
            h_opt = np.maximum(h_opt, 0.0)
            if np.sum(h_opt) > 1e-15:
                h_opt = h_opt / np.sum(h_opt)
            
            sc_opt = compute_ratio(h_opt)
            if sc_opt > best_score:
                best_score = sc_opt
                best_h = h_opt.copy()
                
        except Exception:
            pass
=======
    # --- Phase 2: Refinement ---
    
    # We refine the top candidates.
    # We will use SLSQP on the normalized vector to handle the non-smooth max better.
    
    top_candidates = candidates[:30]
    
    for sc, h_init in top_candidates:
        if elapsed() > deadline - 3:
            break
            
        # Normalize h_init
        h_norm_init = h_init / np.sum(h_init)
        
        # Constraints for SLSQP: sum(h) = 1, h >= 0
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
        bounds = [(0, 1)] * N
        
        try:
            # Use SLSQP with finite difference gradient
            # This is robust to the kink in the max function
            res = minimize(neg_ratio_normalized, h_norm_init, method='SLSQP',
                          bounds=bounds, constraints=constraints,
                          options={'maxiter': 300, 'ftol': 1e-10})
            
            if elapsed() > deadline - 2:
                break
                
            h_opt = res.x
            h_opt = np.maximum(h_opt, 0.0)
            if np.sum(h_opt) > 1e-15:
                h_opt = h_opt / np.sum(h_opt)
            
            sc_opt = compute_ratio(h_opt)
            if sc_opt > best_score:
                best_score = sc_opt
                best_h = h_opt.copy()
                
        except Exception:
            pass
        
        # Also try L-BFGS-B with smooth surrogate for better gradient info
        if elapsed() > deadline - 2:
            break
        try:
            # Convert to log-space for L-BFGS-B
            h_safe = np.maximum(h_norm_init, 1e-10)
            log_h_init = np.log(h_safe)
            
            res2 = minimize(neg_ratio_smooth, log_h_init, method='L-BFGS-B',
                           options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if elapsed() > deadline - 2:
                break
                
            h_opt2 = np.exp(res2.x)
            h_opt2 = np.maximum(h_opt2, 0.0)
            if np.sum(h_opt2) > 1e-15:
                h_opt2 = h_opt2 / np.sum(h_opt2)
            
            sc_opt2 = compute_ratio(h_opt2)
            if sc_opt2 > best_score:
                best_score = sc_opt2
                best_h = h_opt2.copy()
                
        except Exception:
            pass
>>>>>>> REPLACE

