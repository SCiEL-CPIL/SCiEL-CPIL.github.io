def _polish(pos, radii, deadline):
    """High-precision local search using L-BFGS-B and SLSQP."""
    x0 = np.concatenate([pos.flatten(), radii])
    bounds = []
    for i in range(N):
        bounds.append((0.001, 0.999))
        bounds.append((0.001, 0.999))
    for i in range(N):
        bounds.append((1e-5, 0.5))
        
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    # 1. L-BFGS-B with tighter tolerances
    if time.time() < deadline - 5:
        try:
            result = minimize(lambda x: _objective(x, 10000.0), x0, method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-12, 'maxfun': 50000})
            
            x_opt = result.x
            pos_opt = x_opt[:2*N].reshape(N, 2)
            radii_opt = np.maximum(x_opt[2*N:], 1e-5)
            
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
            score = np.sum(radii_pp)
            
            if _is_valid(pos_pp, radii_pp) and score > best_score + 1e-12:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
                x0 = np.concatenate([best_pos.flatten(), best_radii])
        except Exception:
            pass
    
    # 2. SLSQP (often better for tight constraints)
    if time.time() < deadline - 2:
        try:
            result = minimize(lambda x: _objective(x, 10000.0), x0, method='SLSQP', bounds=bounds,
                              options={'maxiter': 1000, 'ftol': 1e-16, 'maxiter': 1000})
            
            x_opt = result.x
            pos_opt = x_opt[:2*N].reshape(N, 2)
            radii_opt = np.maximum(x_opt[2*N:], 1e-5)
            
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
            score = np.sum(radii_pp)
            
            if _is_valid(pos_pp, radii_pp) and score > best_score + 1e-12:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
        except Exception:
            pass
            
    # 3. Final L-BFGS-B pass from the best found
    if time.time() < deadline - 1:
        try:
            x0 = np.concatenate([best_pos.flatten(), best_radii])
            result = minimize(lambda x: _objective(x, 20000.0), x0, method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-14})
            
            x_opt = result.x
            pos_opt = x_opt[:2*N].reshape(N, 2)
            radii_opt = np.maximum(x_opt[2*N:], 1e-5)
            
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
            score = np.sum(radii_pp)
            
            if _is_valid(pos_pp, radii_pp) and score > best_score + 1e-12:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
        except Exception:
            pass
            
    return best_pos, best_radii, best_score
