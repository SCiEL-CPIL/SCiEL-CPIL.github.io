<<<<<<< SEARCH
    # If no good result, fallback
    if best_coeffs is None:
        best_coeffs = np.ones(21) / np.sqrt(21.0)
        
    # Ensure validity
    result = best_coeffs.astype(np.float64)
    if result.shape != (21,):
        result = np.ones(21, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        result = np.ones(21, dtype=np.float64)
        
    max_abs = np.max(np.abs(result))
    if max_abs < 1e-3:
        result = result / max_abs * 0.01 if max_abs > 0 else np.ones(21, dtype=np.float64)
    elif max_abs > 1e5:
        result = result / max_abs * 1e4
        
    return result
=======
    # If no good result, fallback
    if best_coeffs is None:
        best_coeffs = np.ones(21) / np.sqrt(21.0)
    
    # Final polish: run more iterations from the best point with tighter tolerance
    if time.time() < _DEADLINE - 30:
        c = best_coeffs.copy()
        prev_obj = np.inf
        for it in range(200):
            if time.time() > _DEADLINE - 10:
                break
            
            obj, grad = _objective_and_grad(c, H_flat, weights_flat)
            
            if abs(prev_obj - obj) < 1e-15:
                break
            prev_obj = obj
            
            # Very small steps for polish
            c_new = c - 0.005 * grad
            norm_c = np.linalg.norm(c_new)
            if norm_c > 1e-12:
                c_new /= norm_c
            else:
                break
            
            c = c_new
        
        obj, _ = _objective_and_grad(c, H_flat, weights_flat)
        if obj < best_obj:
            best_obj = obj
            best_coeffs = c.copy()
    
    # Ensure validity
    result = best_coeffs.astype(np.float64)
    if result.shape != (21,):
        result = np.ones(21, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        result = np.ones(21, dtype=np.float64)
        
    max_abs = np.max(np.abs(result))
    if max_abs < 1e-3:
        result = result / max_abs * 0.01 if max_abs > 0 else np.ones(21, dtype=np.float64)
    elif max_abs > 1e5:
        result = result / max_abs * 1e4
        
    return result
>>>>>>> REPLACE
