    def objective(params):
        roots = np.sort(params)
        if np.any(roots <= 0):
            return 1e10
        if np.any(np.diff(roots) <= 1e-8):
            return 1e10
            
        coeffs = _get_laguerre_coefficients(roots)
        
        # Penalty for negative coeffs
        neg_min = np.min(coeffs)
        if neg_min < 0:
            # Strong penalty
            return 1e6 * (-neg_min) + np.max(roots)
            
        return np.max(roots)
    