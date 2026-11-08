<<<<<<< SEARCH
        coeffs = np.zeros(deg + 1)
        for k in range(deg + 1):
            L_k_vals = eval_genlaguerre(k, 0, x_quad)  # Standard Laguerre L_k(x)
            coeffs[k] = np.sum(w_quad * P_vals * L_k_vals) / np.math.factorial(k)
=======
        coeffs = np.zeros(deg + 1)
        # Precompute factorials to avoid repeated calls and numpy.math issues
        factorials = np.array([1.0])
        for i in range(1, deg + 1):
            factorials.append(factorials[-1] * i)
        
        for k in range(deg + 1):
            L_k_vals = eval_genlaguerre(k, 0, x_quad)  # Standard Laguerre L_k(x)
            coeffs[k] = np.sum(w_quad * P_vals * L_k_vals) / factorials[k]
>>>>>>> REPLACE
<<<<<<< SEARCH
        # The most standard: C4 upper bound = ||P||_{L2(e^{-x})}^2 = sum_k |c_k|^2 * k!
        norm_sq = np.sum(np.abs(coeffs) ** 2 * np.array([np.math.factorial(k) for k in range(deg + 1)]))
        
        return norm_sq
=======
        # The most standard: C4 upper bound = ||P||_{L2(e^{-x})}^2 = sum_k |c_k|^2 * k!
        # Reuse precomputed factorials or recompute efficiently
        factorials = np.array([1.0])
        for i in range(1, deg + 1):
            factorials.append(factorials[-1] * i)
            
        norm_sq = np.sum(np.abs(coeffs) ** 2 * factorials)
        
        return norm_sq
>>>>>>> REPLACE
