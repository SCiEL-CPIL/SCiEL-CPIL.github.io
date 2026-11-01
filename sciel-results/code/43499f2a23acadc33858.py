    def _c3_upper_bound(h):
        # h is the input to construct, but we ignore it and optimize internally?
        # No, construct() calls _c3_upper_bound?
        # Current seed: construct() calls _c3_upper_bound(np.full(N, 5.0))
        # I should keep this structure.
        
        # Inside _c3_upper_bound, I will perform the optimization.
        # The input argument is effectively ignored or used as seed.
        
        def objective(x):
            h = np.tanh(x)
            sum_h = np.sum(h)
            sum_abs_h = np.sum(np.abs(h))
            # Metric: (sum_h)^2 / (sum_abs_h)^2
            # To minimize, we want sum_h -> 0.
            # Value = (sum_h)^2 / (sum_abs_h)^2
            # Smooth version:
            val = (sum_h**2 + 1e-10) / (sum_abs_h**2 + 1e-10)
            return val
            
        # Initialize with random signs
        x0 = np.random.uniform(-5, 5, N)
        
        res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 500})
        
        h_opt = np.tanh(res.x)
        return h_opt
    