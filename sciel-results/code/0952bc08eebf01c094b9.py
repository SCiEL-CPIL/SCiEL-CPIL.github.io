<<<<<<< SEARCH
def _final_polish(centers, time_limit=10.0):
    """High-precision final polish with very tight tolerances."""
    start_time = time.time()
    x0 = centers.flatten()
    
    # Use high beta for sharp minimum
    for beta in [5000, 10000, 20000]:
        if time.time() - start_time > time_limit:
            break
            
        def obj_grad(x):
            if time.time() - start_time > time_limit:
                return 1e10, np.zeros_like(x)
            val, grad = _softmin_objective_and_grad(x, beta)
            if not np.all(np.isfinite(grad)):
                return 1e10, np.zeros_like(x)
            return val, grad
            
        try:
            result = minimize(
                obj_grad, x0,
                method='L-BFGS-B',
                bounds=[(-CUBE_LIMIT, CUBE_LIMIT)] * (N * 3),
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 100}
            )
            x0 = result.x
        except Exception:
            pass
            
    return x0.reshape(N, 3)
=======
def _final_polish(centers, time_limit=10.0):
    """High-precision final polish with very tight tolerances."""
    start_time = time.time()
    x0 = centers.flatten()
    
    # Use high beta for sharp minimum - fewer steps
    for beta in [5000, 15000]:
        if time.time() - start_time > time_limit:
            break
            
        def obj_grad(x):
            if time.time() - start_time > time_limit:
                return 1e10, np.zeros_like(x)
            val, grad = _softmin_objective_and_grad(x, beta)
            if not np.all(np.isfinite(grad)):
                return 1e10, np.zeros_like(x)
            return val, grad
            
        try:
            result = minimize(
                obj_grad, x0,
                method='L-BFGS-B',
                bounds=[(-CUBE_LIMIT, CUBE_LIMIT)] * (N * 3),
                options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 50}
            )
            x0 = result.x
        except Exception:
            pass
            
    return x0.reshape(N, 3)
>>>>>>> REPLACE

