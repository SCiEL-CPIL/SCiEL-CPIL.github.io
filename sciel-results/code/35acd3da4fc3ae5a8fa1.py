<<<<<<< SEARCH
    try:
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
=======
    try:
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 600, 'ftol': 1e-14, 'gtol': 1e-12})
>>>>>>> REPLACE
