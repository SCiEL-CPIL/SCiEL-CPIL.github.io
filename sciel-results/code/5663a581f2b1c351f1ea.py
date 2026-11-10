<<<<<<< SEARCH
    # Initial guess: Keich construction from the prompt
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    x_keich = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        x_keich.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    x0 = np.array(x_keich, dtype=np.float64)
    
    # Optimize
    # Powell is good for non-smooth functions
    res = minimize(area, x0, method='Powell', options={'maxiter': 1000, 'disp': False})
    
    return res.x
=======
    def objective(x_vars):
        # x_vars is shape (n-1,), corresponding to x_1..x_{n-1}
        # x_0 is fixed at 0
        x_full = np.empty(n)
        x_full[0] = 0.0
        x_full[1:] = x_vars
        return area(x_full)

    # Bounds for x_1..x_{63}
    bounds = [(-1.0, 2.0)] * (n - 1)
    
    # Fallback initial guess (Keich, shifted)
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    x_keich = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        x_keich.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    x_keich = np.array(x_keich, dtype=np.float64)
    x_keich_shifted = x_keich - x_keich[0]
    x0_vars = x_keich_shifted[1:]

    # Run Differential Evolution for global search
    try:
        res_de = differential_evolution(objective, bounds, maxiter=50, popsize=10, seed=42, tol=0.01, polish=False)
        best_x_vars = res_de.x
    except Exception:
        best_x_vars = x0_vars

    # Refine with Powell
    res_powell = minimize(objective, best_x_vars, method='Powell', options={'maxiter': 200, 'disp': False})
    
    x_final_vars = res_powell.x
    
    x_final = np.empty(n)
    x_final[0] = 0.0
    x_final[1:] = x_final_vars
    
    return x_final
>>>>>>> REPLACE
