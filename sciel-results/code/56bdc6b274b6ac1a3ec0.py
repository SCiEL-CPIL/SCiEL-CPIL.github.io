def _nelder_mead(func, x0, bounds=None, max_iter=200, tol=1e-8):
    n = len(x0)
    # Initialize simplex
    simplex = [x0]
    for i in range(n):
        x = x0.copy()
        step = 0.1 if x[i] == 0 else 0.1 * abs(x[i])
        x[i] += step
        simplex.append(x)
    
    # Evaluate initial
    vals = [func(x) for x in simplex]
    
    for _ in range(max_iter):
        # Sort
        order = np.argsort(vals)
        simplex = [simplex[i] for i in order]
        vals = [vals[i] for i in order]
        
        # Check convergence
        if abs(vals[-1] - vals[0]) < tol:
            break
            
        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        x_r = centroid + 0.5 * (centroid - simplex[-1])
        f_r = func(x_r)
        
        if f_r < vals[0]:
            # Expansion
            x_e = centroid + 1.5 * (centroid - simplex[-1])
            f_e = func(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                vals[-1] = f_e
            else:
                simplex[-1] = x_r
                vals[-1] = f_r
        elif f_r < vals[-2]:
            simplex[-1] = x_r
            vals[-1] = f_r
        else:
            # Contraction
            x_c = centroid - 0.5 * (centroid - simplex[-1])
            f_c = func(x_c)
            if f_c < vals[-1]:
                simplex[-1] = x_c
                vals[-1] = f_c
            else:
                # Shrink
                for i in range(1, n+1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    vals[i] = func(simplex[i])
                    
    # Return best
    best_idx = np.argmin(vals)
    return simplex[best_idx], vals[best_idx]
