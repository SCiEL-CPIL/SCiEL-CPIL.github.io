def _optimize_positions_for_radii(x0, y0, r, w, h, time_budget):
    """
    Optimize center positions given fixed radii using L-BFGS-B on a penalty function.
    """
    n = N
    z0 = np.concatenate([x0, y0])
    
    def penalty(z):
        x = z[:n]
        y = z[n:2*n]
        # Boundary penalties
        pen = 0.0
        for i in range(n):
            # Left
            if x[i] < r[i]:
                pen += (r[i] - x[i])**2
            # Right
            if x[i] > w - r[i]:
                pen += (x[i] - (w - r[i]))**2
            # Bottom
            if y[i] < r[i]:
                pen += (r[i] - y[i])**2
            # Top
            if y[i] > h - r[i]:
                pen += (y[i] - (h - r[i]))**2
        
        # Overlap penalties
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                dist = r[i] + r[j]
                if d < dist:
                    pen += (dist - d)**2
        return pen

    # We use L-BFGS-B which is fast and handles bounds well.
    # We need to be careful with local minima. 
    # We can run multiple restarts or just one long run.
    bounds = [(0, w)] * n + [(0, h)] * n
    
    try:
        res = minimize(penalty, z0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15})
        if res.fun < 1e-9: # Consider it valid if penalty is very small
            return res.x[:n], res.x[n:2*n]
    except Exception:
        pass
    return x0, y0
