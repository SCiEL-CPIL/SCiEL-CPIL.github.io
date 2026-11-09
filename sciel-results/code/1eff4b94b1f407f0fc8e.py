<<<<<<< SEARCH
def _high_precision_polish(pts, deadline):
    """Final high-precision polish with very tight tolerances."""
    pts = _project_to_sphere(pts.copy())
    
    def objective_and_grad(flat_pts):
        p = flat_pts.reshape(N, 3)
        p = _project_to_sphere(p)
        e, g = _energy_and_gradient(p)
        g_tangent = g - np.sum(g * p, axis=1, keepdims=True) * p
        return e, g_tangent.flatten()
    
    x0 = pts.flatten()
    try:
        result = minimize(
            objective_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 100}
        )
        if time.time() > deadline:
            return pts, _log_energy(pts)
        opt_pts = _project_to_sphere(result.x.reshape(N, 3))
        opt_energy = _log_energy(opt_pts)
        return opt_pts, opt_energy
    except Exception:
        return pts, _log_energy(pts)
=======
def _high_precision_polish(pts, deadline):
    """Final high-precision polish with very tight tolerances and multiple passes."""
    pts = _project_to_sphere(pts.copy())
    
    def objective_and_grad(flat_pts):
        p = flat_pts.reshape(N, 3)
        p = _project_to_sphere(p)
        e, g = _energy_and_gradient(p)
        g_tangent = g - np.sum(g * p, axis=1, keepdims=True) * p
        return e, g_tangent.flatten()
    
    x0 = pts.flatten()
    current_energy = _log_energy(pts)
    
    # Multiple polish passes with decreasing tolerances
    for pass_num, (maxiter, ftol, gtol) in enumerate([
        (500, 1e-14, 1e-12),
        (500, 1e-15, 1e-13),
        (500, 1e-16, 1e-14)
    ]):
        if time.time() > deadline:
            break
        
        try:
            result = minimize(
                objective_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol, 'maxls': 100}
            )
            if time.time() > deadline:
                break
            opt_pts = _project_to_sphere(result.x.reshape(N, 3))
            opt_energy = _log_energy(opt_pts)
            
            if opt_energy < current_energy:
                pts = opt_pts
                current_energy = opt_energy
                x0 = opt_pts.flatten()
        except Exception:
            break
    
    return pts, current_energy
>>>>>>> REPLACE

