def _local_minimize(x0_flat, maxiter=1000):
    def fun(x):
        pts = x.reshape(-1, 3)
        pts = _project_to_sphere(pts)
        return _log_energy_vec(pts)
    
    def jac(x):
        pts = x.reshape(-1, 3)
        pts = _project_to_sphere(pts)
        g = _log_energy_grad(pts)
        for i in range(N):
            g[i] -= np.dot(g[i], pts[i]) * pts[i]
        return g.flatten()

    try:
        result = minimize(
            fun, x0_flat, jac=jac,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'gtol': 1e-10, 'ftol': 1e-12}
        )
        if np.all(np.isfinite(result.x)):
            return result.x, result.fun
        else:
            return x0_flat, _log_energy_vec(x0_flat.reshape(-1, 3))
    except Exception:
        return x0_flat, _log_energy_vec(x0_flat.reshape(-1, 3))
