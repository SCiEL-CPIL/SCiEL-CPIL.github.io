<<<<<<< SEARCH
def _lbfgs_violation(pts0, r_target, deadline):
    """Use L-BFGS-B to minimize violation for target radius."""
    from scipy.optimize import minimize
    
    def obj(x):
        if time.time() > deadline:
            raise RuntimeError("timeout")
        p = x.reshape(N, 2)
        return _violation(p, r_target)
    
    def grad(x):
        if time.time() > deadline:
            raise RuntimeError("timeout")
        p = x.reshape(N, 2)
        # Numerical gradient
        eps = 1e-6
        g = np.zeros_like(x)
        base = obj(x)
        for i in range(x.size):
            xp = x.copy()
            xp[i] += eps
            g[i] = (obj(xp) - base) / eps
        return g
    
    x0 = pts0.ravel()
    res = minimize(obj, x0, method='L-BFGS-B',
                   jac=grad,
                   bounds=[(-HALF, HALF)] * (2 * N),
                   options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-8})
    pts = res.x.reshape(N, 2)
    pts = np.clip(pts, -HALF, HALF)
    return pts, float(res.fun)
=======
def _lbfgs_violation(pts0, r_target, deadline):
    """Use L-BFGS-B to minimize violation for target radius."""
    from scipy.optimize import minimize
    
    def obj(x):
        if time.time() > deadline:
            raise RuntimeError("timeout")
        p = x.reshape(N, 2)
        v, _ = _violation_and_grad(p, r_target)
        return v
    
    def grad(x):
        if time.time() > deadline:
            raise RuntimeError("timeout")
        p = x.reshape(N, 2)
        _, g = _violation_and_grad(p, r_target)
        return g.ravel()
    
    x0 = pts0.ravel()
    res = minimize(obj, x0, method='L-BFGS-B',
                   jac=grad,
                   bounds=[(-HALF, HALF)] * (2 * N),
                   options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-6})
    pts = res.x.reshape(N, 2)
    pts = np.clip(pts, -HALF, HALF)
    return pts, float(res.fun)
>>>>>>> REPLACE

