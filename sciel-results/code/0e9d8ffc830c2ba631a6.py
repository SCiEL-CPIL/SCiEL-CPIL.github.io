<<<<<<< SEARCH
def _is_feasible(x, w, tol=1e-6):
    h = 2.0 - w
    r = x[:N]
    cx = x[N:2 * N]
    cy = x[2 * N:3 * N]
    if np.any(r < -tol):
        return False
    if np.any(cx < r - tol) or np.any(cx > w - r + tol) or np.any(cy < r - tol) or np.any(cy > h - r + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            d2 = (cx[i] - cx[j]) ** 2 + (cy[i] - cy[j]) ** 2
            if d2 < (r[i] + r[j]) ** 2 - tol:
                return False
    return True


def _sum_radii(x):
    return float(np.sum(x[:N]))
=======
def _is_feasible(x, w, tol=1e-6):
    h = 2.0 - w
    r = x[:N]
    cx = x[N:2 * N]
    cy = x[2 * N:3 * N]
    if np.any(r < -tol):
        return False
    if np.any(cx < r - tol) or np.any(cx > w - r + tol) or np.any(cy < r - tol) or np.any(cy > h - r + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            d2 = (cx[i] - cx[j]) ** 2 + (cy[i] - cy[j]) ** 2
            if d2 < (r[i] + r[j]) ** 2 - tol:
                return False
    return True


def _sum_radii(x):
    return float(np.sum(x[:N]))


def _slsqp_solve(seed, w, h, maxiter=500, ftol=1e-12):
    """Run SLSQP and return the best feasible solution found."""
    cons = _make_constraints(w, h)
    bounds = []
    for i in range(N):
        bounds.append((1e-4, min(w, h) / 2.0))
    for i in range(N):
        bounds.append((0.0, w))
    for i in range(N):
        bounds.append((0.0, h))
    try:
        res = minimize(
            lambda x: _objective(x, w),
            seed,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': maxiter, 'ftol': ftol},
        )
        return res.x
    except Exception:
        return seed
>>>>>>> REPLACE

