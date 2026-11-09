<<<<<<< SEARCH
# ---------------------------------------------------------------------------
# Nelder-Mead optimizer
# ---------------------------------------------------------------------------
def _nelder_mead(objective, x0, max_iter=200, tol=1e-10):
    n = len(x0)
    simplex = [x0[:]]
    for i in range(n):
        xi = x0[:]
        step = 0.1 if abs(xi[i]) < 1e-8 else 0.1 * np.sign(xi[i])
        xi[i] += step
        simplex.append(xi)
    fvals = [objective(x) for x in simplex]

    for it in range(max_iter):
        if _remaining() < 20:
            break
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]

        if fvals[-1] - fvals[0] < tol:
            break

        centroid = np.mean(simplex[:n], axis=0)
        x_ref = centroid + 1.0 * (centroid - simplex[n])
        f_ref = objective(x_ref)

        if f_ref < fvals[0]:
            x_exp = centroid + 2.0 * (centroid - simplex[n])
            f_exp = objective(x_exp)
            if f_exp < f_ref:
                simplex[n] = x_exp; fvals[n] = f_exp
            else:
                simplex[n] = x_ref; fvals[n] = f_ref
        elif f_ref < fvals[n-1]:
            simplex[n] = x_ref; fvals[n] = f_ref
        else:
            x_con = centroid + 0.5 * (simplex[n] - centroid)
            f_con = objective(x_con)
            if f_con < fvals[n]:
                simplex[n] = x_con; fvals[n] = f_con
            else:
                for i in range(1, n+1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    fvals[i] = objective(simplex[i])
    best_idx = int(np.argmin(fvals))
    return simplex[best_idx], fvals[best_idx]
=======
# ---------------------------------------------------------------------------
# Nelder-Mead optimizer
# ---------------------------------------------------------------------------
def _nelder_mead(objective, x0, max_iter=200, tol=1e-10):
    n = len(x0)
    simplex = [x0[:]]
    for i in range(n):
        xi = x0[:]
        step = 0.1 if abs(xi[i]) < 1e-8 else 0.1 * np.sign(xi[i])
        xi[i] += step
        simplex.append(xi)
    fvals = [objective(x) for x in simplex]

    for it in range(max_iter):
        if _remaining() < 20:
            break
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]

        if fvals[-1] - fvals[0] < tol:
            break

        centroid = np.mean(simplex[:n], axis=0)
        x_ref = centroid + 1.0 * (centroid - simplex[n])
        f_ref = objective(x_ref)

        if f_ref < fvals[0]:
            x_exp = centroid + 2.0 * (centroid - simplex[n])
            f_exp = objective(x_exp)
            if f_exp < f_ref:
                simplex[n] = x_exp; fvals[n] = f_exp
            else:
                simplex[n] = x_ref; fvals[n] = f_ref
        elif f_ref < fvals[n-1]:
            simplex[n] = x_ref; fvals[n] = f_ref
        else:
            x_con = centroid + 0.5 * (simplex[n] - centroid)
            f_con = objective(x_con)
            if f_con < fvals[n]:
                simplex[n] = x_con; fvals[n] = f_con
            else:
                for i in range(1, n+1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    fvals[i] = objective(simplex[i])
    best_idx = int(np.argmin(fvals))
    return simplex[best_idx], fvals[best_idx]


def _lbfgsb(objective, x0, max_iter=100):
    """L-BFGS-B optimization with deadline guard."""
    n = len(x0)
    bounds = [(-50.0, 50.0)] * n
    result = _scipy_minimize(
        objective, x0, method='L-BFGS-B', bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    return result.x, result.fun
>>>>>>> REPLACE

