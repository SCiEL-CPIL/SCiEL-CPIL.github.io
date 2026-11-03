def _energy_and_grad_flat(x):
    P = x.reshape(N, 3)
    e, g = _energy_and_grad(P)
    return e, g.flatten()

def _refine(P0, deadline):
    best = _project(P0.copy())
    be, _ = _energy_and_grad(best)
    x0 = best.flatten()
    for _ in range(4):
        if time.time() > deadline:
            break
        try:
            res = minimize(
                _energy_and_grad_flat, x0, jac=True,
                method="L-BFGS-B",
                options={"maxiter": 300, "gtol": 1e-8, "maxfun": 3000},
            )
        except Exception:
            break
        if res.x is None:
            break
        cand = _project(np.asarray(res.x, dtype=float).reshape(N, 3))
        if not np.all(np.isfinite(cand)):
            break
        ce, _ = _energy_and_grad(cand)
        if np.isfinite(ce) and ce < be - 1e-12:
            best, be = cand, ce
            x0 = best.flatten()
        else:
            break
    return best, be
