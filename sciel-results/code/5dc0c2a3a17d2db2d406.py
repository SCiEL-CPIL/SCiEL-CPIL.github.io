<<<<<<< SEARCH
    # A few distinct initial structures / radii for diversity.
    inits = [
        (5, 0.18),   # 5x5 grid, modest radius
        (5, 0.20),
        (5, 0.16),
    ]

    for (k, r0) in inits:
        if _elapsed() > DEADLINE - 5.0:
            break
        x, y, r = _grid_init(rng, k, r0)
        r = _fix_to_boundary(x, y, r)
        vec0 = np.concatenate([x, y, r])

        for restart in range(3):
            if _elapsed() > DEADLINE - 5.0:
                break
            x0 = x.copy(); y0 = y.copy(); r0v = r.copy()
            # add small jitter for multi-start
            jit = rng.normal(0.0, 0.01, size=N)
            x0 = np.clip(x0 + jit, 1e-3, 1 - 1e-3)
            y0 = np.clip(y0 + jit, 1e-3, 1 - 1e-3)
            vec0 = np.concatenate([x0, y0, r0v])

            def objective(v):
                xx = np.clip(v[:N], 1e-6, 1 - 1e-6)
                yy = np.clip(v[N:2*N], 1e-6, 1 - 1e-6)
                rr = np.clip(v[2*N:3*N], 1e-6, 0.5)
                return _penalized(None, xx, yy, rr)

            try:
                res = minimize(objective, vec0, method='Nelder-Mead',
                               options={'maxiter': 4000, 'xatol': 1e-6, 'fatol': 1e-8,
                                        'maxfev': 6000})
            except Exception:
                continue
            v = res.x
            xx = np.clip(v[:N], 1e-6, 1 - 1e-6)
            yy = np.clip(v[N:2*N], 1e-6, 1 - 1e-6)
            rr = np.clip(v[2*N:3*N], 1e-6, 0.5)
            # clamp to square to guarantee boundary feasibility
            rr = _fix_to_boundary(xx, yy, rr)
            if _valid(xx, yy, rr):
                val = float(np.sum(rr))
                if val > best_val:
                    best_val = val
                    best = np.column_stack([xx, yy, rr])
            elif best is None:
                # fallback: at least produce a feasible trivial packing
                pass
=======
    # Hexagonal packing allows for denser arrangements.
    # Start with a radius that fits in the hex grid spacing (~0.2)
    # Max radius for spacing 0.2 is 0.1. We can grow them.
    inits = [
        (0.10),   # Conservative start
        (0.12),   # Aggressive start
        (0.08),   # Loose start
    ]

    for r0 in inits:
        if _elapsed() > DEADLINE - 5.0:
            break
        x, y, r = _hex_init(rng, r0)
        r = _fix_to_boundary(x, y, r)
        
        for restart in range(5):
            if _elapsed() > DEADLINE - 5.0:
                break
            x0 = x.copy(); y0 = y.copy(); r0v = r.copy()
            # add small jitter for multi-start
            jit = rng.normal(0.0, 0.02, size=N)
            x0 = np.clip(x0 + jit, 1e-3, 1 - 1e-3)
            y0 = np.clip(y0 + jit, 1e-3, 1 - 1e-3)
            vec0 = np.concatenate([x0, y0, r0v])

            def objective(v):
                xx = np.clip(v[:N], 1e-6, 1 - 1e-6)
                yy = np.clip(v[N:2*N], 1e-6, 1 - 1e-6)
                rr = np.clip(v[2*N:3*N], 1e-6, 0.5)
                # Higher penalty weights to ensure feasibility is prioritized
                return _penalized(None, xx, yy, rr, w_overlap=100.0, w_bound=100.0)

            try:
                res = minimize(objective, vec0, method='Nelder-Mead',
                               options={'maxiter': 5000, 'xatol': 1e-7, 'fatol': 1e-9,
                                        'maxfev': 8000})
            except Exception:
                continue
            v = res.x
            xx = np.clip(v[:N], 1e-6, 1 - 1e-6)
            yy = np.clip(v[N:2*N], 1e-6, 1 - 1e-6)
            rr = np.clip(v[2*N:3*N], 1e-6, 0.5)
            # clamp to square to guarantee boundary feasibility
            rr = _fix_to_boundary(xx, yy, rr)
            if _valid(xx, yy, rr):
                val = float(np.sum(rr))
                if val > best_val:
                    best_val = val
                    best = np.column_stack([xx, yy, rr])
            elif best is None:
                # fallback: at least produce a feasible trivial packing
                pass
>>>>>>> REPLACE
