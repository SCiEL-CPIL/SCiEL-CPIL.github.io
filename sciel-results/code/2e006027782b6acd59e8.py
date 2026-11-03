<<<<<<< SEARCH
def _valid(x, y, r, eps=1e-6):
    d = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    if np.any(r + eps > d):
        return False
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    need = r[:, None] + r[None, :]
    if np.any(need + eps > dist):
        return False
    return True


def construct():
    best = None
    best_val = -np.inf
    rng = np.random.default_rng(12345)

    inits = [
        _grid_init_5x5_plus1,
        _grid_init_4x6,
        _grid_init_6x4,
        _random_init,
        _hex_init,
    ]

    for init_func in inits:
        if _elapsed() > DEADLINE - 5.0:
            break
        x, y = init_func(rng)
        x = np.clip(x, 0.02, 0.98)
        y = np.clip(y, 0.02, 0.98)
        
        for restart in range(3):
            if _elapsed() > DEADLINE - 5.0:
                break
            x0 = x.copy()
            y0 = y.copy()
            jit = rng.normal(0.0, 0.02, size=N)
            x0 = np.clip(x0 + jit, 0.02, 0.98)
            y0 = np.clip(y0 + jit, 0.02, 0.98)
            
            vec0 = np.zeros(2 * N)
            vec0[0::2] = x0
            vec0[1::2] = y0

            try:
                res = minimize(_objective_positions, vec0, method='L-BFGS-B',
                               options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-6,
                                        'maxfun': 5000})
            except Exception:
                continue
            
            v = res.x
            xx = np.clip(v[0::2], 0.01, 0.99)
            yy = np.clip(v[1::2], 0.01, 0.99)
            rr = _compute_radii(xx, yy)
            
            # Final fix: ensure no overlaps by shrinking
            for _ in range(5):
                changed = False
                d_bound = np.minimum(np.minimum(xx, 1.0 - xx), np.minimum(yy, 1.0 - yy))
                rr = np.minimum(rr, d_bound)
                
                dx = xx[:, None] - xx[None, :]
                dy = yy[:, None] - yy[None, :]
                dist = np.sqrt(dx * dx + dy * dy)
                np.fill_diagonal(dist, np.inf)
                
                for i in range(N):
                    for j in range(i + 1, N):
                        if dist[i, j] < rr[i] + rr[j]:
                            # Shrink both proportionally
                            scale = dist[i, j] / (rr[i] + rr[j]) * 0.99
                            rr[i] *= scale
                            rr[j] *= scale
                            changed = True
                if not changed:
                    break
            
            if _valid(xx, yy, rr):
                val = float(np.sum(rr))
                if val > best_val:
                    best_val = val
                    best = np.column_stack([xx, yy, rr])

    if best is not None:
        return best

    # Guaranteed fallback
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
             for i in range(side) for j in range(side)]
    cells = cells[:N]
    while len(cells) < N:
        cells.append((0.5 / side, 0.5 / side, 0.1 / side))
    return np.array(cells[:N])
=======
def _fix_and_shrink(x, y, r, max_iter=20):
    """Ensure validity by shrinking radii and adjusting positions."""
    x = x.copy()
    y = y.copy()
    r = r.copy()
    
    for _ in range(max_iter):
        changed = False
        # Boundary fix
        for i in range(N):
            max_r = min(x[i], 1.0 - x[i], y[i], 1.0 - y[i])
            if r[i] > max_r:
                r[i] = max_r * 0.999
                changed = True
        
        # Overlap fix
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(dist, np.inf)
        
        for i in range(N):
            for j in range(i + 1, N):
                if dist[i, j] < r[i] + r[j]:
                    scale = dist[i, j] / (r[i] + r[j]) * 0.999
                    r[i] *= scale
                    r[j] *= scale
                    changed = True
        
        if not changed:
            break
    
    return x, y, r


def _polish(params, max_iters=100):
    """Local polish with gradient-based steps."""
    x = params[0::3].copy()
    y = params[1::3].copy()
    r = params[2::3].copy()
    
    for it in range(max_iters):
        if _elapsed() > DEADLINE - 3.0:
            break
        
        # Check validity
        if _is_valid(x, y, r, eps=1e-4):
            # Try to grow radii slightly
            for i in range(N):
                max_r = min(x[i], 1.0 - x[i], y[i], 1.0 - y[i])
                dx = x[i] - x
                dy = y[i] - y
                dist = np.sqrt(dx * dx + dy * dy)
                np.fill_diagonal(dist, np.inf)
                min_dist = np.min(dist)
                new_r = min(max_r, 0.5 * min_dist)
                if new_r > r[i] * 1.001:
                    r[i] = new_r
                    break
            else:
                break
        else:
            # Shrink problematic circles
            x, y, r = _fix_and_shrink(x, y, r, max_iter=5)
    
    return np.column_stack([x, y, r])


def construct():
    best = None
    best_val = -np.inf
    rng = np.random.default_rng(12345)

    # Build list of initializations
    init_params_list = []
    
    # Grid 5x5 + 1
    init_params_list.append(_init_params_grid(rng, 5, 5))
    # Grid 4x6
    init_params_list.append(_init_params_grid(rng, 4, 6))
    # Grid 6x4
    init_params_list.append(_init_params_grid(rng, 6, 4))
    # Hex
    init_params_list.append(_init_params_hex(rng))
    # Random
    for _ in range(3):
        init_params_list.append(_init_params_random(rng))

    for p0 in init_params_list:
        if _elapsed() > DEADLINE - 10.0:
            break
        
        # Add small perturbation for diversity
        jit = rng.normal(0.0, 0.01, size=3 * N)
        p0_jit = p0 + jit
        p0_jit[0::3] = np.clip(p0_jit[0::3], 0.02, 0.98)
        p0_jit[1::3] = np.clip(p0_jit[1::3], 0.02, 0.98)
        p0_jit[2::3] = np.maximum(p0_jit[2::3], 0.01)

        try:
            res = minimize(_objective, p0_jit, jac=_gradient, method='L-BFGS-B',
                           options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8,
                                    'maxfun': 3000})
        except Exception:
            continue
        
        v = res.x
        xx = np.clip(v[0::3], 0.0, 1.0)
        yy = np.clip(v[1::3], 0.0, 1.0)
        rr = np.maximum(v[2::3], 0.0)
        
        # Fix and polish
        xx, yy, rr = _fix_and_shrink(xx, yy, rr)
        result = _polish(np.column_stack([xx, yy, rr]))
        xx, yy, rr = result[:, 0], result[:, 1], result[:, 2]
        
        if _is_valid(xx, yy, rr):
            val = float(np.sum(rr))
            if val > best_val:
                best_val = val
                best = np.column_stack([xx, yy, rr])

    if best is not None:
        return best

    # Guaranteed fallback: 5x5 grid with 1 extra
    side = 5
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
             for i in range(side) for j in range(side)]
    cells = cells[:N]
    while len(cells) < N:
        cells.append((0.5 / side, 0.5 / side, 0.05 / side))
    return np.array(cells[:N])
>>>>>>> REPLACE
