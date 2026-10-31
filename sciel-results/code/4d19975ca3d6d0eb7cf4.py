<<<<<<< SEARCH
def _slsqp_polish(pos, radii, deadline):
    """SLSQP polishing with explicit constraints."""
    pos = pos.copy()
    radii = radii.copy()
    pos, radii = _post_process(pos, radii, deadline)
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)

    def objective(x):
        p = x[:2*N].reshape(N, 2)
        r = x[2*N:]
        return -np.sum(r)

    def make_constraint(i, j):
        def constraint(x):
            p = x[:2*N].reshape(N, 2)
            r = x[2*N:]
            dx = p[i, 0] - p[j, 0]
            dy = p[i, 1] - p[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            return dist - r[i] - r[j]
        return constraint

    # Boundary constraints
    boundary_cons = []
    for i in range(N):
        boundary_cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i] - x[2*N+i]})
        boundary_cons.append({'type': 'ineq', 'fun': lambda x, i=i: (1.0 - x[2*N+i]) - x[2*i]})
        boundary_cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1] - x[2*N+i]})
        boundary_cons.append({'type': 'ineq', 'fun': lambda x, i=i: (1.0 - x[2*N+i]) - x[2*i+1]})

    # Overlap constraints
    overlap_cons = []
    for i in range(N):
        for j in range(i + 1, N):
            overlap_cons.append({'type': 'ineq', 'fun': make_constraint(i, j)})

    x0 = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.005, 0.995))
        bounds.append((0.005, 0.995))
    for i in range(N):
        bounds.append((0.005, 0.35))

    try:
        result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                          constraints=boundary_cons + overlap_cons,
                          options={'maxiter': 300, 'ftol': 1e-12})
        if time.time() < deadline:
            x_opt = result.x
            p_opt = x_opt[:2*N].reshape(N, 2)
            r_opt = x_opt[2*N:]
            p_opt, r_opt = _post_process(p_opt, r_opt, deadline)
            if _is_valid(p_opt, r_opt):
                s = np.sum(r_opt)
                if s > best_score:
                    best_score = s
                    best_pos = p_opt.copy()
                    best_radii = r_opt.copy()
    except Exception:
        pass

    return best_pos, best_radii, best_score
=======
def _slsqp_polish(pos, radii, deadline):
    """SLSQP polishing with explicit constraints."""
    pos = pos.copy()
    radii = radii.copy()
    pos, radii = _post_process(pos, radii, deadline)
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)

    def objective(x):
        p = x[:2*N].reshape(N, 2)
        r = x[2*N:]
        return -np.sum(r)

    def make_constraint(i, j):
        def constraint(x):
            p = x[:2*N].reshape(N, 2)
            r = x[2*N:]
            dx = p[i, 0] - p[j, 0]
            dy = p[i, 1] - p[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            return dist - r[i] - r[j]
        return constraint

    # Boundary constraints
    boundary_cons = []
    for i in range(N):
        boundary_cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i] - x[2*N+i]})
        boundary_cons.append({'type': 'ineq', 'fun': lambda x, i=i: (1.0 - x[2*N+i]) - x[2*i]})
        boundary_cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1] - x[2*N+i]})
        boundary_cons.append({'type': 'ineq', 'fun': lambda x, i=i: (1.0 - x[2*N+i]) - x[2*i+1]})

    # Overlap constraints
    overlap_cons = []
    for i in range(N):
        for j in range(i + 1, N):
            overlap_cons.append({'type': 'ineq', 'fun': make_constraint(i, j)})

    x0 = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.005, 0.995))
        bounds.append((0.005, 0.995))
    for i in range(N):
        bounds.append((0.005, 0.35))

    try:
        result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                          constraints=boundary_cons + overlap_cons,
                          options={'maxiter': 500, 'ftol': 1e-14})
        if time.time() < deadline:
            x_opt = result.x
            p_opt = x_opt[:2*N].reshape(N, 2)
            r_opt = x_opt[2*N:]
            p_opt, r_opt = _post_process(p_opt, r_opt, deadline)
            if _is_valid(p_opt, r_opt):
                s = np.sum(r_opt)
                if s > best_score:
                    best_score = s
                    best_pos = p_opt.copy()
                    best_radii = r_opt.copy()
    except Exception:
        pass

    return best_pos, best_radii, best_score
>>>>>>> REPLACE

