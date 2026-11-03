<<<<<<< SEARCH
def _lbfgs_polish(c, t0, time_budget=20.0):
    """
    L-BFGS-B polish using soft-min radius with adaptive decreasing temperature.
    """
    x = c.ravel()
    best_x = x.copy()
    best_r = _radius_fast(c)

    # Adaptive temperature schedule: start high, decrease geometrically
    temps = [200.0, 100.0, 50.0, 25.0, 12.0, 6.0, 3.0, 1.5, 0.8]
    maxiter_per_temp = 400

    for t in temps:
        elapsed = time.time() - t0
        if elapsed > t0 + time_budget:
            break

        # For smaller temps, use more iterations
        if t < 5.0:
            mi = maxiter_per_temp
        else:
            mi = maxiter_per_temp // 2

        res = minimize(
            lambda xx: _soft_min_radius_neg(xx, t),
            x,
            jac=lambda xx: _soft_min_radius_grad(xx, t),
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': mi, 'gtol': 1e-12, 'ftol': 1e-16}
        )
        c_new = res.x.reshape(N, 2)
        r_new = _radius_fast(c_new)
        if r_new > best_r:
            best_r = r_new
            best_x = res.x.copy()
            x = res.x.copy()

    return best_x.reshape(N, 2), best_r
=======
def _slsqp_epigraph_polish(c, t0, time_budget=20.0):
    """
    SLSQP epigraph polish: maximize r subject to constraints d_ij >= 2r and wall_dist >= r.
    This is highly effective for final refinement.
    """
    def neg_r(x):
        return -x[2*N]

    def constraints(x):
        c_i = x[:2*N].reshape(N, 2)
        r = x[2*N]
        cons = []
        
        # Pairwise constraints: ||c_i - c_j|| - 2r >= 0
        d = cdist(c_i, c_i)
        np.fill_diagonal(d, 1.0) # Ignore self
        for i in range(N):
            for j in range(i+1, N):
                cons.append({'type': 'ineq', 'fun': lambda x, i=i, j=j: x[:2*N].reshape(N,2)[i] and 0}) # Placeholder, will vectorize below
        # Vectorized constraint construction is faster but SLSQP needs individual functions or a vector function.
        # We'll construct a vector of all constraints.
        
        # Actually, for N=50, number of pairs is 1225. Walls 100. Total ~1325 constraints.
        # SLSQP might be slow with this many. We can reduce by only active constraints?
        # But we don't know active constraints a priori.
        # However, SLSQP handles large N constraints reasonably if well-conditioned.
        
        # Let's use a vectorized approach if possible or just list them.
        # For speed, we'll just return the list of constraints.
        
        # Wall constraints: 0.5 - |c_ik| - r >= 0
        for i in range(N):
            for k in range(2):
                cons.append({'type': 'ineq', 'fun': lambda x, i=i, k=k: 0.5 - abs(x[i*2+k]) - x[2*N]})
                
        # Pair constraints
        for i in range(N):
            for j in range(i+1, N):
                cons.append({'type': 'ineq', 'fun': lambda x, i=i, j=j: np.linalg.norm(x[i*2:i*2+2] - x[j*2:j*2+2]) - 2*x[2*N]})
        
        return cons

    # Initial guess for r
    r0 = _radius_fast(c)
    x0 = np.concatenate([c.ravel(), [r0]])
    
    # Bounds
    bounds = [(-0.5, 0.5)] * (2*N) + [(0, 0.5)]
    
    best_x = x0.copy()
    best_r = r0
    
    # Run SLSQP
    # maxiter needs to be sufficient. 100-200 is usually enough for polish.
    res = minimize(
        neg_r,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 300, 'ftol': 1e-12}
    )
    
    if res.success:
        x_new = res.x
        c_new = x_new[:2*N].reshape(N, 2)
        r_new = x_new[2*N]
        
        # Verify with fast radius calculation
        r_true = _radius_fast(c_new)
        if r_true > best_r:
            best_r = r_true
            best_x = x_new.copy()
            
    return best_x[:2*N].reshape(N, 2), best_r
>>>>>>> REPLACE

