<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + 150.0 * DEADLINE_FRAC
    
    np.random.seed(42)
    
    # Multiple initialization strategies
    inits = [
        _grid_init(),
        _hex_init(),
        _random_init(),
        _random_init(),
    ]
    
    best_centers = None
    best_r = 0.0
    
    for init in inits:
        if time.time() > deadline:
            break
        
        # Stage 1: Force relaxation
        relaxed = _force_relax_vectorized(init, max_iter=3000, dt=0.0003, deadline=deadline)
        
        # Stage 2: Langevin dynamics
        lang_centers, lang_r = _langevin_dynamics(relaxed, deadline, max_iter=50000, T0=0.015, T_end=0.0005, dt=0.0008)
        
        if lang_r > best_r:
            best_r = lang_r
            best_centers = lang_centers.copy()
        
        # Stage 3: Gradient descent on soft-min
        if time.time() < deadline - 10.0:
            gd_centers, gd_r = _gradient_descent(lang_centers, deadline, max_iter=3000, lr=0.002, beta=40.0)
            if gd_r > best_r:
                best_r = gd_r
                best_centers = gd_centers.copy()
        
        if time.time() > deadline:
            break
    
    # Final local refinement
    if best_centers is not None:
        refined, final_r = _local_refine(best_centers, deadline)
        if final_r > best_r:
            best_r = final_r
            best_centers = refined
    
    # Ensure validity
    if best_centers is None:
        best_centers = _grid_init()
    
    best_centers = np.clip(best_centers, -0.4999, 0.4999)
    best_centers = np.nan_to_num(best_centers, nan=0.0, posinf=0.0, neginf=0.0)
    
    return best_centers
=======
def _force_relax(centers, max_iter=1000, dt=0.001):
    """Quick force relaxation to spread points."""
    centers = centers.copy()
    for _ in range(max_iter):
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, 1.0)
        inv_d3 = 1.0 / (dist ** 3 + 1e-10)
        np.fill_diagonal(inv_d3, 0.0)
        force = np.sum(diff * inv_d3[:, :, np.newaxis], axis=1)
        
        # Wall repulsion
        for k in range(2):
            force[:, k] -= 50.0 * np.maximum(centers[:, k] - 0.45, 0) ** 2
            force[:, k] += 50.0 * np.maximum(-0.45 - centers[:, k], 0) ** 2
            
        centers += dt * force
        centers[:, 0] = np.clip(centers[:, 0], -0.49, 0.49)
        centers[:, 1] = np.clip(centers[:, 1], -0.49, 0.49)
    return centers

def _slsqp_optimize(centers, deadline):
    """Optimize radius using SLSQP."""
    x0 = np.concatenate([centers.flatten(), [0.01]])
    n = len(centers)
    
    def objective(x):
        r = x[-1]
        return -r

    def constraint_pair(i, j):
        def c(x):
            ci = x[2*i:2*i+2]
            cj = x[2*j:2*j+2]
            r = x[-1]
            return np.sum((ci - cj)**2) - 4.0 * r**2
        return c

    def constraint_wall_x(i):
        def c1(x):
            return 0.5 - x[2*i] - x[-1]
        def c2(x):
            return 0.5 + x[2*i] - x[-1]
        return [c1, c2]

    def constraint_wall_y(i):
        def c1(x):
            return 0.5 - x[2*i+1] - x[-1]
        def c2(x):
            return 0.5 + x[2*i+1] - x[-1]
        return [c1, c2]

    constraints = []
    for i in range(n):
        for j in range(i+1, n):
            constraints.append({'type': 'ineq', 'fun': constraint_pair(i, j)})
            
    for i in range(n):
        constraints.extend([{'type': 'ineq', 'fun': c} for c in constraint_wall_x(i)])
        constraints.extend([{'type': 'ineq', 'fun': c} for c in constraint_wall_y(i)])
        
    bounds = [(-0.5, 0.5)] * (2*n) + [(0.0, 0.5)]

    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 500, 'ftol': 1e-9})
        if res.success or res.fun < objective(x0):
            best_x = res.x
        else:
            best_x = x0
    except Exception:
        best_x = x0

    best_centers = best_x[:2*n].reshape(n, 2)
    best_r = best_x[-1]
    
    best_centers = np.clip(best_centers, -0.5, 0.5)
    actual_r = _radius(best_centers)
    if actual_r > 0:
        return best_centers, actual_r
    else:
        return centers, _radius(centers)

def construct():
    start_time = time.time()
    deadline = start_time + 120.0 * 0.9
    
    np.random.seed(42)
    
    # Initial guesses
    inits = []
    # Grid
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    inits.append(grid[:N])
    
    # Random
    inits.append((np.random.rand(N, 2) - 0.5) * 0.9)
    
    # Hex-like
    pts = []
    for row in range(7):
        for col in range(8):
            x = (col + 0.5) / 8.0 - 0.5 + (0.5/8.0 if row % 2 else 0)
            y = (row + 0.5) / 7.0 - 0.5
            if abs(x) < 0.5 and abs(y) < 0.5:
                pts.append([x, y])
    pts = np.array(pts[:N])
    if len(pts) < N:
        extra = (np.random.rand(N - len(pts), 2) - 0.5) * 0.9
        pts = np.vstack([pts, extra])
    inits.append(pts)
    
    best_centers = inits[0]
    best_r = _radius(best_centers)
    
    for init in inits:
        if time.time() > deadline:
            break
        
        # Relax
        relaxed = _force_relax(init, max_iter=500)
        r_relax = _radius(relaxed)
        
        if r_relax > best_r:
            best_r = r_relax
            best_centers = relaxed.copy()
            
        # SLSQP Polish
        if time.time() < deadline - 10:
            polished, r_polish = _slsqp_optimize(relaxed, deadline)
            if r_polish > best_r:
                best_r = r_polish
                best_centers = polished.copy()
                
    return best_centers
>>>>>>> REPLACE
