    seeds = []
    
    # Seed 1: Center + 6 + 4 (dist 2)
    a1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    seeds.append(axial_to_cartesian(a1))
    
    # Seed 2: Center + 6 + 4 (axes)
    a2 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    seeds.append(axial_to_cartesian(a2))
    
    # Seed 3: 3x3 block + 2
    a3 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1)
    ]
    seeds.append(axial_to_cartesian(a3))
    
    # Seed 4: Compact diamond
    a4 = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (-1, -1), (1, -1), (-1, 1),
        (2, 0), (-2, 0)
    ]
    seeds.append(axial_to_cartesian(a4))

    best_global_S = float('inf')
    best_global_params = None
    
    for seed_pts in seeds:
        seed_pts = seed_pts - np.mean(seed_pts, axis=0)
        p0 = np.zeros(23)
        p0[:22] = seed_pts.flatten()
        p0[22] = 0.0
        
        # Try a few angles
        for init_beta in [0.0, math.pi/12, math.pi/6]:
            p_test = p0.copy()
            p_test[22] = init_beta
            try:
                res = minimize(objective, p_test, method='Nelder-Mead', 
                               options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
                if res.fun < best_global_S:
                    best_global_S = res.fun
                    best_global_params = res.x
            except Exception:
                pass

    if best_global_params is None:
        best_global_params = p0
        best_global_S = objective(p0)
