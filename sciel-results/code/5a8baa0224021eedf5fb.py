<<<<<<< SEARCH
    if best_global_center is None or best_global_S == float('inf'):
        best_global_pts = axial_to_cartesian(candidates[0])
        best_global_center = (float(np.mean(best_global_pts[:, 0])), float(np.mean(best_global_pts[:, 1])))
        best_global_beta = 0.0
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = best_global_pts[:, 0] * nx + best_global_pts[:, 1] * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        best_global_S = (max_proj + APOTHEM) / APOTHEM
=======
    if best_global_center is None or best_global_S == float('inf'):
        best_global_pts = axial_to_cartesian(candidates[0])
        best_global_center = (float(np.mean(best_global_pts[:, 0])), float(np.mean(best_global_pts[:, 1])))
        best_global_beta = 0.0
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = best_global_pts[:, 0] * nx + best_global_pts[:, 1] * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        best_global_S = (max_proj + APOTHEM) / APOTHEM
    
    # Final ultra-precise polish on the best solution
    if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN and best_global_center is not None:
        try:
            def final_objective(params):
                cx, cy, beta = params
                beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                normals = np.array([
                    [math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
                     math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0)]
                    for k in range(6)
                ])
                projs = (best_global_pts - np.array([cx, cy])) @ normals.T
                max_proj = np.max(np.abs(projs))
                S = (max_proj + APOTHEM) / APOTHEM
                return S
            
            # Very tight optimization from the current best
            init = [best_global_center[0], best_global_center[1], best_global_beta]
            res = minimize(final_objective, init, method='Nelder-Mead',
                         options={'maxiter': 10000, 'xatol': 1e-16, 'fatol': 1e-16})
            if res.fun < best_global_S:
                best_global_S = res.fun
                best_global_center = (res.x[0], res.x[1])
                best_global_beta = ((res.x[2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
        except Exception:
            pass
>>>>>>> REPLACE

