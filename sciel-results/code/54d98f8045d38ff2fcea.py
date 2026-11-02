<<<<<<< SEARCH
    def optimize_shape(pts):
        best_S = float('inf')
        best_beta = 0.0
        best_center = None
        
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None
        
        # Coarse grid search over beta - finer grid
        for i in range(73):
            beta = i * (math.pi / 6) / 72
            S, center, D = calc_S_for_beta(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center
        
        # Refine beta with bounded scalar optimization
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                res = minimize_scalar(lambda b: calc_S_for_beta(pts, b)[0], bounds=(0, math.pi / 6), method='bounded',
                                     options={'maxiter': 300, 'xatol': 1e-12})
                if res.fun < best_S:
                    best_S = res.fun
                    best_beta = res.x
                    _, best_center, _ = calc_S_for_beta(pts, best_beta)
        except Exception:
            pass
        
        # Smooth surrogate optimization with adaptive temperature
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                temperatures = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
                start_betas = [best_beta, best_beta + 0.05, best_beta - 0.05, 0.0, math.pi/12, math.pi/6, best_beta + 0.1, best_beta - 0.1]
                
                for temp in temperatures:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    for sb in start_betas:
                        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                            break
                        
                        def smooth_obj(params, t=temp):
                            cx, cy, beta = params
                            beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                            S, _, _ = smooth_calc_S(pts, beta, cx, cy, temperature=t)
                            return S
                        
                        init = [best_center[0], best_center[1], sb]
                        res = minimize(smooth_obj, init, method='Nelder-Mead',
                                     options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10})
                        
                        if res.fun < 1e10:
                            cx_r, cy_r, beta_r = res.x
                            beta_r = ((beta_r % (math.pi/6)) + math.pi/6) % (math.pi/6)
                            S_exact, c_exact, _ = calc_S_for_beta(pts, beta_r)
                            if S_exact < best_S:
                                best_S = S_exact
                                best_beta = beta_r
                                best_center = c_exact
        except Exception:
            pass
        
        # Final high-precision polish using exact objective
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                def objective(params):
                    cx, cy, beta = params
                    beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                    normals = []
                    for k in range(6):
                        ang = beta + math.pi / 6.0 + k * math.pi / 3.0
                        normals.append((math.cos(ang), math.sin(ang)))
                    
                    max_proj = 0.0
                    for nx, ny in normals:
                        for px, py in pts:
                            proj = abs((px - cx) * nx + (py - cy) * ny)
                            if proj > max_proj:
                                max_proj = proj
                    
                    S = (max_proj + APOTHEM) / APOTHEM
                    return S
                
                # Generate more diverse starting points
                polish_starts = [
                    [best_center[0], best_center[1], best_beta],
                    [best_center[0], best_center[1], best_beta + 0.02],
                    [best_center[0], best_center[1], best_beta - 0.02],
                    [best_center[0] + 0.02, best_center[1], best_beta],
                    [best_center[0], best_center[1] + 0.02, best_beta],
                    [best_center[0] - 0.02, best_center[1], best_beta],
                    [best_center[0], best_center[1] - 0.02, best_beta],
                    [best_center[0] + 0.05, best_center[1] + 0.05, best_beta],
                    [best_center[0] - 0.05, best_center[1] - 0.05, best_beta],
                    [best_center[0] + 0.05, best_center[1] - 0.05, best_beta],
                    [best_center[0] - 0.05, best_center[1] + 0.05, best_beta],
                    [best_center[0], best_center[1], best_beta + 0.05],
                    [best_center[0], best_center[1], best_beta - 0.05],
                ]
                
                for init in polish_starts:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    res = minimize(objective, init, method='Nelder-Mead',
                                 options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-14})
                    if res.fun < best_S:
                        best_S = res.fun
                        best_center = (res.x[0], res.x[1])
                        best_beta = ((res.x[2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
        except Exception:
            pass
        
        # Second pass: re-optimize with even tighter tolerance if time permits
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN - 5:
                def objective2(params):
                    cx, cy, beta = params
                    beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                    normals = []
                    for k in range(6):
                        ang = beta + math.pi / 6.0 + k * math.pi / 3.0
                        normals.append((math.cos(ang), math.sin(ang)))
                    
                    max_proj = 0.0
                    for nx, ny in normals:
                        for px, py in pts:
                            proj = abs((px - cx) * nx + (py - cy) * ny)
                            if proj > max_proj:
                                max_proj = proj
                    
                    S = (max_proj + APOTHEM) / APOTHEM
                    return S
                
                final_starts = [
                    [best_center[0], best_center[1], best_beta],
                    [best_center[0] + 0.01, best_center[1], best_beta],
                    [best_center[0], best_center[1] + 0.01, best_beta],
                    [best_center[0], best_center[1], best_beta + 0.01],
                ]
                
                for init in final_starts:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN - 2:
                        break
                    res = minimize(objective2, init, method='Nelder-Mead',
                                 options={'maxiter': 20000, 'xatol': 1e-15, 'fatol': 1e-15})
                    if res.fun < best_S:
                        best_S = res.fun
                        best_center = (res.x[0], res.x[1])
                        best_beta = ((res.x[2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
        except Exception:
            pass
            
        return best_S, best_beta, best_center
=======
    def optimize_shape(pts):
        best_S = float('inf')
        best_beta = 0.0
        best_center = None
        
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None
        
        # Coarse grid search over beta
        for i in range(73):
            beta = i * (math.pi / 6) / 72
            S, center, D = calc_S_for_beta(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center
        
        # Refine beta with bounded scalar optimization
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                res = minimize_scalar(lambda b: calc_S_for_beta(pts, b)[0], bounds=(0, math.pi / 6), method='bounded',
                                     options={'maxiter': 300, 'xatol': 1e-12})
                if res.fun < best_S:
                    best_S = res.fun
                    best_beta = res.x
                    _, best_center, _ = calc_S_for_beta(pts, best_beta)
        except Exception:
            pass
        
        # Smooth surrogate optimization with adaptive temperature annealing
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                temperatures = [0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002]
                start_betas = [best_beta, best_beta + 0.05, best_beta - 0.05, 0.0, math.pi/12, math.pi/6, 
                               best_beta + 0.1, best_beta - 0.1, best_beta + 0.15, best_beta - 0.15]
                
                for temp in temperatures:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    for sb in start_betas:
                        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                            break
                        
                        def smooth_obj(params, t=temp):
                            cx, cy, beta = params
                            beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                            S, _, _ = smooth_calc_S(pts, beta, cx, cy, temperature=t)
                            return S
                        
                        init = [best_center[0], best_center[1], sb]
                        res = minimize(smooth_obj, init, method='Nelder-Mead',
                                     options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
                        
                        if res.fun < 1e10:
                            cx_r, cy_r, beta_r = res.x
                            beta_r = ((beta_r % (math.pi/6)) + math.pi/6) % (math.pi/6)
                            S_exact, c_exact, _ = calc_S_for_beta(pts, beta_r)
                            if S_exact < best_S:
                                best_S = S_exact
                                best_beta = beta_r
                                best_center = c_exact
        except Exception:
            pass
        
        # Final high-precision polish using vectorized exact objective
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                def objective(params):
                    cx, cy, beta = params
                    beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                    normals = np.array([
                        [math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
                         math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0)]
                        for k in range(6)
                    ])
                    
                    centered = pts - np.array([cx, cy])
                    projs = np.abs(centered @ normals.T)
                    max_proj = np.max(projs)
                    
                    S = (max_proj + APOTHEM) / APOTHEM
                    return S
                
                # Generate diverse starting points
                polish_starts = [
                    [best_center[0], best_center[1], best_beta],
                    [best_center[0], best_center[1], best_beta + 0.02],
                    [best_center[0], best_center[1], best_beta - 0.02],
                    [best_center[0] + 0.02, best_center[1], best_beta],
                    [best_center[0], best_center[1] + 0.02, best_beta],
                    [best_center[0] - 0.02, best_center[1], best_beta],
                    [best_center[0], best_center[1] - 0.02, best_beta],
                    [best_center[0] + 0.05, best_center[1] + 0.05, best_beta],
                    [best_center[0] - 0.05, best_center[1] - 0.05, best_beta],
                    [best_center[0] + 0.05, best_center[1] - 0.05, best_beta],
                    [best_center[0] - 0.05, best_center[1] + 0.05, best_beta],
                    [best_center[0], best_center[1], best_beta + 0.05],
                    [best_center[0], best_center[1], best_beta - 0.05],
                    [best_center[0] + 0.1, best_center[1], best_beta],
                    [best_center[0] - 0.1, best_center[1], best_beta],
                    [best_center[0], best_center[1] + 0.1, best_beta],
                    [best_center[0], best_center[1] - 0.1, best_beta],
                ]
                
                for init in polish_starts:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    res = minimize(objective, init, method='Nelder-Mead',
                                 options={'maxiter': 15000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True})
                    if res.fun < best_S:
                        best_S = res.fun
                        best_center = (res.x[0], res.x[1])
                        best_beta = ((res.x[2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
        except Exception:
            pass
        
        # Second pass: re-optimize with even tighter tolerance if time permits
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN - 5:
                def objective2(params):
                    cx, cy, beta = params
                    beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                    normals = np.array([
                        [math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
                         math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0)]
                        for k in range(6)
                    ])
                    
                    centered = pts - np.array([cx, cy])
                    projs = np.abs(centered @ normals.T)
                    max_proj = np.max(projs)
                    
                    S = (max_proj + APOTHEM) / APOTHEM
                    return S
                
                final_starts = [
                    [best_center[0], best_center[1], best_beta],
                    [best_center[0] + 0.01, best_center[1], best_beta],
                    [best_center[0], best_center[1] + 0.01, best_beta],
                    [best_center[0], best_center[1], best_beta + 0.01],
                    [best_center[0] - 0.01, best_center[1], best_beta],
                    [best_center[0], best_center[1] - 0.01, best_beta],
                    [best_center[0], best_center[1], best_beta - 0.01],
                ]
                
                for init in final_starts:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN - 2:
                        break
                    res = minimize(objective2, init, method='Nelder-Mead',
                                 options={'maxiter': 30000, 'xatol': 1e-15, 'fatol': 1e-15, 'adaptive': True})
                    if res.fun < best_S:
                        best_S = res.fun
                        best_center = (res.x[0], res.x[1])
                        best_beta = ((res.x[2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
        except Exception:
            pass
            
        return best_S, best_beta, best_center
>>>>>>> REPLACE

