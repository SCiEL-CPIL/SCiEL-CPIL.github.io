<<<<<<< SEARCH
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                def objective(params):
                    cx, cy, beta = params
                    beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                    normals = np.array([
                        (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
                         math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0))
                        for k in range(6)
                    ])
                    pts_centered = pts - np.array([cx, cy])
                    projs = np.abs(pts_centered @ normals.T)
                    max_proj = np.max(projs)
                    S = (max_proj + APOTHEM) / APOTHEM
                    return S

                polish_starts = []
                bcx, bcy = best_center
                bb = best_beta
                for dx in [-0.05, -0.02, 0.0, 0.02, 0.05]:
                    for dy in [-0.05, -0.02, 0.0, 0.02, 0.05]:
                        for db in [-0.03, -0.01, 0.0, 0.01, 0.03]:
                            polish_starts.append([bcx + dx, bcy + dy, bb + db])

                for init in polish_starts:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    res = minimize(objective, init, method='Nelder-Mead',
                                   options={'maxiter': 8000, 'xatol': 1e-14, 'fatol': 1e-14,
                                            'adaptive': True})
                    if res.fun < best_S:
                        best_S = res.fun
                        best_center = (res.x[0], res.x[1])
                        best_beta = ((res.x[2] % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
        except Exception:
            pass

        return best_S, best_beta, best_center
=======
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                def objective(params):
                    cx, cy, beta = params
                    beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                    normals = np.array([
                        (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
                         math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0))
                        for k in range(6)
                    ])
                    pts_centered = pts - np.array([cx, cy])
                    projs = np.abs(pts_centered @ normals.T)
                    max_proj = np.max(projs)
                    S = (max_proj + APOTHEM) / APOTHEM
                    return S

                polish_starts = []
                bcx, bcy = best_center
                bb = best_beta
                for dx in [-0.05, -0.02, 0.0, 0.02, 0.05]:
                    for dy in [-0.05, -0.02, 0.0, 0.02, 0.05]:
                        for db in [-0.03, -0.01, 0.0, 0.01, 0.03]:
                            polish_starts.append([bcx + dx, bcy + dy, bb + db])

                for init in polish_starts:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    res = minimize(objective, init, method='Nelder-Mead',
                                   options={'maxiter': 8000, 'xatol': 1e-14, 'fatol': 1e-14,
                                            'adaptive': True})
                    if res.fun < best_S:
                        best_S = res.fun
                        best_center = (res.x[0], res.x[1])
                        best_beta = ((res.x[2] % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
        except Exception:
            pass

        # Final high-precision polish with very tight tolerances
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN and best_center is not None:
                def precise_objective(params):
                    cx, cy, beta = params
                    beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                    normals = np.array([
                        (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
                         math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0))
                        for k in range(6)
                    ])
                    pts_centered = pts - np.array([cx, cy])
                    projs = np.abs(pts_centered @ normals.T)
                    max_proj = np.max(projs)
                    S = (max_proj + APOTHEM) / APOTHEM
                    return S

                bcx, bcy = best_center
                bb = best_beta
                fine_starts = []
                for dx in [-0.01, -0.005, 0.0, 0.005, 0.01]:
                    for dy in [-0.01, -0.005, 0.0, 0.005, 0.01]:
                        for db in [-0.01, -0.005, 0.0, 0.005, 0.01]:
                            fine_starts.append([bcx + dx, bcy + dy, bb + db])

                for init in fine_starts:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    res = minimize(precise_objective, init, method='Nelder-Mead',
                                   options={'maxiter': 20000, 'xatol': 1e-16, 'fatol': 1e-16,
                                            'adaptive': True})
                    if res.fun < best_S:
                        best_S = res.fun
                        best_center = (res.x[0], res.x[1])
                        best_beta = ((res.x[2] % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
        except Exception:
            pass

        return best_S, best_beta, best_center
>>>>>>> REPLACE

