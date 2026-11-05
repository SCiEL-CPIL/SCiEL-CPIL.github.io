<<<<<<< SEARCH
    # Final high-precision polish with L-BFGS-B (tighter tolerance, LSE objective)
    remaining_time = TIME_LIMIT - (time.time() - t0)
    if remaining_time > 5.0:
        try:
            v0 = flatten(best_cfg, best_center, best_angle)
            start_time = time.time()
            def callback2(xk):
                return time.time() - start_time > (remaining_time - 2.0)
            
            result = minimize(
                smooth_objective_lse,
                v0,
                method='L-BFGS-B',
                callback=callback2,
                options={'maxiter': 2000, 'maxfun': 20000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            
            v_opt = result.x
            cfg_opt = []
            for i in range(N):
                cfg_opt.append([v_opt[3*i], v_opt[3*i+1], v_opt[3*i+2]])
            center_opt = [v_opt[3*N], v_opt[3*N+1]]
            angle_opt = v_opt[3*N+2]
            
            verify_score = full_score(cfg_opt, tuple(center_opt), angle_opt)
            if verify_score is not None and verify_score < best_score:
                best_score = verify_score
                best_cfg = cfg_opt
                best_center = center_opt
                best_angle = angle_opt
        except Exception:
            pass
=======
    # Final high-precision polish with L-BFGS-B (tighter tolerance, LSE objective)
    remaining_time = TIME_LIMIT - (time.time() - t0)
    if remaining_time > 5.0:
        try:
            v0 = flatten(best_cfg, best_center, best_angle)
            start_time = time.time()
            def callback2(xk):
                return time.time() - start_time > (remaining_time - 2.0)
            
            result = minimize(
                smooth_objective_lse,
                v0,
                method='L-BFGS-B',
                callback=callback2,
                options={'maxiter': 2000, 'maxfun': 20000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            
            v_opt = result.x
            cfg_opt = []
            for i in range(N):
                cfg_opt.append([v_opt[3*i], v_opt[3*i+1], v_opt[3*i+2]])
            center_opt = [v_opt[3*N], v_opt[3*N+1]]
            angle_opt = v_opt[3*N+2]
            
            verify_score = full_score(cfg_opt, tuple(center_opt), angle_opt)
            if verify_score is not None and verify_score < best_score:
                best_score = verify_score
                best_cfg = cfg_opt
                best_center = center_opt
                best_angle = angle_opt
        except Exception:
            pass
    
    # Coordinate descent polish: fine-tune each variable individually
    remaining_time = TIME_LIMIT - (time.time() - t0)
    if remaining_time > 3.0:
        try:
            cd_start = time.time()
            for _round in range(3):
                if time.time() - cd_start > remaining_time - 1.0:
                    break
                # Try moving each hexagon x, y, angle in small steps
                for i in range(N):
                    for var in range(3):
                        for delta in [-0.01, -0.005, 0.005, 0.01]:
                            old_val = best_cfg[i][var]
                            best_cfg[i][var] = old_val + delta
                            s = full_score(best_cfg, tuple(best_center), best_angle)
                            if s is not None and s < best_score:
                                best_score = s
                            else:
                                best_cfg[i][var] = old_val
                # Try adjusting center and outer angle
                for var in range(2):
                    for delta in [-0.01, -0.005, 0.005, 0.01]:
                        old_val = best_center[var]
                        best_center[var] = old_val + delta
                        s = full_score(best_cfg, tuple(best_center), best_angle)
                        if s is not None and s < best_score:
                            best_score = s
                        else:
                            best_center[var] = old_val
                for delta in [-0.5, -0.2, 0.2, 0.5]:
                    old_val = best_angle
                    best_angle = old_val + delta
                    s = full_score(best_cfg, tuple(best_center), best_angle)
                    if s is not None and s < best_score:
                        best_score = s
                    else:
                        best_angle = old_val
        except Exception:
            pass
>>>>>>> REPLACE
