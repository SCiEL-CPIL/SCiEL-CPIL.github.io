<<<<<<< SEARCH
    # Phase 2.5: Polish the best structure found so far
    if best_positions is not None and check_time():
        try_polish(best_positions)
        
=======
    # Phase 2.5: Polish the best structure found so far with geometric guidance
    if best_positions is not None and check_time():
        try_polish(best_positions)
        
    # Phase 2.6: Geometric re-optimization from best structure
    if best_positions is not None and check_time():
        try:
            pos_flat = best_positions.flatten().copy()
            n = len(pos_flat) // 3
            
            def geom_obj_and_grad(x):
                p = x.reshape(-1, 3)
                d = np.linalg.norm(p[:, np.newaxis, :] - p[np.newaxis, :, :], axis=2)
                np.fill_diagonal(d, np.inf)
                if np.min(d) < 0.49:
                    return 1e10, np.zeros_like(x)
                loss, grad = _geometric_loss(p, energy_weight=1.0, voronoi_weight=0.05)
                return loss, grad.flatten()
            
            res = minimize(geom_obj_and_grad, pos_flat, method='L-BFGS-B',
                          jac=True, options={'maxiter': 150, 'gtol': 1e-6, 'maxfun': 500})
            if res.fun < 1e9:
                polished = res.x.reshape(-1, 3)
                d = np.linalg.norm(polished[:, np.newaxis, :] - polished[np.newaxis, :, :], axis=2)
                np.fill_diagonal(d, np.inf)
                if np.min(d) >= 0.5 - 1e-6:
                    # Verify with official relax
                    energy, relaxed_pos = relax(polished, potential, param)
                    if energy < best_energy:
                        best_energy = energy
                        best_positions = relaxed_pos.copy()
        except Exception:
            pass
        
>>>>>>> REPLACE

