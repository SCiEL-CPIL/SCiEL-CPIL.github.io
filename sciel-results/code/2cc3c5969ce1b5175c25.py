<<<<<<< SEARCH
        # Evaluate exact covering radius
        score = _covering_radius_from_hull(points)
        
        if score < best_score:
            best_score = score
            best_points = points.copy()
        
        # Early exit if we're close to the known best
        if best_score < 0.1975:  # ~11.3 degrees
            break
=======
        # Step 3: Final polish with tighter tolerance
        if time.time() - start_time < total_time_limit - 5.0:
            def polish_objective_and_grad(x_flat):
                pts = x_flat.reshape(N, 3)
                norms = np.linalg.norm(pts, axis=1, keepdims=True)
                norms = np.where(norms < 1e-12, 1.0, norms)
                pts = pts / norms
                
                val, grad = _soft_covering_radius_and_grad(pts, grid_points, sigma=0.01)
                
                return val, grad.flatten()
            
            x0 = points.flatten()
            
            try:
                res = minimize(
                    polish_objective_and_grad,
                    x0,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-7, 'disp': False}
                )
                if res.x is not None:
                    points = res.x.reshape(N, 3)
                    norms = np.linalg.norm(points, axis=1, keepdims=True)
                    norms = np.where(norms < 1e-12, 1.0, norms)
                    points = points / norms
            except Exception:
                pass
        
        # Evaluate exact covering radius
        score = _covering_radius_from_hull(points)
        
        if score < best_score:
            best_score = score
            best_points = points.copy()
        
        # Early exit if we're close to the known best
        if best_score < 0.1975:  # ~11.3 degrees
            break
>>>>>>> REPLACE
