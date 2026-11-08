<<<<<<< SEARCH
    def local_optimization(circles, max_iter=500, lr=0.01):
        """
        Simple gradient-free local optimization using random perturbations.
        """
        n = len(circles)
        best_circles = circles.copy()
        best_score = compute_score(circles)
        current_circles = circles.copy()
        
        for iteration in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Random perturbation
            candidate = current_circles.copy()
            noise_scale = 0.01
            candidate[:, :2] += np.random.randn(n, 2) * noise_scale
            candidate[:, 2] *= (1 + np.random.randn(n) * 0.005)
            candidate[:, 2] = np.clip(candidate[:, 2], 1e-6, 0.5)
            
            # Project to constraints
            candidate = project_constraints(candidate, max_iter=30)
            
            # Check validity
            if check_validity(candidate, tol=1e-8):
                score = compute_score(candidate)
                if score > best_score:
                    best_score = score
                    best_circles = candidate.copy()
                    current_circles = candidate.copy()
        
        return best_circles
=======
    def local_optimization(circles, max_iter=500, lr=0.01):
        """
        Local optimization using structured perturbations with adaptive step size.
        Tries position-only moves, radius-only moves, and combined moves.
        """
        n = len(circles)
        best_circles = circles.copy()
        best_score = compute_score(circles)
        current_circles = circles.copy()
        step_size = 0.02
        no_improve = 0
        
        for iteration in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Choose perturbation type
            pert_type = iteration % 3
            candidate = current_circles.copy()
            
            if pert_type == 0:
                # Position-only perturbation
                candidate[:, :2] += np.random.randn(n, 2) * step_size
            elif pert_type == 1:
                # Radius-only perturbation (small)
                candidate[:, 2] *= (1 + np.random.randn(n) * 0.01)
                candidate[:, 2] = np.clip(candidate[:, 2], 1e-6, 0.5)
            else:
                # Combined small perturbation
                candidate[:, :2] += np.random.randn(n, 2) * step_size * 0.5
                candidate[:, 2] *= (1 + np.random.randn(n) * 0.005)
                candidate[:, 2] = np.clip(candidate[:, 2], 1e-6, 0.5)
            
            # Project to constraints
            candidate = project_constraints(candidate, max_iter=30)
            
            # Check validity
            if check_validity(candidate, tol=1e-8):
                score = compute_score(candidate)
                if score > best_score:
                    best_score = score
                    best_circles = candidate.copy()
                    current_circles = candidate.copy()
                    no_improve = 0
                    # Slightly increase step size on success
                    step_size = min(step_size * 1.05, 0.05)
                else:
                    no_improve += 1
            else:
                no_improve += 1
            
            # Decrease step size when stuck
            if no_improve >= 10:
                step_size *= 0.8
                no_improve = 0
        
        return best_circles
>>>>>>> REPLACE

