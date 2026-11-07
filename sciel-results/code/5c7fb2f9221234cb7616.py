"""Ridge-aware optimizer using Nelder-Mead with restarts and adaptive coordinate descent."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead optimizer with strict budget control."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        def clip(v):
            return np.clip(np.asarray(v, dtype=float), -5.0, 5.0)

        def time_ok():
            return time.time() < deadline

        best_x = None
        best_f = np.inf
        calls_used = 0

        # Nelder-Mead parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        max_iter = 200  # Iterations per NM run, not calls

        def run_nelder_mead(center, initial_step):
            nonlocal best_x, best_f, calls_used
            
            # Initialize simplex
            simplex = []
            simplex_f = []
            
            # First point is center
            x0 = clip(center)
            f0 = func(x0)
            calls_used += 1
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()
            simplex.append(x0)
            simplex_f.append(f0)
            
            if calls_used >= budget - 5:
                return

            # Other points: center + step * e_i
            for i in range(dim):
                if calls_used >= budget - 5:
                    return
                x_i = center.copy()
                x_i[i] += initial_step
                x_i = clip(x_i)
                f_i = func(x_i)
                calls_used += 1
                if f_i < best_f:
                    best_f = f_i
                    best_x = x_i.copy()
                simplex.append(x_i)
                simplex_f.append(f_i)

            # Main NM loop
            for _ in range(max_iter):
                if not time_ok() or calls_used >= budget - 5:
                    break

                # Sort simplex by fitness
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                # Check for convergence
                if max(simplex_f) - min(simplex_f) < 1e-12:
                    break

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                worst_f = simplex_f[-1]

                # Reflection
                x_r = centroid + alpha * (centroid - worst)
                x_r = clip(x_r)
                f_r = func(x_r)
                calls_used += 1
                if f_r < best_f:
                    best_f = f_r
                    best_x = x_r.copy()

                if calls_used >= budget - 5:
                    break

                if f_r < simplex_f[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = clip(x_e)
                    f_e = func(x_e)
                    calls_used += 1
                    if f_e < best_f:
                        best_f = f_e
                        best_x = x_e.copy()
                    
                    if calls_used >= budget - 5:
                        break

                    if f_e < f_r:
                        simplex[-1] = x_e
                        simplex_f[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                elif f_r < simplex_f[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
                else:
                    # Contraction
                    if f_r < worst_f:
                        # Outer contraction
                        x_c = centroid + rho * (x_r - centroid)
                        x_c = clip(x_c)
                        f_c = func(x_c)
                        calls_used += 1
                        if f_c < best_f:
                            best_f = f_c
                            best_x = x_c.copy()
                        
                        if calls_used >= budget - 5:
                            break

                        if f_c < f_r:
                            simplex[-1] = x_c
                            simplex_f[-1] = f_c
                        else:
                            # Shrink
                            for i in range(1, len(simplex)):
                                if calls_used >= budget - 5:
                                    return
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = clip(simplex[i])
                                f_s = func(simplex[i])
                                calls_used += 1
                                if f_s < best_f:
                                    best_f = f_s
                                    best_x = simplex[i].copy()
                                simplex_f[i] = f_s
                    else:
                        # Inner contraction
                        x_c = centroid - rho * (centroid - worst)
                        x_c = clip(x_c)
                        f_c = func(x_c)
                        calls_used += 1
                        if f_c < best_f:
                            best_f = f_c
                            best_x = x_c.copy()
                        
                        if calls_used >= budget - 5:
                            break

                        if f_c < worst_f:
                            simplex[-1] = x_c
                            simplex_f[-1] = f_c
                        else:
                            # Shrink
                            for i in range(1, len(simplex)):
                                if calls_used >= budget - 5:
                                    return
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = clip(simplex[i])
                                f_s = func(simplex[i])
                                calls_used += 1
                                if f_s < best_f:
                                    best_f = f_s
                                    best_x = simplex[i].copy()
                                simplex_f[i] = f_s

        def adaptive_coordinate_descent(center, max_iters=30):
            """Adaptive coordinate descent with dynamic rotation based on recent evaluations."""
            nonlocal best_x, best_f, calls_used
            
            x = clip(center).copy()
            f_curr = func(x)
            calls_used += 1
            if f_curr < best_f:
                best_f = f_curr
                best_x = x.copy()
            
            # Track recent evaluations for PCA
            recent_points = [x.copy()]
            recent_values = [f_curr]
            
            for it in range(max_iters):
                if not time_ok() or calls_used >= budget - 5:
                    break
                
                # Compute rotation matrix from recent points
                if len(recent_points) >= 3:
                    # Center the points
                    pts = np.array(recent_points)
                    pts_centered = pts - np.mean(pts, axis=0)
                    # Compute covariance
                    cov = np.cov(pts_centered.T)
                    # Get principal directions
                    try:
                        eigenvalues, eigenvectors = np.linalg.eigh(cov)
                        # Sort by eigenvalue (descending)
                        idx = np.argsort(eigenvalues)[::-1]
                        eigenvectors = eigenvectors[:, idx]
                        # Use top 2 directions
                        R = eigenvectors[:, :2]
                        # Extend to full dimensionality
                        if dim > 2:
                            # Complete the basis
                            remaining = np.eye(dim)[:, 2:]
                            # Orthogonalize remaining against R
                            for j in range(2, dim):
                                for i in range(2):
                                    remaining[:, j-2] -= np.dot(remaining[:, j-2], R[:, i]) * R[:, i]
                                remaining[:, j-2] /= np.linalg.norm(remaining[:, j-2])
                            R = np.hstack([R, remaining])
                    except:
                        R = np.eye(dim)
                else:
                    R = np.eye(dim)
                
                # Perform coordinate descent in rotated space
                improved = False
                for d in range(dim):
                    if not time_ok() or calls_used >= budget - 5:
                        break
                    
                    # Try positive and negative steps along direction d
                    direction = R[:, d]
                    for step_sign in [1.0, -1.0]:
                        # Adaptive step size
                        step_size = 0.5 * (1.0 / (1.0 + it * 0.1))
                        x_trial = clip(x + step_sign * step_size * direction)
                        f_trial = func(x_trial)
                        calls_used += 1
                        
                        # Track for PCA
                        recent_points.append(x_trial.copy())
                        recent_values.append(f_trial)
                        if len(recent_points) > 10:
                            recent_points.pop(0)
                            recent_values.pop(0)
                        
                        if f_trial < f_curr:
                            x = x_trial
                            f_curr = f_trial
                            improved = True
                            if f_curr < best_f:
                                best_f = f_curr
                                best_x = x.copy()
                        else:
                            # Try smaller step
                            x_trial2 = clip(x + step_sign * step_size * 0.5 * direction)
                            f_trial2 = func(x_trial2)
                            calls_used += 1
                            
                            recent_points.append(x_trial2.copy())
                            recent_values.append(f_trial2)
                            if len(recent_points) > 10:
                                recent_points.pop(0)
                                recent_values.pop(0)
                            
                            if f_trial2 < f_curr:
                                x = x_trial2
                                f_curr = f_trial2
                                improved = True
                                if f_curr < best_f:
                                    best_f = f_curr
                                    best_x = x.copy()
                
                if not improved:
                    break

        # Phase 1: Coarse search with a few random starts
        # We have 500 calls. Let's use 3 restarts.
        restarts = 3

        for r in range(restarts):
            if not time_ok() or calls_used >= budget - 10:
                break
            
            if r == 0:
                center = np.zeros(dim)
                step = 2.0
            else:
                # Perturb the current best or random
                if best_x is not None:
                    center = best_x + rng.normal(0, 1.0, dim)
                else:
                    center = rng.uniform(-4, 4, dim)
                step = 1.0

            run_nelder_mead(center, step)
            
            # After NM, do adaptive coordinate descent to polish
            if best_x is not None and calls_used < budget - 20 and time_ok():
                adaptive_coordinate_descent(best_x, max_iters=20)

        if best_x is None:
            # Fallback
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_used += 1

        # Final polish with adaptive coordinate descent if we have budget left
        if calls_used < budget - 10 and time_ok():
            adaptive_coordinate_descent(best_x, max_iters=15)

        return best_x, best_f
# EVOLVE-BLOCK-END
