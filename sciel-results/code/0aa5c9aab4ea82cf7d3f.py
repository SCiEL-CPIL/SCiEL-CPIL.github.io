"""Valley-Tracking Coordinate Descent with Adaptive Nelder-Mead for Rosenbrock."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Optimizer for Rosenbrock function (f8) using Valley-Tracking CD and Adaptive NM.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        calls = 0

        def evaluate(x):
            nonlocal calls
            x = np.clip(x, -5, 5)
            f = func(x)
            calls += 1
            return f

        # --- Phase 0: Intelligent Initialization ---
        # Rosenbrock valley: x_{i+1} approx x_i^2
        # Try points along the valley and near the origin.
        
        # Point 1: Origin
        x0 = np.zeros(dim)
        f0 = evaluate(x0)
        
        # Point 2: Along valley starting from 1
        x1 = np.ones(dim)
        for i in range(dim - 1):
            x1[i+1] = x1[i]**2
        x1 = np.clip(x1, -5, 5)
        f1 = evaluate(x1)

        # Point 3: Along valley starting from -1 (might be outside box, clip handles it)
        x2 = np.ones(dim)
        x2[0] = -1.0
        for i in range(dim - 1):
            x2[i+1] = x2[i]**2
        x2 = np.clip(x2, -5, 5)
        f2 = evaluate(x2)

        # Point 4: Random point
        x3 = rng.uniform(-2, 2, dim)
        f3 = evaluate(x3)

        # Select best
        best_x = x0
        best_f = f0
        for x, f in [(x1, f1), (x2, f2), (x3, f3)]:
            if f < best_f:
                best_x = x
                best_f = f

        x = best_x.copy()
        fx = best_f

        # --- Phase 1: Valley-Tracking Coordinate Descent ---
        # Strategy: Iterate through coordinates. For each coordinate i, 
        # try to move it. Crucially, after moving x_i, project x_{i+1} to x_i^2 
        # and accept if it improves. This exploits the specific geometry of Rosenbrock.
        
        max_cd_iters = 20
        step_size = 1.0
        
        for it in range(max_cd_iters):
            if calls >= budget - 10:
                break
                
            improved = False
            for i in range(dim):
                if calls >= budget - 10:
                    break
                    
                # 1. Standard coordinate search for x_i
                # Try a few steps
                best_step = 0
                best_f_i = fx
                base_x = x.copy()
                
                # Adaptive step sizes: shrink if we are close to convergence
                current_step = step_size * (1.0 / (1.0 + it))
                
                for sign in [1.0, -1.0]:
                    for mult in [1.0, 0.5, 2.0]:
                        s = sign * current_step * mult
                        if abs(s) < 1e-8:
                            continue
                        x_trial = base_x.copy()
                        x_trial[i] += s
                        x_trial = np.clip(x_trial, -5, 5)
                        
                        if np.allclose(x_trial, base_x, atol=1e-10):
                            continue
                        
                        f_trial = evaluate(x_trial)
                        
                        if f_trial < best_f_i:
                            best_f_i = f_trial
                            best_step = s
                            best_x_i = x_trial.copy()
                
                if best_step != 0:
                    x = best_x_i.copy()
                    fx = best_f_i
                    improved = True
                    
                    # 2. Valley Projection: Update x_{i+1} to x_i^2
                    if i < dim - 1:
                        x_proj = x.copy()
                        x_proj[i+1] = x[i]**2
                        x_proj = np.clip(x_proj, -5, 5)
                        
                        if not np.allclose(x_proj, x, atol=1e-10):
                            f_proj = evaluate(x_proj)
                            if f_proj < fx:
                                x = x_proj
                                fx = f_proj
                                improved = True
                                # Also project the next one? x_{i+2} to x_{i+1}^2
                                if i+1 < dim - 1:
                                    x_proj2 = x.copy()
                                    x_proj2[i+2] = x[i+1]**2
                                    x_proj2 = np.clip(x_proj2, -5, 5)
                                    if not np.allclose(x_proj2, x, atol=1e-10):
                                        f_proj2 = evaluate(x_proj2)
                                        if f_proj2 < fx:
                                            x = x_proj2
                                            fx = f_proj2
                                            improved = True

            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            if not improved:
                step_size *= 0.5
                if step_size < 1e-4:
                    break

        # --- Phase 2: Adaptive Nelder-Mead ---
        # Initialize simplex around best_x
        if calls < budget - (dim + 2):
            simplex = [best_x.copy()]
            f_simplex = [best_f]
            
            # Base step for simplex vertices
            base_step = 0.5
            
            for i in range(dim):
                if calls >= budget - 2:
                    break
                v = best_x.copy()
                v[i] += base_step
                v = np.clip(v, -5, 5)
                fv = evaluate(v)
                simplex.append(v)
                f_simplex.append(fv)

            # NM Parameters
            alpha_nm = 1.0
            gamma_nm = 2.0
            rho_nm = 0.5
            sigma_nm = 0.5
            
            nm_iter = 0
            max_nm_iter = min(budget - calls, 50)
            
            while nm_iter < max_nm_iter and calls < budget - 2:
                nm_iter += 1
                
                # Sort
                order = np.argsort(f_simplex)
                simplex = [simplex[i] for i in order]
                f_simplex = [f_simplex[i] for i in order]
                
                # Check convergence
                if f_simplex[-1] - f_simplex[0] < 1e-8:
                    break
                    
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_ref = centroid + alpha_nm * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = evaluate(x_ref)
                
                if f_ref < f_simplex[0]:
                    # Expansion
                    x_exp = centroid + gamma_nm * (centroid - simplex[-1])
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = evaluate(x_exp)
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        f_simplex[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        f_simplex[-1] = f_ref
                elif f_ref < f_simplex[-2]:
                    simplex[-1] = x_ref
                    f_simplex[-1] = f_ref
                else:
                    # Contraction
                    x_con = centroid + rho_nm * (simplex[-1] - centroid)
                    x_con = np.clip(x_con, -5, 5)
                    f_con = evaluate(x_con)
                    if f_con < f_ref:
                        simplex[-1] = x_con
                        f_simplex[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            if calls >= budget:
                                break
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_simplex[i] = evaluate(simplex[i])
                            
                # Update best
                min_idx = np.argmin(f_simplex)
                if f_simplex[min_idx] < best_f:
                    best_f = f_simplex[min_idx]
                    best_x = simplex[min_idx].copy()

        # --- Phase 3: Final Polish ---
        # Very small coordinate steps
        if calls < budget:
            steps = np.ones(dim) * 0.01
            for _ in range(10):
                if calls >= budget:
                    break
                for i in range(dim):
                    if calls >= budget:
                        break
                    for sign in [1.0, -1.0]:
                        if calls >= budget:
                            break
                        x_trial = best_x.copy()
                        x_trial[i] += sign * steps[i]
                        x_trial = np.clip(x_trial, -5, 5)
                        if np.allclose(x_trial, best_x, atol=1e-12):
                            continue
                        f_trial = evaluate(x_trial)
                        if f_trial < best_f:
                            best_x = x_trial
                            best_f = f_trial
                steps *= 0.7

        return best_x, best_f
# EVOLVE-BLOCK-END
