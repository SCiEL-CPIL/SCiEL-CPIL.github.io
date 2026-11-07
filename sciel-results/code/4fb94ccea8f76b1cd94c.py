import numpy as np
import time

class Optimizer:
    """
    Derivative-free optimizer for BBOB f12 (Bent Cigar).
    
    Strategy:
    1. Population-based search with adaptive step sizes per dimension.
    2. Estimate local Hessian via finite differences to identify the valley direction.
    3. Take large steps along the valley (small curvature) and small steps perpendicular.
    4. Adapt the search distribution based on improvement history.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 330s limit

        eval_count = 0
        best_x = np.zeros(d)
        best_f = float('inf')

        def eval_func(x):
            nonlocal eval_count, best_x, best_f
            if eval_count >= budget or time.time() > deadline:
                return float('inf')
            
            x = np.clip(np.asarray(x, dtype=float).flatten(), -5, 5)
            
            try:
                f = float(func(x))
            except Exception:
                return float('inf')
                
            if not np.isfinite(f):
                return float('inf')

            eval_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
                
            return f

        # --- Phase 1: Global exploration ---
        rng = np.random.default_rng(42)
        
        # Evaluate origin and a few random points
        eval_func(np.zeros(d))
        
        for _ in range(20):
            if eval_count >= budget - 50:
                break
            x_rand = rng.uniform(-5, 5, d)
            eval_func(x_rand)

        # --- Phase 2: Adaptive population-based search ---
        # We maintain a population of candidate points and adapt step sizes
        # based on finite-difference Hessian estimates.

        # Initial step sizes: moderate for exploration
        step_sizes = np.ones(d) * 1.0
        
        # Population size
        pop_size = 15
        max_iter = (budget - eval_count - 10) // pop_size
        max_iter = min(max_iter, 30)  # Cap iterations to leave budget for refinement

        # Track best point and its neighborhood
        center = best_x.copy()
        
        for iteration in range(max_iter):
            if eval_count >= budget - 10 or time.time() > deadline:
                break

            # Generate population around center
            population = []
            for i in range(pop_size):
                # Mix of small and large perturbations
                scale = step_sizes * (0.5 + rng.random() * 1.5)
                x_new = center + rng.standard_normal(d) * scale
                x_new = np.clip(x_new, -5, 5)
                f_new = eval_func(x_new)
                if np.isfinite(f_new):
                    population.append((f_new, x_new))
            
            if not population:
                break

            # Sort by fitness
            population.sort(key=lambda t: t[0])
            
            # Update center to best in population
            best_pop_f, best_pop_x = population[0]
            if best_pop_f < best_f:
                best_f = best_pop_f
                best_x = best_pop_x.copy()
                center = best_x.copy()
            else:
                # If no improvement, keep center but shrink steps slightly
                step_sizes *= 0.7
                continue

            # Estimate Hessian via finite differences at the best point
            # Use central differences with small perturbations
            h = 1e-4
            f0 = best_f
            hessian = np.zeros((d, d))
            
            # Only estimate if we have enough budget
            if eval_count < budget - 30:
                for i in range(d):
                    if eval_count >= budget - 20:
                        break
                    # Central difference for diagonal
                    x_plus = best_x.copy()
                    x_minus = best_x.copy()
                    x_plus[i] += h
                    x_minus[i] -= h
                    x_plus = np.clip(x_plus, -5, 5)
                    x_minus = np.clip(x_minus, -5, 5)
                    
                    f_plus = eval_func(x_plus)
                    if not np.isfinite(f_plus):
                        f_plus = f0
                    f_minus = eval_func(x_minus)
                    if not np.isfinite(f_minus):
                        f_minus = f0
                    
                    hessian[i, i] = (f_plus - 2*f0 + f_minus) / (h*h)
                    
                    # Off-diagonal terms (forward differences for efficiency)
                    for j in range(i+1, d):
                        if eval_count >= budget - 15:
                            break
                        x_pp = best_x.copy()
                        x_pm = best_x.copy()
                        x_mp = best_x.copy()
                        x_mm = best_x.copy()
                        x_pp[i] += h
                        x_pp[j] += h
                        x_pm[i] += h
                        x_pm[j] -= h
                        x_mp[i] -= h
                        x_mp[j] += h
                        x_mm[i] -= h
                        x_mm[j] -= h
                        
                        x_pp = np.clip(x_pp, -5, 5)
                        x_pm = np.clip(x_pm, -5, 5)
                        x_mp = np.clip(x_mp, -5, 5)
                        x_mm = np.clip(x_mm, -5, 5)
                        
                        f_pp = eval_func(x_pp)
                        f_pm = eval_func(x_pm)
                        f_mp = eval_func(x_mp)
                        f_mm = eval_func(x_mm)
                        
                        if not np.isfinite(f_pp): f_pp = f0
                        if not np.isfinite(f_pm): f_pm = f0
                        if not np.isfinite(f_mp): f_mp = f0
                        if not np.isfinite(f_mm): f_mm = f0
                        
                        hessian[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4*h*h)
                        hessian[j, i] = hessian[i, j]
            
            # Adapt step sizes based on Hessian eigenvalues
            # Large eigenvalue -> small step (perpendicular to valley)
            # Small eigenvalue -> large step (along valley)
            try:
                eigvals, eigvecs = np.linalg.eigh(hessian)
                eigvals = np.abs(eigvals)  # Ensure positive
                # Normalize: map eigenvalues to step sizes
                # Small curvature -> large step, large curvature -> small step
                eigvals_clipped = np.clip(eigvals, 1e-6, 1e6)
                # Inverse relationship: step ~ 1/sqrt(curvature)
                step_scales = 1.0 / np.sqrt(eigvals_clipped)
                # Normalize to keep average step size reasonable
                step_scales = step_scales / np.mean(step_scales)
                # Clip to reasonable range
                step_scales = np.clip(step_scales, 0.1, 10.0)
                
                # Update step sizes per dimension based on projection onto eigenvectors
                # For simplicity, use the magnitude of each dimension's contribution
                for i in range(d):
                    # Compute how much dimension i aligns with low-curvature directions
                    # Use the diagonal of V @ diag(step_scales) @ V^T
                    # Simpler: just scale by the average of step_scales weighted by eigenvector components
                    weight = np.sum(eigvecs[:, i] ** 2 * step_scales)
                    step_sizes[i] *= (0.5 + 0.5 * weight)
                    # Clip step sizes
                    step_sizes[i] = np.clip(step_sizes[i], 0.01, 5.0)
            except np.linalg.LinAlgError:
                # Hessian estimation failed, keep current step sizes
                pass

            # Shrink steps slightly each iteration to converge
            step_sizes *= 0.95

        # --- Phase 3: Final refinement with coordinate-wise search ---
        if eval_count < budget - 5:
            # Simple coordinate-wise line search for final polish
            x = best_x.copy()
            f = best_f
            coord_steps = np.ones(d) * 0.1
            
            for _ in range(50):
                if eval_count >= budget or time.time() > deadline:
                    break
                
                improved = False
                for i in range(d):
                    if eval_count >= budget or time.time() > deadline:
                        break
                    
                    # Try positive
                    x_plus = x.copy()
                    x_plus[i] += coord_steps[i]
                    x_plus = np.clip(x_plus, -5, 5)
                    f_plus = eval_func(x_plus)
                    
                    if f_plus < f:
                        x = x_plus
                        f = f_plus
                        coord_steps[i] *= 1.5
                        improved = True
                    else:
                        coord_steps[i] *= 0.5
                    
                    # Try negative
                    x_minus = x.copy()
                    x_minus[i] -= coord_steps[i]
                    x_minus = np.clip(x_minus, -5, 5)
                    f_minus = eval_func(x_minus)
                    
                    if f_minus < f:
                        x = x_minus
                        f = f_minus
                        coord_steps[i] *= 1.5
                        improved = True
                    else:
                        coord_steps[i] *= 0.5
                
                if not improved:
                    break
                
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                
                if np.all(coord_steps < 1e-5):
                    break

        return best_x, best_f
