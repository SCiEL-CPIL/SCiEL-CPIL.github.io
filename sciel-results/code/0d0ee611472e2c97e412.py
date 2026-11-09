"""Multi-Start Nelder-Mead optimizer for BBOB f22."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings

warnings.filterwarnings('ignore')

class Optimizer:
    """
    Optimizer for BBOB f22 (Gallagher 21 peaks).
    Strategy:
    1. Generate diverse starting points using Latin Hypercube Sampling.
    2. Evaluate all starting points.
    3. Run Nelder-Mead local refinement on the best starting points.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 290.0  # Safety margin before 300s kill
        best_x = np.zeros(self.dim)
        best_f = np.inf
        calls_made = 0
        n = self.dim

        def check_time():
            return (time.time() - start_time) < time_limit

        def safe_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= self.budget:
                return np.inf
            if not check_time():
                return np.inf
            calls_made += 1
            try:
                fx = func(x)
                fx = float(fx)
            except Exception:
                return np.inf
            if not np.isfinite(fx):
                fx = np.inf
            if fx < best_f:
                best_f = fx
                best_x = np.array(x, dtype=float).copy()
            return fx

        def nelder_mead(x_start, f_start, max_steps=60, initial_delta=0.5):
            nonlocal calls_made, best_x, best_f
            dim = self.dim
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_shrink = 0.5

            simplex = [x_start.copy()]
            for i in range(dim):
                xi = x_start.copy()
                xi[i] += initial_delta
                xi = np.clip(xi, -5, 5)
                if np.allclose(xi, x_start):
                    xi[i] = x_start[i] + initial_delta * 0.1
                    xi = np.clip(xi, -5, 5)
                simplex.append(xi)

            simplex_f = [f_start]
            for i in range(1, dim + 1):
                if not check_time() or calls_made >= self.budget:
                    return
                fx = safe_func(simplex[i])
                simplex_f.append(fx)

            for step in range(max_steps):
                if not check_time() or calls_made >= self.budget:
                    break

                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                centroid = np.mean(simplex[:-1], axis=0)
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fxr = safe_func(xr)
                if not check_time() or calls_made >= self.budget:
                    break

                if fxr < simplex_f[0]:
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fxe = safe_func(xe)
                    if not check_time() or calls_made >= self.budget:
                        break
                    if fxe < fxr:
                        simplex[-1] = xe
                        simplex_f[-1] = fxe
                    else:
                        simplex[-1] = xr
                        simplex_f[-1] = fxr
                elif fxr < simplex_f[-2]:
                    simplex[-1] = xr
                    simplex_f[-1] = fxr
                else:
                    xc = centroid + rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fxc = safe_func(xc)
                    if not check_time() or calls_made >= self.budget:
                        break
                    if fxc < simplex_f[-1]:
                        simplex[-1] = xc
                        simplex_f[-1] = fxc
                    else:
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma_shrink * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            if not check_time() or calls_made >= self.budget:
                                return
                            simplex_f[i] = safe_func(simplex[i])

                if np.max(simplex_f) - np.min(simplex_f) < 1e-14:
                    break

        def generate_lhs_points(n_points, dim):
            """Generate n_points using Latin Hypercube Sampling in [-5, 5]^dim."""
            points = np.zeros((n_points, dim))
            for d in range(dim):
                # Stratify the interval [0, 1] into n_points equal intervals
                # Pick one random point in each interval
                intervals = np.linspace(0, 1, n_points + 1)
                random_points = np.random.uniform(intervals[:-1], intervals[1:])
                # Shuffle to decorrelate dimensions
                np.random.shuffle(random_points)
                # Map to [-5, 5]
                points[:, d] = random_points * 10 - 5
            return points

        # Phase 1: Generate diverse starting points
        # Use 20% of budget for initial sampling, at least 10 points
        n_starts = max(10, int(self.budget * 0.2))
        # Cap number of starts to avoid wasting budget if budget is huge (not the case here)
        n_starts = min(n_starts, 50)
        
        starts = generate_lhs_points(n_starts, n)
        
        # Evaluate all starting points
        start_vals = []
        for x in starts:
            if not check_time() or calls_made >= self.budget:
                break
            fx = safe_func(x)
            start_vals.append(fx)
        
        if not start_vals:
            return best_x, best_f

        # Sort starting points by fitness
        indices = np.argsort(start_vals)
        
        # Phase 2: Local Refinement with Nelder-Mead
        # Select top K points for local refinement
        # We have remaining budget. Let's allocate remaining budget to top K points.
        # K should be small enough to allow deep refinement.
        # Let's try top 5 points.
        k_best = min(5, len(indices))
        
        remaining_budget = self.budget - calls_made
        if remaining_budget <= 0:
            return best_x, best_f
        
        # Allocate budget to each NM run
        # NM needs at least dim+1 for simplex initialization + steps
        # Let's give each run a fair share
        nm_budget_per_run = int(remaining_budget / k_best)
        
        # Ensure we don't overspend
        for idx in indices[:k_best]:
            if not check_time() or calls_made >= self.budget:
                break
            
            x_start = starts[idx]
            f_start = start_vals[idx]
            
            # Run Nelder-Mead
            # max_steps is a soft limit, safe_func handles hard budget
            nelder_mead(x_start, f_start, max_steps=nm_budget_per_run, initial_delta=0.5)

        return best_x, best_f

# EVOLVE-BLOCK-END
