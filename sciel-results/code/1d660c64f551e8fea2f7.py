"""HYBRID: DE-based global search with Rastrigin-aware initialization and L-BFGS-B polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

class Optimizer:
    """Hybrid optimizer: DE for global basin finding + L-BFGS-B for precise local polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 330s kill limit

        call_count = 0
        max_calls = self.budget
        best_x = np.zeros(self.dim)
        best_f = float('inf')

        def counted_func(x):
            nonlocal call_count, best_x, best_f
            if time.time() > deadline:
                raise RuntimeError("Time limit exceeded")
            if call_count >= max_calls:
                raise RuntimeError("Budget exceeded")

            call_count += 1
            x = np.asarray(x, dtype=np.float64).ravel()[:self.dim]
            x = np.clip(x, -5.0, 5.0)
            f = float(func(x))

            if f < best_f:
                best_f = f
                best_x = x.copy()

            return f

        rng = np.random.default_rng(42)
        dim = self.dim

        # Phase 1: Rastrigin-aware coarse sampling to identify promising basins
        # Büche-Rastrigin has local minima near integer coordinates
        # Sample a mix of integer lattice points and random points for good coverage
        n_phase1 = min(80, max_calls // 5)
        
        # Generate integer lattice points in [-4, 4]^5 (subset)
        lattice_candidates = []
        for i in range(-4, 5, 2):  # Step by 2 to reduce count
            for j in range(-4, 5, 2):
                for k in range(-4, 5, 2):
                    for l in range(-4, 5, 2):
                        for m in range(-4, 5, 2):
                            lattice_candidates.append([float(i), float(j), float(k), float(l), float(m)])
        
        # Add random points
        random_points = rng.uniform(-5.0, 5.0, size=(n_phase1 - len(lattice_candidates) // 2, dim))
        
        # Combine and select
        all_points = np.array(lattice_candidates[:n_phase1 // 2])
        if len(random_points) > 0:
            all_points = np.vstack([all_points, random_points[:n_phase1 - len(all_points)]])
        
        # Shuffle
        perm = rng.permutation(len(all_points))
        all_points = all_points[perm]
        
        # Evaluate first phase
        for i in range(min(n_phase1, len(all_points))):
            if time.time() > deadline or call_count >= max_calls:
                break
            x = all_points[i]
            try:
                counted_func(x)
            except Exception:
                break

        if time.time() > deadline or call_count >= max_calls:
            return best_x, best_f

        # Phase 2: DE-based global refinement around best region
        remaining_budget = max_calls - call_count
        if remaining_budget > 40:
            # Use DE with bounds centered around best_x but wide enough to explore
            bounds_de = []
            for d in range(dim):
                lower = max(-5.0, best_x[d] - 3.0)
                upper = min(5.0, best_x[d] + 3.0)
                bounds_de.append((lower, upper))
            
            # Configure DE
            popsize = min(12, remaining_budget // 5)
            maxiter = max(1, (remaining_budget - popsize) // popsize)
            
            try:
                differential_evolution(
                    counted_func,
                    bounds_de,
                    maxiter=maxiter,
                    popsize=popsize,
                    tol=1e-10,
                    seed=42,
                    init='latinhypercube',
                    polish=False  # We'll do our own polish
                )
            except Exception:
                pass

        if time.time() > deadline or call_count >= max_calls:
            return best_x, best_f

        # Phase 3: L-BFGS-B local polish from best point found by DE
        remaining_budget = max_calls - call_count
        if remaining_budget > 20 and best_f < float('inf'):
            try:
                res = minimize(
                    counted_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
                )
            except Exception:
                pass

        if time.time() > deadline or call_count >= max_calls:
            return best_x, best_f

        # Phase 4: Final Nelder-Mead polish for extra precision
        remaining_budget = max_calls - call_count
        if remaining_budget > 10 and best_f < float('inf'):
            try:
                res = minimize(
                    counted_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 50, 'xatol': 1e-10, 'fatol': 1e-12}
                )
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
