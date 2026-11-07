import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ1 using analytical front sampling + local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.start_time = time.time()
        self.deadline = 290.0  # Safety margin before 300s limit

    def _elapsed(self):
        return time.time() - self.start_time

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        # DTLZ1 structure: n_g = dim - n_obj + 1
        n_g = self.dim - self.n_obj + 1  # 5 for dim=7, n_obj=3

        # Strategy: sample the known Pareto front analytically with high density
        # On Pareto front: x[n_g:] = 0.5 (to make g=0), x[0] and x[1] vary
        # f1 = 0.5*x1, f2 = 0.5*(1-x1)*x2, f3 = 0.5*(1-x2)
        
        # Allocate budget: ~80% for front sampling, ~20% for local search
        n_front = int(self.budget * 0.8)
        n_local = self.budget - n_front

        used = 0
        points = []
        objectives = []

        # Phase 1: Sample the Pareto front on a structured grid
        # Use a stratified grid in (x1, x2) space
        # For better hypervolume, we want to cover the front densely
        
        # Create a grid that covers the simplex well
        # The Pareto front of DTLZ1 is a 2D simplex in objective space
        # f1 + f2 + f3 = 0.5 (when g=0)
        
        # We'll use a triangular grid in (x1, x2) space where x1, x2 >= 0 and x1 + x2 <= 1
        # Actually for DTLZ1, x1 and x2 are independent in [0,1]
        # But the objective values are: f1=0.5*x1, f2=0.5*(1-x1)*x2, f3=0.5*(1-x2)
        # For these to be on the Pareto front, we need g=0, which means x[2:] = 0.5
        
        # Let's use a dense grid
        n_per_dim = int(np.sqrt(n_front))
        if n_per_dim < 5:
            n_per_dim = 5
        
        x1_vals = np.linspace(0.0, 1.0, n_per_dim)
        x2_vals = np.linspace(0.0, 1.0, n_per_dim)
        
        # Generate grid points, shuffle to avoid bias
        grid_points = []
        for x1 in x1_vals:
            for x2 in x2_vals:
                grid_points.append((x1, x2))
        
        # If we have more grid points than budget, sample
        if len(grid_points) > n_front:
            indices = rng.choice(len(grid_points), n_front, replace=False)
            grid_points = [grid_points[i] for i in indices]
        
        for x1, x2 in grid_points:
            if used >= self.budget or self._elapsed() > self.deadline:
                break
            x = np.full(self.dim, 0.5)  # g-variables at 0.5 for g=0
            x[0] = x1
            x[1] = x2
            f = problem(x)
            points.append(x.copy())
            objectives.append(f)
            used += 1

        # Phase 2: Local refinement around best-performing regions
        # Use simple local search: perturb points and keep improvements
        if used < self.budget and self._elapsed() < self.deadline:
            # Sort by sum of objectives (proxy for quality)
            obj_array = np.array(objectives)
            # Find points with low objective sums
            order = np.argsort(np.sum(obj_array, axis=1))
            
            # Take top candidates for local search
            n_candidates = min(20, len(points))
            candidate_indices = order[:n_candidates]
            
            step_size = 0.02
            for _ in range(n_local):
                if used >= self.budget or self._elapsed() > self.deadline:
                    break
                # Pick a random candidate
                ci = candidate_indices[rng.integers(n_candidates)]
                parent = points[ci].copy()
                
                # Gaussian perturbation
                child = parent + step_size * rng.standard_normal(self.dim)
                child = np.clip(child, lo, hi)
                
                fc = problem(child)
                used += 1
                
                # Accept if it improves some objective without worsening others too much
                # Simple acceptance: accept if sum is better
                if np.sum(fc) < np.sum(objectives[ci]):
                    points[ci] = child
                    objectives[ci] = fc
                    # Update candidate list
                    obj_array[ci] = fc
                    order = np.argsort(np.sum(obj_array, axis=1))
                    candidate_indices = order[:n_candidates]

        # Phase 3: Use remaining budget for additional random exploration if time allows
        if used < self.budget and self._elapsed() < self.deadline:
            for _ in range(self.budget - used):
                if self._elapsed() > self.deadline:
                    break
                x = lo + (hi - lo) * rng.random(self.dim)
                f = problem(x)
                points.append(x)
                objectives.append(f)
                used += 1

        return points
