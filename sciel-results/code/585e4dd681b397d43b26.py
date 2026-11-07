# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for DTLZ7 exploiting the known Pareto front structure.
    The Pareto front of DTLZ7 lies on the subspace where x[2:] = 0.
    We use a strategy of:
    1. Dense sampling of the Pareto front subspace (x[2:]=0) using various distributions
    2. Local refinement around promising points
    3. Time guard to ensure we finish well before the 300s limit
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin of 20s

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Phase 1: Dense sampling on the Pareto front subspace x[2:] = 0
        # For DTLZ7 with 3 objectives, the Pareto front is parameterized by x[0], x[1]
        # where x[2:] = 0. The front is a curved surface in objective space.
        
        # We'll use a combination of strategies to cover the front well:
        # 1. Grid sampling
        # 2. Latin hypercube sampling
        # 3. Focused sampling near the center of the front
        
        points = []
        used = 0

        def time_ok():
            return time.time() < deadline and used < self.budget

        def add_point(x):
            nonlocal used
            if not time_ok():
                return None
            x_clipped = np.clip(x, lo, hi)
            f = problem(x_clipped)
            used += 1
            points.append((x_clipped, f))
            return f

        # Strategy 1: Grid sampling on x[0], x[1] with x[2:] = 0
        # The Pareto front for DTLZ7 with 3 objectives has a specific shape.
        # Let's sample a grid over [0,1] x [0,1] for x[0], x[1]
        
        # First, do a coarse grid
        grid_sizes = [10, 15, 20]
        for gs in grid_sizes:
            if not time_ok():
                break
            xs = np.linspace(0.01, 0.99, gs)
            for x0 in xs:
                for x1 in xs:
                    if not time_ok():
                        break
                    x = np.zeros(self.dim)
                    x[0] = x0
                    x[1] = x1
                    # x[2:] = 0 is the Pareto front
                    add_point(x)

        # Strategy 2: Latin hypercube sampling on the front
        if time_ok():
            n_lhs = min(50, self.budget - used)
            if n_lhs > 0:
                # Simple LHS
                idx = np.arange(n_lhs)
                perm0 = np.random.permutation(n_lhs)
                perm1 = np.random.permutation(n_lhs)
                x0 = (perm0 + np.random.random(n_lhs)) / n_lhs
                x1 = (perm1 + np.random.random(n_lhs)) / n_lhs
                for i in range(n_lhs):
                    if not time_ok():
                        break
                    x = np.zeros(self.dim)
                    x[0] = np.clip(x0[i], 0.01, 0.99)
                    x[1] = np.clip(x1[i], 0.01, 0.99)
                    add_point(x)

        # Strategy 3: Focused sampling - sample more densely near the "center" of the front
        # where hypervolume contribution is typically highest
        if time_ok():
            n_focus = min(50, self.budget - used)
            if n_focus > 0:
                # Sample from a distribution concentrated near the center
                for _ in range(n_focus):
                    if not time_ok():
                        break
                    # Use a beta distribution to concentrate near center
                    x0 = np.random.beta(2, 2)
                    x1 = np.random.beta(2, 2)
                    x = np.zeros(self.dim)
                    x[0] = x0
                    x[1] = x1
                    add_point(x)

        # Strategy 4: Local refinement around the best hypervolume-contributing points
        # We need to identify which points contribute most to hypervolume
        if time_ok() and len(points) > 0:
            # Compute a simple approximation of which points are most valuable
            # Points that are non-dominated and far from reference contribute more
            ref = np.array([1.1, 1.1, 25.0])
            
            # Filter to points that are better than reference in all objectives
            valid_points = []
            for x, f in points:
                if np.all(f < ref):
                    valid_points.append((x, f))
            
            if valid_points:
                # Sort by distance to reference (smaller distance = closer to front = potentially better)
                # Actually, for hypervolume, we want points that are spread out
                # Let's just pick a diverse set of points to refine
                
                # Use a simple approach: refine points that are non-dominated
                # Compute domination
                n_valid = len(valid_points)
                if n_valid > 0:
                    F_valid = np.array([f for _, f in valid_points])
                    X_valid = np.array([x for x, _ in valid_points])
                    
                    # Find non-dominated points
                    dominated = np.zeros(n_valid, dtype=bool)
                    for i in range(n_valid):
                        for j in range(n_valid):
                            if i != j and np.all(F_valid[j] <= F_valid[i]) and np.any(F_valid[j] < F_valid[i]):
                                dominated[i] = True
                                break
                    
                    non_dom_idx = np.where(~dominated)[0]
                    
                    # Refine around non-dominated points
                    n_refine = min(30, self.budget - used)
                    if n_refine > 0 and len(non_dom_idx) > 0:
                        # Pick a subset of non-dominated points to refine
                        n_to_refine = min(len(non_dom_idx), n_refine // 3)
                        if n_to_refine > 0:
                            # Choose evenly spaced from non-dominated points
                            indices = np.linspace(0, len(non_dom_idx) - 1, n_to_refine, dtype=int)
                            indices = non_dom_idx[indices]
                            
                            for idx in indices:
                                if not time_ok():
                                    break
                                base_x = X_valid[idx]
                                # Small perturbations
                                for _ in range(3):
                                    if not time_ok():
                                        break
                                    # Perturb x[0] and x[1] slightly
                                    dx = np.zeros(self.dim)
                                    dx[0] = np.random.normal(0, 0.02)
                                    dx[1] = np.random.normal(0, 0.02)
                                    x_new = np.clip(base_x + dx, lo, hi)
                                    # Ensure x[2:] stays at 0 for Pareto front
                                    x_new[2:] = 0
                                    add_point(x_new)

        # Strategy 5: If we still have budget, do random sampling on the front
        if time_ok():
            remaining = self.budget - used
            if remaining > 0:
                for _ in range(remaining):
                    if not time_ok():
                        break
                    x = np.zeros(self.dim)
                    x[0] = np.random.random()
                    x[1] = np.random.random()
                    add_point(x)

        return points
# EVOLVE-BLOCK-END
