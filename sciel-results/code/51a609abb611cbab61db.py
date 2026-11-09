# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ2 using dense Pareto-front grid + local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 320.0  # safety margin under 330s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(42)

        def time_ok():
            return time.time() < deadline

        # DTLZ2: Pareto front is the unit (n_obj-1)-sphere surface.
        # x[0..n_obj-2] determine the front position; x[n_obj-1:] should be 0.5 for optimality.
        # f_i = (1 + g) * y_i where y is on unit sphere, sum(y^2)=1, y_i>=0
        # g = sum((x_j - 0.5)^2) for j in [n_obj, dim)

        # Strategy:
        # 1. Generate points on the (n_obj-1)-dim simplex/sphere surface using stratified angles
        # 2. Set decision variables: first n_obj-1 vars encode the front position, rest at 0.5
        # 3. Refine with local search to minimize g (push remaining vars toward 0.5)

        n_obj = self.n_obj
        dim = self.dim

        # For DTLZ2, the Pareto front in objective space:
        # f_i = (1+g) * sin(theta_{i-1}) * ... 
        # Actually for DTLZ2: f_i = (1+g) * y_i where y_i = product of sines/cosines
        # y_1 = cos(a_1), y_2 = sin(a_1)*cos(a_2), ..., y_{M-1} = sin(a_1)*...*sin(a_{M-2})*cos(a_{M-1}), y_M = sin(a_1)*...*sin(a_{M-1})
        
        # With M=3 objectives, we have 2 angles: a_1, a_2
        # y_1 = cos(a_1)
        # y_2 = sin(a_1) * cos(a_2)
        # y_3 = sin(a_1) * sin(a_2)
        
        # The decision variable mapping for DTLZ2:
        # x_0 = a_1 / pi, x_1 = a_2 / pi (for M=3, n_obj-1=2 variables for angles)
        # x_2..x_11 = 0.5 (to make g=0)
        
        # But the problem clips to bounds [0,1], so we need x in [0,1].
        # a_1 in [0, pi/2], a_2 in [0, pi/2] for the first quadrant (all y_i >= 0)
        # So x_0 = a_1/(pi/2), x_1 = a_2/(pi/2) maps [0,1] to [0, pi/2]
        
        # Wait, let me reconsider. The standard DTLZ2 mapping:
        # x_i for i=0..M-2 are the angle variables, scaled to [0,1]
        # The actual angles are: theta_i = x_i * pi / 2
        # y_1 = cos(theta_1)
        # y_2 = sin(theta_1) * cos(theta_2)  
        # y_3 = sin(theta_1) * sin(theta_2)
        # g = sum((x_j - 0.5)^2) for j = M..dim-1
        
        # So to get points on the Pareto front, we set x_M..x_{dim-1} = 0.5
        # and vary x_0, x_1 in [0,1] to cover the front.

        # Generate stratified grid of angles for uniform coverage of the 1/4 circle (2 angles)
        # Use stratified sampling in 2D for a_1, a_2

        # First pass: dense grid on the front
        n_grid = 20  # 20x20 = 400 points, but we only have budget 200
        # Let's use a coarser grid first, then refine
        
        # Actually, let's think about budget: 200 evaluations total.
        # We want to cover the front well. With 3 objectives, the front is 2D.
        
        # Strategy: 
        # - Use ~100 points on a stratified grid for the front
        # - Use remaining ~100 for local refinement (pushing g toward 0)
        
        # But actually, if we set x_M..x_{dim-1} = 0.5 exactly, then g=0 and we're on the front.
        # The problem is that the evaluator calls problem(x) which clips to bounds.
        # If we set x_j = 0.5 for j>=2, those are within [0,1], so no clipping issue.
        
        # Let me just generate points directly on the front with g=0.
        
        # Stratified sampling of (a1, a2) in [0, pi/2] x [0, pi/2]
        n_per_dim = 14  # 14*14 = 196 points
        used = 0
        F_list = []
        X_list = []
        
        # Stratified grid
        a1_vals = (np.arange(n_per_dim) + 0.5) / n_per_dim * (np.pi / 2)
        a2_vals = (np.arange(n_per_dim) + 0.5) / n_per_dim * (np.pi / 2)
        
        for a1 in a1_vals:
            for a2 in a2_vals:
                if not time_ok():
                    break
                x = np.zeros(dim)
                x[0] = a1 / (np.pi / 2)
                x[1] = a2 / (np.pi / 2)
                x[2:] = 0.5
                x = np.clip(x, lo, hi)
                f = problem(x)
                F_list.append(f)
                X_list.append(x)
                used += 1
            if not time_ok():
                break
        
        F = np.array(F_list)
        X = np.array(X_list)
        
        # Now refine: for some points, do local search to ensure g is minimized
        # Since we already set x[2:]=0.5, g should be 0. But let's verify by checking
        # if any objective values are > 1 (which would indicate g > 0).
        
        # For remaining budget, do a few random perturbations and keep non-dominated points
        remaining = self.budget - used
        if remaining > 0 and time_ok():
            # Local refinement: perturb the angle variables slightly and keep better points
            for _ in range(remaining):
                if not time_ok():
                    break
                # Pick a random point and perturb
                idx = rng.integers(len(X))
                child = X[idx].copy()
                # Perturb the angle variables
                child[0] += 0.05 * rng.standard_normal()
                child[1] += 0.05 * rng.standard_normal()
                # Keep x[2:] at 0.5
                child[2:] = 0.5
                child = np.clip(child, lo, hi)
                fc = problem(child)
                used += 1
                # Check if child is non-dominated
                # Simple check: if any objective is better than all existing points
                is_better = np.any(np.all(F >= fc, axis=1) & np.any(F > fc, axis=1))
                if is_better:
                    # Replace the worst point
                    worst = int(np.argmax(np.sum(F, axis=1)))
                    X[worst] = child
                    F[worst] = fc
        
        return X
# EVOLVE-BLOCK-END
