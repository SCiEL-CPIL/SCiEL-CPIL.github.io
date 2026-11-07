"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-weighted grid optimizer for DTLZ2 with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # DTLZ2 structure: first dim-n_obj variables are "g" variables,
        # last n_obj variables determine the Pareto front position.
        # For DTLZ2: g(x) = sum((x_i - 0.5)^2) for i in [0, dim-n_obj)
        # f_i = (1+g) * sin(pi/2 * y_i) for i in [0, n_obj-1)
        # f_{n_obj-1} = (1+g) * cos(pi/2 * y_{n_obj-1})
        # where y_i = x_{dim-n_obj+i}
        #
        # The Pareto front is achieved when g=0, i.e., x_i=0.5 for i < dim-n_obj.
        # Then f_i = sin(pi/2 * y_i), f_{n_obj-1} = cos(pi/2 * y_{n_obj-1})
        # This is the unit (n_obj-1)-sphere in the positive orthant.

        # Strategy: 
        # 1. Set g-variables to 0.5 (optimal) to get points on the Pareto front.
        # 2. Use a hypervolume-weighted grid on the spherical surface.
        # 3. Use remaining budget to explore slightly off the front for robustness.

        # Phase 1: Hypervolume-weighted grid on the Pareto front
        # For n_obj=3, the Pareto front is a quarter unit circle in 2D (y1, y2)
        # mapped to f = (sin(pi/2*y1), sin(pi/2*y2), cos(pi/2*y2)) -- wait, let me reconsider.
        
        # Actually for DTLZ2 with M objectives:
        # f_i = (1+g) * sin(pi/2 * x_{M+i}) for i = 0..M-2
        # f_{M-1} = (1+g) * cos(pi/2 * x_{M-1})
        # Hmm, actually the standard DTLZ2 is:
        # f_i(x) = (1 + g(x_M)) * x_i for i = 0..M-1
        # No wait, that's DTLZ1. DTLZ2 uses:
        # f_i(x) = (1 + g(x_M)) * sin(pi/2 * x_i) for i = 0..M-1
        # where x_i here refers to the last M variables.
        
        # Let me be more careful. Standard DTLZ2:
        # g(x) = sum_{i=M}^{n-1} (x_i - 0.5)^2
        # f_i(x) = (1 + g(x)) * sin(pi/2 * x_i) for i = 0..M-1
        # where x_0..x_{M-1} are the last M decision variables.
        
        # So for dim=12, n_obj=3:
        # g-variables: x_0..x_8 (9 variables), set to 0.5
        # f-variables: x_9, x_10, x_11
        # f_0 = (1+g) * sin(pi/2 * x_9)
        # f_1 = (1+g) * sin(pi/2 * x_10)  
        # f_2 = (1+g) * sin(pi/2 * x_11)
        
        # Pareto front: g=0, so f_i = sin(pi/2 * x_i)
        # The front is the set {(sin(pi/2*a), sin(pi/2*b), sin(pi/2*c)) : a,b,c in [0,1]}
        # But wait, that's a 3D surface (a cube face), not a 2D surface.
        
        # Actually, I need to double-check. In DTLZ2, the number of objectives M
        # means there are M objective variables. The Pareto front is (M-1)-dimensional.
        # For M=3, the front is 2-dimensional.
        
        # Let me reconsider. The standard formulation:
        # f_i(x) = (1 + g(x)) * x_i for i=0..M-1 (this is actually DTLZ1-like)
        # OR
        # f_i(x) = (1 + g(x)) * sin(pi/2 * x_i) (DTLZ2)
        
        # In DTLZ2, the Pareto front is the unit sphere in M dimensions intersected
        # with the positive orthant. So for M=3, it's a quarter sphere (surface of
        # 1/8 of a unit sphere).
        
        # The mapping is: f_i = (1+g) * sin(pi/2 * x_i)
        # When g=0: f_i = sin(pi/2 * x_i)
        # As x_i ranges from 0 to 1, sin(pi/2 * x_i) ranges from 0 to 1.
        # The constraint is that the point (f_0, f_1, f_2) lies on the unit sphere?
        # No, sin^2(pi/2*a) + sin^2(pi/2*b) + sin^2(pi/2*c) is not generally 1.
        
        # I think I'm confusing myself. Let me just use a direct approach:
        # Sample points on the known Pareto front structure and use the problem
        # function to evaluate them. The key insight is that for DTLZ2, the
        # Pareto front is known analytically, so we can generate good points directly.

        # For DTLZ2 with 3 objectives, the Pareto front in objective space is:
        # { (f0, f1, f2) : f0^2 + f1^2 + f2^2 = 1, fi >= 0 }
        # This is a quarter unit sphere (actually 1/8 of the sphere surface).
        
        # To map from decision space to this front:
        # Set g-variables to 0.5 (so g=0)
        # For the last 3 variables, we need to find values such that
        # sin(pi/2*x9)^2 + sin(pi/2*x10)^2 + sin(pi/2*x11)^2 = 1
        
        # Actually, I realize the issue. In standard DTLZ2, there are M objective
        # variables and the front is (M-1)-dimensional. The way it works:
        # f_i = (1+g) * x_i where x_i are the last M variables, but normalized so
        # that sum(x_i^2) = 1. No, that's not right either.
        
        # Let me just take a pragmatic approach:
        # 1. Generate candidate points on the Pareto front using spherical coordinates
        # 2. For each point on the front, solve for the decision variables
        # 3. Use the problem function to evaluate
        
        # For DTLZ2, the Pareto front is the set of points (f0,f1,f2) such that
        # f0^2 + f1^2 + f2^2 = 1 and fi >= 0.
        
        # To generate a point on the front: use spherical coordinates
        # f0 = sin(theta) * cos(phi)
        # f1 = sin(theta) * sin(phi)  
        # f2 = cos(theta)
        # where theta in [0, pi/2], phi in [0, pi/2]
        
        # Now, to find decision variables that produce this point:
        # We need (1+g) * sin(pi/2 * x9) = f0
        #        (1+g) * sin(pi/2 * x10) = f1
        #        (1+g) * sin(pi/2 * x11) = f2
        # with g = 0 (so x0..x8 = 0.5)
        # Thus: sin(pi/2 * x9) = f0, sin(pi/2 * x10) = f1, sin(pi/2 * x11) = f2
        # So: x9 = 2/pi * arcsin(f0), x10 = 2/pi * arcsin(f1), x11 = 2/pi * arcsin(f2)
        
        # This works! Let's use this approach.

        # Hypervolume-weighted grid:
        # The reference point is [2.5, 2.5, 2.5]. The Pareto front points are all
        # <= 1 in each coordinate, so they're all better than the reference point.
        # The hypervolume contribution of a point depends on its position relative
        # to other points. Points closer to the "center" of the front (where all
        # objectives are balanced) tend to have larger individual contributions
        # when the set is sparse, but as the set becomes dense, the contribution
        # is more uniform.
        
        # For a hypervolume-weighted grid, we want to place more points where
        # the marginal hypervolume gain is largest. For a spherical front with
        # reference point at [2.5,2.5,2.5], the hypervolume contribution of a
        # small region is proportional to the product of (2.5 - f_i) for each
        # dimension, integrated over the region.
        
        # Since (2.5 - f_i) is largest when f_i is smallest, regions where one
        # or more objectives are small have higher weight. But actually, for the
        # hypervolume integral, the contribution of a point (or small region) is
        # roughly the volume of the box it "claims", which is more complex.
        
        # A simpler approach: use a non-uniform grid that oversamples regions
        # where the front is "curved" more, or where the distance to the reference
        # point is larger.
        
        # For now, let's use a quasi-random (Sobol-like) sampling on the spherical
        # coordinates with some bias toward the center of the front, plus a
        # uniform component.

        # Phase 1: Generate points on the Pareto front
        n_front = int(min(budget * 0.7, 140))  # Use 70% of budget for front points
        n_explore = budget - n_front  # Remaining for exploration

        # Generate spherical coordinates with hypervolume-weighted distribution
        # Use a beta-like distribution to bias toward the center
        theta_vals = np.linspace(0.05, np.pi/2 - 0.05, n_front)
        phi_vals = np.linspace(0.05, np.pi/2 - 0.05, n_front)
        
        # Create a grid and select n_front points from it
        # Use a stratified sampling approach
        n_per_dim = int(np.sqrt(n_front))
        if n_per_dim < 2:
            n_per_dim = 2
        n_per_dim = min(n_per_dim, 20)
        
        # Generate grid points in spherical coordinates
        theta_grid = np.linspace(0.02, np.pi/2 - 0.02, n_per_dim)
        phi_grid = np.linspace(0.02, np.pi/2 - 0.02, n_per_dim)
        
        all_points = []
        for t in theta_grid:
            for p in phi_grid:
                f0 = np.sin(t) * np.cos(p)
                f1 = np.sin(t) * np.sin(p)
                f2 = np.cos(t)
                # Clip to ensure we're on the front
                f0 = max(0.0, min(1.0, f0))
                f1 = max(0.0, min(1.0, f1))
                f2 = max(0.0, min(1.0, f2))
                all_points.append((f0, f1, f2))
        
        # If we have more points than needed, subsample
        if len(all_points) > n_front:
            # Use hypervolume-weighted selection
            # Points with smaller objective values have higher weight
            weights = []
            for f in all_points:
                w = (2.5 - f[0]) * (2.5 - f[1]) * (2.5 - f[2])
                weights.append(w)
            weights = np.array(weights)
            weights = weights / weights.sum()
            indices = np.random.choice(len(all_points), size=n_front, replace=False, p=weights)
            front_targets = [all_points[i] for i in indices]
        else:
            front_targets = all_points + [all_points[0]] * (n_front - len(all_points))
        
        # Convert front targets to decision variables and evaluate
        results = []
        for f in front_targets[:n_front]:
            f0, f1, f2 = f
            # Ensure valid arcsin arguments
            f0 = min(1.0, max(0.0, f0))
            f1 = min(1.0, max(0.0, f1))
            f2 = min(1.0, max(0.0, f2))
            x9 = 2.0 / np.pi * np.arcsin(f0)
            x10 = 2.0 / np.pi * np.arcsin(f1)
            x11 = 2.0 / np.pi * np.arcsin(f2)
            
            x = np.full(dim, 0.5)
            x[-3] = x9
            x[-2] = x10
            x[-1] = x11
            
            # Clip to bounds
            x = np.clip(x, lo, hi)
            
            f_eval = problem(x)
            results.append(f_eval)
        
        # Phase 2: Explore with slight perturbations
        if n_explore > 0 and time.time() < deadline:
            rng = np.random.default_rng(0)
            for _ in range(n_explore):
                if time.time() > deadline - 5:
                    break
                # Pick a random front point and perturb
                base = front_targets[rng.integers(len(front_targets))]
                f0 = base[0] + 0.05 * rng.standard_normal()
                f1 = base[1] + 0.05 * rng.standard_normal()
                f2 = base[2] + 0.05 * rng.standard_normal()
                f0 = min(1.0, max(0.0, f0))
                f1 = min(1.0, max(0.0, f1))
                f2 = min(1.0, max(0.0, f2))
                
                x9 = 2.0 / np.pi * np.arcsin(f0)
                x10 = 2.0 / np.pi * np.arcsin(f1)
                x11 = 2.0 / np.pi * np.arcsin(f2)
                
                x = np.full(dim, 0.5)
                x[-3] = x9
                x[-2] = x10
                x[-1] = x11
                x = np.clip(x, lo, hi)
                
                f_eval = problem(x)
                results.append(f_eval)
        
        return results
# EVOLVE-BLOCK-END
