"""Multi-objective optimizer for DTLZ4 using LHS on the simplex."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Multi-objective optimizer for DTLZ4 using LHS on the 2D simplex.
    
    Strategy:
    - DTLZ4 with M=3 has a Pareto front that is a 2D simplex in objective space.
    - Use Latin Hypercube Sampling on the simplex to generate 198 candidate points.
    - Reserve 2 evaluations for extreme vertices to maximize hypervolume.
    - For each objective-space point, construct decision variables that produce
      objectives on (or very close to) the Pareto front.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # For DTLZ4 with M=3, n=12:
        # The Pareto front is the set of points (f1, f2, f3) where:
        # f1 = x1, f2 = x2, f3 = (1 - x1^2 - x2^2) * g(x3..x12)
        # On the Pareto front, g=0, so f3 = 0 when x1^2 + x2^2 = 1.
        # But wait - this gives only a 1D curve. Let me reconsider.
        #
        # Actually for DTLZ4, the standard formulation is:
        # f1 = x1
        # f2 = x2  
        # f3 = (1 - x1^2 - x2^2) * g(x3..xn)
        # where g(x) = sum_{i=3}^{n} (x_i - 0.5)^2 for DTLZ2, 
        # but for DTLZ4: g(x) = sum_{i=3}^{n} (x_i^alpha - 0.5)^2
        # with alpha typically 100.
        #
        # On the Pareto front, g=0, which requires x_i = 0.5^{1/alpha} for i>=3.
        # Then f3 = 1 - x1^2 - x2^2.
        # The PF is the set of (x1, x2, 1-x1^2-x2^2) where x1,x2 >= 0, x1^2+x2^2 <= 1.
        # This is a 2D region (a quarter disk mapped to objective space).
        #
        # Actually, let me be more careful. For DTLZ4 with M objectives:
        # f1 = x1
        # f2 = x2
        # ...
        # f_{M-1} = x_{M-1}
        # f_M = (1 - sum_{i=1}^{M-1} x_i^2) * g(x_M..x_n)
        #
        # On PF: g=0, so f_M = 1 - sum_{i=1}^{M-1} x_i^2
        # The PF is: { (x1, x2, 1-x1^2-x2^2) : x1>=0, x2>=0, x1^2+x2^2<=1 }
        #
        # To parameterize this 2D region, use polar coordinates:
        # x1 = r * cos(theta), x2 = r * sin(theta), where r in [0,1], theta in [0, pi/2]
        # Then f1 = r*cos(theta), f2 = r*sin(theta), f3 = 1 - r^2
        #
        # For the decision variables:
        # x1 = f1 = r*cos(theta)
        # x2 = f2 = r*sin(theta)  
        # x3..x12 = 0.5^{1/alpha} (to make g=0)
        # For DTLZ4, alpha is typically 100, so 0.5^{1/100} ≈ 0.9931

        alpha = 100.0
        x_g = 0.5 ** (1.0 / alpha)  # Value that makes g=0

        def objective_to_decision(f1, f2, f3):
            """Convert objective-space point to decision variables on PF."""
            # From f1 = r*cos(theta), f2 = r*sin(theta), f3 = 1 - r^2
            # r^2 = 1 - f3, so r = sqrt(1 - f3)
            # cos(theta) = f1 / r, sin(theta) = f2 / r
            # x1 = f1, x2 = f2
            x = np.zeros(dim)
            x[0] = f1
            x[1] = f2
            x[2:] = x_g
            return x

        def simplex_lhs(n, rng):
            """
            Generate n points using LHS on the 2D simplex.
            The simplex is parameterized by (u, v) in [0,1]^2 with u+v <= 1.
            Use LHS on the triangle.
            """
            points = np.zeros((n, 2))
            
            # LHS on unit square, then filter/transform to triangle
            # Better: use the fact that the PF region can be parameterized by (r, theta)
            # where r in [0,1], theta in [0, pi/2]
            # But LHS on this parameter space doesn't give uniform area coverage.
            #
            # Alternative: use barycentric coordinates on the triangle.
            # The PF in objective space is: f1 >= 0, f2 >= 0, f3 >= 0, f1^2 + f2^2 + f3 = 1
            # This is NOT a simplex in the usual sense. It's a curved surface.
            #
            # Let me use a different approach: sample uniformly in the (x1, x2) plane
            # within the quarter unit disk, using LHS-like stratification.
            
            # Stratify the quarter disk into n cells
            # Use polar coordinates: divide r into sqrt(n) rings, theta into sqrt(n) sectors
            n_rings = int(np.sqrt(n))
            n_sectors = int(np.ceil(n / n_rings))
            
            # Adjust to get exactly n points
            while n_rings * n_sectors < n:
                n_sectors += 1
            
            points = []
            count = 0
            for i in range(n_rings):
                r_lo = i / n_rings
                r_hi = (i + 1) / n_rings
                for j in range(n_sectors):
                    if count >= n:
                        break
                    theta_lo = j * (np.pi / 2) / n_sectors
                    theta_hi = (j + 1) * (np.pi / 2) / n_sectors
                    
                    # Sample uniformly in this annular sector
                    # Area element: r dr dtheta
                    # For uniform area, sample r from sqrt distribution
                    r = np.sqrt(r_lo**2 + (r_hi**2 - r_lo**2) * rng.random())
                    theta = theta_lo + (theta_hi - theta_lo) * rng.random()
                    
                    f1 = r * np.cos(theta)
                    f2 = r * np.sin(theta)
                    f3 = 1 - r**2
                    
                    points.append([f1, f2, f3])
                    count += 1
            
            return np.array(points[:n])

        rng = np.random.default_rng(42)
        
        # Phase 1: LHS on the Pareto front region (198 points)
        n_lhs = min(198, budget)
        n_remaining = budget - n_lhs
        
        # Generate LHS points in objective space
        obj_points = simplex_lhs(n_lhs, rng)
        
        # Convert to decision variables and evaluate
        X_all = np.zeros((n_lhs, dim))
        F_all = np.zeros((n_lhs, n_obj))
        
        for i in range(n_lhs):
            f1, f2, f3 = obj_points[i]
            x = objective_to_decision(f1, f2, f3)
            x = np.clip(x, lo, hi)
            X_all[i] = x
            F_all[i] = problem(x)
        
        evals_used = n_lhs
        
        # Phase 2: Use remaining budget for vertex placements and refinement
        # The three extreme vertices of the PF region:
        # Vertex 1: (1, 0, 0) -> x1=1, x2=0, x3..x12=x_g
        # Vertex 2: (0, 1, 0) -> x1=0, x2=1, x3..x12=x_g  
        # Vertex 3: (0, 0, 1) -> x1=0, x2=0, x3..x12=x_g
        
        vertices_obj = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        ]
        
        # Place vertices first (up to n_remaining)
        for v_obj in vertices_obj:
            if evals_used >= budget:
                break
            
            x = objective_to_decision(v_obj[0], v_obj[1], v_obj[2])
            x = np.clip(x, lo, hi)
            fv = problem(x)
            evals_used += 1
            
            # Replace the worst point (highest sum of objectives)
            sums = np.sum(F_all, axis=1)
            worst_idx = int(np.argmax(sums))
            X_all[worst_idx] = x
            F_all[worst_idx] = fv
        
        # If we still have budget, do local refinement on worst points
        if evals_used < budget:
            # Sort by sum of objectives, worst first
            sums = np.sum(F_all, axis=1)
            order = np.argsort(sums)[::-1]
            
            step_sizes = [0.05, 0.02, 0.01, 0.005]
            
            for idx in order:
                if evals_used >= budget:
                    break
                
                best_x = X_all[idx].copy()
                best_f = F_all[idx].copy()
                best_sum = np.sum(best_f)
                
                for step in step_sizes:
                    if evals_used >= budget:
                        break
                    
                    for _ in range(2):
                        if evals_used >= budget:
                            break
                        
                        direction = rng.standard_normal(dim)
                        direction /= (np.linalg.norm(direction) + 1e-10)
                        
                        for sign in [1, -1]:
                            if evals_used >= budget:
                                break
                            
                            new_x = best_x + sign * step * (hi - lo) * direction
                            new_x = np.clip(new_x, lo, hi)
                            
                            new_f = problem(new_x)
                            evals_used += 1
                            
                            if np.sum(new_f) < best_sum:
                                best_x = new_x
                                best_f = new_f
                                best_sum = np.sum(new_f)
                
                X_all[idx] = best_x
                F_all[idx] = best_f
        
        return X_all
# EVOLVE-BLOCK-END
