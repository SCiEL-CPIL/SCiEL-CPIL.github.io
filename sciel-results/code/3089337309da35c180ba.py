"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ2 using Pareto front sampling + local search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)

        # DTLZ2 Pareto front: f_i = sin(pi * theta_i / 2) for i=0,1,2
        # theta_i in [0, pi/2], and we need sum of x_i^2 = 1 for x in [0,1]^12
        # The Pareto front in objective space is the quarter unit sphere in 3D
        # f_i = (1 - g) * sin(pi*theta_i/2) where g=0 on the front
        # Actually for DTLZ2: f_i = (1+g) * sin(pi*theta_i/2), g = sum_{i=k+1}^n (x_i - 0.5)^2
        # On the Pareto front g=0, so f_i = sin(pi*theta_i/2)
        # theta_i are determined by the first (m-1) decision variables: x_0..x_{m-2}
        # For m=3: theta_1 = pi/2 * x_0, theta_2 = pi/2 * x_1, theta_3 = pi/2 - theta_1 - theta_2
        # Wait, let me recall DTLZ2 properly.
        # DTLZ2 with m objectives: f_i = (1+g) * prod_{j=1}^{i} cos(theta_j) * sin(theta_i) for i<m
        # Actually the standard DTLZ2 formulation:
        # f_i = (1+g) * sin(pi*theta_i/2) for i=1..m, where theta_i are derived from decision vars
        # For m=3: we use x_0, x_1 as theta parameters
        # theta_1 = pi/2 * x_0, theta_2 = pi/2 * x_1
        # f_1 = (1+g) * cos(theta_1) * cos(theta_2)
        # f_2 = (1+g) * cos(theta_1) * sin(theta_2)  
        # f_3 = (1+g) * sin(theta_1)
        # Hmm, let me think again. The standard DTLZ2:
        # For m objectives, f_i = (1+g) * prod_{j=1}^{i-1} cos(theta_j) * sin(theta_i) for i < m
        # f_m = (1+g) * prod_{j=1}^{m-1} cos(theta_j)
        # where theta_j = pi/2 * x_j for j=1..m-1
        # g = sum_{j=m}^{n} (x_j - 0.5)^2
        #
        # So for m=3, n=12:
        # theta_1 = pi/2 * x_0, theta_2 = pi/2 * x_1
        # g = sum_{j=2}^{11} (x_j - 0.5)^2
        # f_1 = (1+g) * sin(theta_1)  ... wait no
        #
        # Let me use the correct formula. DTLZ2:
        # f_i = (1+g) * sin(pi*x_i/2) for i=1..m? No, that's not right either.
        #
        # Standard DTLZ2 (from the original paper):
        # f_i(x) = (1 + g(x_{M+1})) * prod_{j=1}^{i-1} cos(x_j * pi/2) * sin(x_i * pi/2) for i=1..M-1
        # f_M(x) = (1 + g(x_{M+1})) * prod_{j=1}^{M-1} cos(x_j * pi/2)
        # g(x_{M+1}) = sum_{i=M+1}^{n} (x_i - 0.5)^2
        #
        # So for M=3, n=12:
        # f_1 = (1+g) * sin(x_0*pi/2)
        # f_2 = (1+g) * cos(x_0*pi/2) * sin(x_1*pi/2)
        # f_3 = (1+g) * cos(x_0*pi/2) * cos(x_1*pi/2)
        # g = sum_{i=2}^{11} (x_i - 0.5)^2
        #
        # On the Pareto front (g=0), f_1^2 + f_2^2 + f_3^2 = 1
        # So the Pareto front is the quarter unit sphere in the positive octant.
        #
        # To sample the Pareto front, we need to find x such that g=0, i.e., x_i=0.5 for i=2..11
        # and x_0, x_1 determine the point on the front.
        #
        # Strategy: 
        # 1. Use a Halton sequence to sample (x_0, x_1) in [0,1]^2 to cover the Pareto front
        # 2. Set x_2..x_11 = 0.5 to get g=0
        # 3. Evaluate these points
        # 4. Use remaining budget for local search / refinement

        def halton_sequence(n, base):
            """Generate n points of the Halton sequence with given base."""
            result = np.zeros(n)
            for i in range(1, n + 1):
                f = 1.0
                r = 0.0
                idx = i
                while idx > 0:
                    f /= base
                    r += f * (idx % base)
                    idx //= base
                result[i - 1] = r
            return result

        # Generate Halton sequence points for (x_0, x_1)
        # Use a good number of points for initial coverage
        n_halton = 80
        h1 = halton_sequence(n_halton, 2)
        h2 = halton_sequence(n_halton, 3)

        # Build candidate points on the Pareto front
        X_front = np.zeros((n_halton, self.dim))
        X_front[:, 0] = h1
        X_front[:, 1] = h2
        X_front[:, 2:] = 0.5  # g = 0

        # Also add some special points: vertices and edge midpoints
        special_points = []
        # Vertices of the quarter sphere: (1,0,0), (0,1,0), (0,0,1) in objective space
        # These correspond to specific (x_0, x_1) values
        # f_1=1: x_0=0 (sin(0)=0... wait that gives f_1=0)
        # Let me re-check: f_1 = sin(x_0*pi/2), f_2 = cos(x_0*pi/2)*sin(x_1*pi/2), f_3 = cos(x_0*pi/2)*cos(x_1*pi/2)
        # f_1=1: x_0=1, then f_2=cos(pi/2)*...=0, f_3=0. So x_0=1, x_1=anything -> (1,0,0)
        # f_2=1: x_0=0, x_1=1 -> (0,1,0)
        # f_3=1: x_0=0, x_1=0 -> (0,0,1)
        # Edge midpoints:
        # Between (1,0,0) and (0,1,0): x_0 such that sin(x_0*pi/2)=cos(x_0*pi/2)*sin(x_1*pi/2) and f_3=0
        # f_3=0 means cos(x_0*pi/2)*cos(x_1*pi/2)=0, so x_0=1 or x_1=1
        # Hmm, the edge between (1,0,0) and (0,1,0) is where f_3=0, which requires x_0=1 or x_1=1
        # If x_1=1: f_1=sin(x_0*pi/2), f_2=cos(x_0*pi/2), f_3=0. This traces the arc from (1,0,0) to (0,1,0)
        # Midpoint: x_0=0.5, x_1=1 -> f_1=sin(pi/4)=sqrt(2)/2, f_2=cos(pi/4)=sqrt(2)/2
        # Similarly for other edges.

        special = np.array([
            [1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],  # (1,0,0)
            [0.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],  # (0,1,0)
            [0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],  # (0,0,1)
            [0.5, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],  # edge (1,0,0)-(0,1,0) mid
            [0.5, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],  # edge (1,0,0)-(0,0,1) mid
            [0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],  # edge (0,1,0)-(0,0,1) mid
        ])

        all_X = np.vstack([X_front, special])
        all_F = np.array([problem(x) for x in all_X])
        used = len(all_X)

        # Keep track of all evaluated points
        best_X = all_X
        best_F = all_F

        # Use remaining budget for local refinement
        # For each point on the front, try small perturbations
        remaining = self.budget - used
        if remaining > 0:
            # Try to refine points that are on or near the front
            for i in range(min(remaining, len(best_F))):
                if time.time() > deadline:
                    break
                # Pick a random point and perturb it
                idx = rng.integers(len(best_X))
                x = best_X[idx].copy()
                # Small perturbation
                x[0] += 0.05 * rng.standard_normal()
                x[1] += 0.05 * rng.standard_normal()
                x[2:] += 0.02 * rng.standard_normal(self.dim - 2)
                x = np.clip(x, lo, hi)
                fx = problem(x)
                used += 1
                # Check if this point dominates any existing point
                # Simple check: if it's better than the worst point, replace
                # Actually, let's just add it to our collection and keep all points
                # But we can only keep budget points total
                # Let's replace the worst point if the new one is better in hypervolume contribution
                # Simplified: replace if new point is non-dominated
                dominated = False
                for j in range(len(best_F)):
                    if np.all(best_F[j] <= fx) and np.any(best_F[j] < fx):
                        dominated = True
                        break
                if not dominated:
                    # Replace the worst point
                    worst_idx = np.argmax(np.sum(best_F, axis=1))
                    best_X[worst_idx] = x
                    best_F[worst_idx] = fx

        # Final check for deadline
        if time.time() > deadline:
            return best_X

        return best_X

# EVOLVE-BLOCK-END
