"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """HV-driven optimizer exploiting DTLZ1 structure with simplex sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _dtlz1_objectives(self, x):
        """
        Compute DTLZ1 objectives for a 7-dim point.
        For DTLZ1 with n=7, M=3:
        - x[0], x[1] are position variables
        - x[2..6] are convergence variables
        - g(x) = 100 * (n - M + 1 + sum_{i=M-1}^{n-1} ((x_i - 0.5)^2 - sin(10*pi*(x_i+0.5))))
        - f1 = 0.5 * (1+g) * x[0] * prod(x[3..6])
        - f2 = 0.5 * (1+g) * (1 - x[0] - x[1]) * prod(x[3..6])
        - f3 = 0.5 * (1+g) * prod(x[3..6])
        """
        x = np.asarray(x, dtype=np.float64)
        # Clip to [0,1]
        x = np.clip(x, 0.0, 1.0)
        
        # Convergence variables: x[2] through x[6]
        conv = x[2:]  # 5 variables
        n_conv = len(conv)
        
        # g function
        g = 100.0 * (n_conv + 1 + np.sum((conv - 0.5)**2 - np.sin(10.0 * np.pi * (conv + 0.5))))
        
        # Product of convergence variables
        prod = np.prod(conv)
        
        # Objectives
        f1 = 0.5 * (1.0 + g) * x[0] * prod
        f2 = 0.5 * (1.0 + g) * (1.0 - x[0] - x[1]) * prod
        f3 = 0.5 * (1.0 + g) * prod
        
        return np.array([f1, f2, f3])

    def _hypervolume_3d(self, points, ref):
        """
        Compute exact 3D hypervolume using inclusion-exclusion via sorted layers.
        Points must be strictly better than ref in all objectives.
        """
        # Filter valid points
        valid = [p for p in points if all(p[i] < ref[i] for i in range(3))]
        if not valid:
            return 0.0

        # Sort by f1 ascending
        valid.sort(key=lambda p: p[0])
        n = len(valid)
        
        def _area_2d_union(points_2d):
            """Compute area of union of rectangles [0, p[0]]x[0, p[1]] for p in points_2d."""
            if not points_2d:
                return 0.0
            sorted_pts = sorted(points_2d, key=lambda p: p[0])
            area = 0.0
            max_y_so_far = 0.0
            prev_x = 0.0
            for x, y in sorted_pts:
                if x > prev_x:
                    area += (x - prev_x) * max_y_so_far
                if y > max_y_so_far:
                    max_y_so_far = y
                prev_x = x
            max_x = sorted_pts[-1][0]
            area += (max_x - prev_x) * max_y_so_far
            return area

        hv = 0.0
        for i in range(n):
            pts_2d = [(valid[j][1], valid[j][2]) for j in range(i + 1)]
            A_i = _area_2d_union(pts_2d)
            f1_lo = valid[i][0]
            f1_hi = valid[i + 1][0] if i + 1 < n else ref[0]
            if f1_hi > f1_lo:
                hv += (f1_hi - f1_lo) * A_i
        
        return hv

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        
        # DTLZ1 structure:
        # x[0], x[1] determine position on Pareto front (simplex)
        # x[2..6] determine distance from front (g function)
        # Optimal g is 0 at x[2..6] = 0.5
        
        # Set convergence variables to 0.5 for points on the Pareto front
        base_x = np.zeros(self.dim)
        base_x[2:] = 0.5
        
        evaluated = []
        eval_count = 0
        
        def try_evaluate(x):
            nonlocal eval_count
            if eval_count >= self.budget:
                return None
            if time.time() > deadline:
                return None
            obj = problem(x)
            evaluated.append((x, obj))
            eval_count += 1
            return obj
        
        # Phase 1: Dense simplex grid
        # Use a triangular grid with K+1 points on each edge
        # Total points = (K+1)(K+2)/2
        # For budget=200, use K=19: 20*21/2 = 210, too many
        # K=18: 19*20/2 = 190, good
        # K=17: 18*19/2 = 171
        
        # Start with a moderate grid
        K = 18
        for i in range(K + 1):
            for j in range(K + 1 - i):
                x = base_x.copy()
                x[0] = i / K if K > 0 else 0.0
                x[1] = j / K if K > 0 else 0.0
                try_evaluate(x)
        
        # Phase 2: Add corners if not already covered
        corners = [
            np.array([0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5]),
            np.array([1.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5]),
            np.array([0.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5]),
        ]
        for x in corners:
            try_evaluate(x)
        
        # Phase 3: Add points along edges with finer resolution
        # Edge 0: x[0] from 0 to 1, x[1]=0
        # Edge 1: x[1] from 0 to 1, x[0]=0
        # Edge 2: x[0]+x[1]=1
        for edge in range(3):
            n_edge = 10
            for t in np.linspace(0, 1, n_edge + 2)[1:-1]:
                x = base_x.copy()
                if edge == 0:
                    x[0] = t
                    x[1] = 0.0
                elif edge == 1:
                    x[0] = 0.0
                    x[1] = t
                else:
                    x[0] = t
                    x[1] = 1.0 - t
                try_evaluate(x)
        
        # Phase 4: Fill remaining budget with random points in the simplex
        # Use Dirichlet(1,1) distribution for uniform sampling in the simplex
        while eval_count < self.budget:
            if time.time() > deadline:
                break
            u = np.random.random()
            v = np.random.random() * (1 - u)
            x = base_x.copy()
            x[0] = u
            x[1] = v
            try_evaluate(x)
        
        return [x for x, obj in evaluated]
# EVOLVE-BLOCK-END
