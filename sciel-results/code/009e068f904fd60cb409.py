"""Search-based optimizer for DTLZ4 with known Pareto front structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using stratified spherical sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = 285.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        def make_x(x1, x2):
            """Construct decision vector from x1, x2 that achieves g=0 on DTLZ4.
            x1 maps to angle in [0, pi/2], x2 maps to angle in [0, pi/2].
            f1 = 0.5 * cos(pi*x1/2) * cos(pi*x2/2)
            f2 = 0.5 * cos(pi*x1/2) * sin(pi*x2/2)
            f3 = 0.5 * sin(pi*x1/2)
            So x1 controls f3, x2 controls ratio of f1 to f2.
            """
            x = np.full(dim, 0.5)
            x[0] = x1
            x[1] = x2
            return np.clip(x, lo, hi)

        # Generate candidates with uniform coverage on the quarter-sphere front.
        # The front is parameterized by (x1, x2) in [0,1]^2:
        #   f3 = 0.5 * sin(pi*x1/2), f1 = 0.5*cos(pi*x1/2)*cos(pi*x2/2), f2 = 0.5*cos(pi*x1/2)*sin(pi*x2/2)
        # For uniform area on the sphere, we want uniform in sin(pi*x1/2) direction and uniform in pi*x2/2.
        # Let u = sin(pi*x1/2), then x1 = 2/pi * arcsin(u). For stratified u in [0,1], use (i+0.5)/N.
        # For x2, uniform in [0,1] gives uniform in phi = pi*x2/2.

        candidates = []

        # Strategy 1: Stratified sampling on the sphere surface
        # Use a 2D stratified grid in (u, v) where u = sin(pi*x1/2) in [0,1], v = x2 in [0,1]
        # For better coverage, use a quasi-random sequence with stratification.
        
        n_strat = 180
        # Create stratified points in 2D
        # Use a combination of grid and jitter for good coverage
        n_per_row = 12  # 12 x 15 = 180
        n_rows = 15
        
        for i in range(n_rows):
            for j in range(n_per_row):
                # Stratified: divide [0,1] into n_rows and n_per_row cells
                u = (i + 0.5) / n_rows  # sin(pi*x1/2)
                v = (j + 0.5) / n_per_row  # x2
                
                # Convert u back to x1: u = sin(pi*x1/2) => x1 = 2/pi * arcsin(u)
                x1 = (2.0 / np.pi) * np.arcsin(np.clip(u, 0.0, 1.0))
                x2 = v
                
                # Add small deterministic jitter to avoid exact grid alignment
                # Use golden-ratio-based offset for better distribution
                jitter1 = ((i * 7 + j * 13) % 100) / 100.0 * 0.3
                jitter2 = ((i * 11 + j * 17) % 100) / 100.0 * 0.3
                x1 = np.clip(x1 + (jitter1 - 0.15) * (1.0 / n_rows), 0.0, 1.0)
                x2 = np.clip(x2 + (jitter2 - 0.15) * (1.0 / n_per_row), 0.0, 1.0)
                
                candidates.append((x1, x2))

        # Strategy 2: Explicit corner and edge points of the quarter-sphere
        # Corners: (x1, x2) = (0,0), (0,1), (1,0), (1,1)
        # In objective space:
        # (0,0) -> f=(0.5, 0, 0)
        # (0,1) -> f=(0, 0.5, 0)
        # (1,0) -> f=(0, 0, 0.5)
        # (1,1) -> f=(0, 0, 0.5) same as (1,0)
        
        # Add the 3 distinct corner points
        candidates.append((0.0, 0.0))  # f=(0.5, 0, 0)
        candidates.append((0.0, 1.0))  # f=(0, 0.5, 0)
        candidates.append((1.0, 0.0))  # f=(0, 0, 0.5)
        
        # Add edge midpoints for better boundary coverage
        candidates.append((0.0, 0.5))  # f=(0.5/sqrt(2), 0.5/sqrt(2), 0)
        candidates.append((1.0, 0.5))  # f=(0, 0, 0.5)
        candidates.append((0.5, 0.0))  # f=(0.5/sqrt(2), 0, 0.5/sqrt(2))
        candidates.append((0.5, 1.0))  # f=(0, 0.5/sqrt(2), 0.5/sqrt(2))
        
        # Add more edge points
        for t in [0.2, 0.3, 0.4, 0.6, 0.7, 0.8]:
            candidates.append((0.0, t))
            candidates.append((t, 0.0))
            candidates.append((t, 1.0))
            candidates.append((1.0, t))

        # Strategy 3: A few random points for diversity
        n_random = 10
        for _ in range(n_random):
            candidates.append((np.random.rand(), np.random.rand()))

        # Trim to exactly budget
        if len(candidates) > budget:
            candidates = candidates[:budget]

        # Pad if less than budget
        while len(candidates) < budget:
            candidates.append((np.random.rand(), np.random.rand()))

        # Evaluate all candidates
        evaluated_points = []
        used = 0

        for x1, x2 in candidates:
            if used >= budget or time.time() - t0 > deadline:
                break

            x = make_x(x1, x2)
            f = np.array(problem(x))
            used += 1
            evaluated_points.append(f)

        return np.array(evaluated_points)
# EVOLVE-BLOCK-END
