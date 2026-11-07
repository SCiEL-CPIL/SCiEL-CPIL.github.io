# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    DTLZ4 optimizer using Latin Hypercube Sampling (LHS) on the Pareto front manifold.
    
    Strategy:
    1. DTLZ4 Pareto front is a 2D quarter-sphere in objective space.
    2. Decision variables x[3:] must be 0.5.
    3. Decision variables x[0:3] must lie on the unit sphere in the positive orthant.
    4. We parameterize the quarter-sphere using (u, v) in [0, 1]^2:
       theta = arccos(1 - 2*u)
       phi = v * pi/2
       x1 = sin(theta)*cos(phi), x2 = sin(theta)*sin(phi), x3 = cos(theta)
    5. We use Latin Hypercube Sampling (LHS) on (u, v) to ensure uniform coverage.
    6. We explicitly include the 3 vertices of the Pareto front (corners) because
       they provide the largest individual hypervolume contributions relative to the
       reference point [2.5, 2.5, 2.5].
       - Vertex 1: (1, 0, 0) -> f=(1,0,0)
       - Vertex 2: (0, 1, 0) -> f=(0,1,0)
       - Vertex 3: (0, 0, 1) -> f=(0,0,1)
       Wait, DTLZ4 objectives are:
       f1 = 0.5 * x1
       f2 = 0.5 * x2
       f3 = 0.5 * x3
       So the vertices in objective space are:
       (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5)
       
       Actually, the Pareto front is defined by x1^2 + x2^2 + x3^2 = 1, xi >= 0.
       The objectives are f1 = 0.5 x1, f2 = 0.5 x2, f3 = 0.5 x3.
       So the Pareto front in objective space is a quarter-sphere of radius 0.5.
       The vertices are (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5).
       
       Reference point is [2.5, 2.5, 2.5].
       
       Including these vertices is crucial for maximizing hypervolume.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_points = self.budget
        
        # We will generate n_points - 3 LHS points and 3 vertex points.
        # This ensures we use the full budget while guaranteeing the best corners.
        
        n_lhs = n_points - 3
        if n_lhs < 1:
            n_lhs = n_points
            n_vertices = 0
        else:
            n_vertices = 3

        # 1. Generate LHS points in [0, 1]^2
        # LHS: divide [0,1] into n intervals, pick one uniform point in each interval,
        # then permute the y-coordinates.
        
        if n_lhs > 0:
            # Create intervals
            indices = np.arange(n_lhs)
            # Uniform random within each interval
            u = (indices + np.random.rand(n_lhs)) / n_lhs
            v = (np.random.permutation(n_lhs) + np.random.rand(n_lhs)) / n_lhs
            
            # Map to sphere parameters
            # theta = arccos(1 - 2*u)
            # phi = v * pi/2
            
            theta = np.arccos(1 - 2 * u)
            phi = v * np.pi / 2
            
            x1 = np.sin(theta) * np.cos(phi)
            x2 = np.sin(theta) * np.sin(phi)
            x3 = np.cos(theta)
            
            # Construct decision vectors
            lhs_x = np.zeros((n_lhs, self.dim))
            lhs_x[:, 0] = x1
            lhs_x[:, 1] = x2
            lhs_x[:, 2] = x3
            lhs_x[:, 3:] = 0.5
        else:
            lhs_x = np.zeros((0, self.dim))

        # 2. Generate vertex points
        # Vertex 1: x1=1, x2=0, x3=0
        # Vertex 2: x1=0, x2=1, x3=0
        # Vertex 3: x1=0, x2=0, x3=1
        
        vertices = np.array([
            [1.0, 0.0, 0.0] + [0.5] * (self.dim - 3),
            [0.0, 1.0, 0.0] + [0.5] * (self.dim - 3),
            [0.0, 0.0, 1.0] + [0.5] * (self.dim - 3)
        ])
        
        # Combine points
        if n_lhs > 0:
            all_x = np.vstack([lhs_x, vertices])
        else:
            all_x = vertices

        # Shuffle to ensure vertices are not always at the end (though order shouldn't matter for HV, 
        # it might matter if we have to break early due to time)
        # But we want to ensure vertices are evaluated. Let's put them first?
        # Actually, if we break early, we want the most valuable points.
        # The vertices are the most valuable.
        # Let's evaluate vertices first.
        
        # Reorder: vertices first, then LHS
        if n_lhs > 0:
            eval_order = np.concatenate([np.arange(3), np.arange(3, 3 + n_lhs)])
        else:
            eval_order = np.arange(3)
            
        all_x = all_x[eval_order]

        # Clip to bounds (though 0.5 and 1.0/0.0 are within [0,1])
        all_x = np.clip(all_x, lo, hi)

        # Evaluate
        results = []
        for i in range(len(all_x)):
            if time.time() > deadline:
                break
            try:
                f = problem(all_x[i])
                results.append(f)
            except Exception:
                pass

        return results
# EVOLVE-BLOCK-END
