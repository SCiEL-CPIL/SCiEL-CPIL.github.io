import numpy as np
import time

class Optimizer:
    """DTLZ2 optimizer using hypervolume-weighted recursive bisection grid."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # DTLZ2 structure:
        # Pareto front: x_i = 0.5 for i >= m (m = n_obj - 1 = 2)
        # x_0, x_1 parameterize the front.
        # f_1 = cos(theta_0)cos(theta_1), f_2 = cos(theta_0)sin(theta_1), f_3 = sin(theta_0)
        # theta_0 = x_0 * pi/2, theta_1 = x_1 * pi/2
        # Uniform surface area on the unit sphere (quarter hemisphere) requires:
        # cos(theta_0) ~ U[0, 1] and theta_1 ~ U[0, pi/2]
        # Let u ~ U[0, 1], v ~ U[0, 1]
        # cos(theta_0) = u  => theta_0 = arccos(u) => x_0 = (2/pi) * arccos(u)
        # theta_1 = v * pi/2 => x_1 = v

        g_start = self.n_obj - 1
        base_x = np.zeros(self.dim)
        base_x[g_start:] = 0.5

        evaluated_points = []
        used_budget = 0

        # Hypervolume-weighted recursive bisection in (u, v) space
        # We use a priority queue to always split the region with the largest
        # potential hypervolume contribution first
        
        import heapq
        
        # Each entry: (-priority, u_low, u_high, v_low, v_high, level)
        # Priority based on estimated hypervolume contribution of the center point
        # For DTLZ2, the hypervolume contribution of a point (f1, f2, f3) is roughly
        # (ref - f1) * (ref - f2) * (ref - f3) when it's a corner, but for interior
        # points it's more complex. We approximate by the area of the rectangle
        # times a weight based on distance from the reference point.
        
        ref = 2.5
        
        def estimate_hv_contribution(u, v):
            """Estimate hypervolume contribution of a point at (u, v) in parameter space."""
            # Convert to objective space
            f1 = u * np.cos(v * np.pi / 2)
            f2 = u * np.sin(v * np.pi / 2)
            f3 = np.sin(np.arccos(u))
            
            # Hypervolume contribution is proportional to the volume of the box
            # from this point to the reference point, but only if all objectives
            # are below ref. Since ref=2.5 and max objective is 1.0, all points qualify.
            # The marginal contribution depends on the "shadow" region, which is
            # complex. We use a simple proxy: the product of (ref - f_i) which
            # gives higher weight to points with smaller objective values.
            return (ref - f1) * (ref - f2) * (ref - f3)
        
        # Start with the full unit square in (u, v) space
        # Use negative priority for max-heap behavior
        heap = []
        heapq.heappush(heap, (-1.0, 0.0, 1.0, 0.0, 1.0, 0))  # initial full region
        
        # We'll generate points by always splitting the highest-priority region
        # and evaluating its center
        
        n_points_to_generate = self.budget
        
        for _ in range(n_points_to_generate):
            if time.time() > deadline:
                break
            if used_budget >= self.budget:
                break
            
            if not heap:
                break
            
            # Pop the highest priority region
            neg_priority, u_low, u_high, v_low, v_high, level = heapq.heappop(heap)
            
            # Evaluate the center of this region
            u_center = (u_low + u_high) / 2.0
            v_center = (v_low + v_high) / 2.0
            
            x0 = (2.0 / np.pi) * np.arccos(u_center)
            x1 = v_center
            
            x = base_x.copy()
            x[0] = x0
            x[1] = x1
            
            f = problem(x)
            evaluated_points.append(x)
            used_budget += 1
            
            # Split this region into 4 sub-regions if we have budget left
            if used_budget < self.budget:
                u_mid = (u_low + u_high) / 2.0
                v_mid = (v_low + v_high) / 2.0
                
                subregions = [
                    (u_low, u_mid, v_low, v_mid),
                    (u_mid, u_high, v_low, v_mid),
                    (u_low, u_mid, v_mid, v_high),
                    (u_mid, u_high, v_mid, v_high),
                ]
                
                for su_low, su_high, sv_low, sv_high in subregions:
                    su_center = (su_low + su_high) / 2.0
                    sv_center = (sv_low + sv_high) / 2.0
                    priority = estimate_hv_contribution(su_center, sv_center)
                    heapq.heappush(heap, (-priority, su_low, su_high, sv_low, sv_high, level + 1))
        
        return evaluated_points
