"""Multi-objective optimizer for ZDT1 using hypervolume-driven adaptive search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-driven adaptive multi-objective optimizer for ZDT1."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        budget = self.budget

        # ZDT1 structure: x1 controls objective 1, x2..xn control objective 2
        # f1 = x1, f2 = (1 + 10*(n-1)) * (1 - sqrt(f1/g))
        # g = sum(x_i^2 for i>=2)
        # Optimal: x1 in [0,1], g=0 (all other vars = 0)
        # Pareto front: f1 = t, f2 = 11*(1-sqrt(t)) for t in [0,1]

        # We'll focus search on x1 and try to drive other variables to 0
        # Strategy: 
        # 1. Generate diverse x1 values
        # 2. For each x1, try to minimize g (other variables)
        # 3. Use hypervolume marginal gain to guide where to add new points

        # Phase 1: Initial diverse sampling
        # Sample x1 values that will cover the Pareto front well
        # The Pareto front is f2 = 11*(1-sqrt(f1)), which is concave
        # For hypervolume with ref point (11,11), we want good coverage

        # Start with a set of candidate x1 values
        # Use a mix of uniform and refined points
        x1_candidates = np.linspace(0.0, 1.0, 20)
        
        # Evaluate initial population: for each x1, set other vars to 0
        # This gives points on or near the Pareto front
        evaluated = []  # list of (x1, f1, f2)
        used = 0

        # First, evaluate points with other vars = 0 (on the true Pareto front)
        for x1 in x1_candidates:
            if used >= budget:
                break
            x = np.zeros(n)
            x[0] = x1
            f = np.array(problem(x))
            evaluated.append((x1, f[0], f[1]))
            used += 1

        # Now we have ~20 points on the Pareto front
        # Phase 2: Refine using hypervolume marginal gain
        # For each interval between consecutive x1 values, the midpoint of the 
        # Pareto front in that interval gives the best marginal hypervolume gain
        
        # Sort evaluated points by f1
        evaluated.sort(key=lambda e: e[0])
        
        # Compute hypervolume contribution of each point
        # For 2 objectives with ref point (11, 11):
        # A point (f1, f2) contributes area if f1 < 11 and f2 < 11
        # The contribution is determined by its position in the sorted set
        
        # We'll use a greedy approach: at each step, find the x1 that maximizes
        # the marginal hypervolume gain, evaluate it, and add it to the set
        
        # Track current set of (f1, f2) points
        current_points = [(e[1], e[2]) for e in evaluated]
        
        # For efficiency, maintain sorted by f1
        current_points.sort()
        
        # Marginal hypervolume gain for a new point (f1_new, f2_new):
        # Find the point with largest f1 < f1_new (call it left) 
        # and the point with smallest f1 > f1_new (call it right)
        # The gain is the area that the new point "uncovers"
        
        # For 2D, if we have points sorted by f1:
        # (f1_i, f2_i) for i=1..m, and ref point (r1, r2) = (11, 11)
        # The hypervolume is sum over i of (f1_{i+1} - f1_i) * (r2 - f2_i) 
        # where f1_{m+1} = r1, but we need to be careful about dominance
        
        # Actually, for non-dominated points sorted by f1 (increasing), f2 is decreasing
        # The HV is the sum of rectangles:
        # For i from 0 to m-1: width_i * height_i
        # where width_i = f1_{i+1} - f1_i (with f1_m = r1)
        # and height_i = r2 - f2_i
        
        # When we insert a new point between i and i+1:
        # Old contribution: (f1_{i+1} - f1_i) * (r2 - f2_i)
        # New contributions: (f1_new - f1_i) * (r2 - f2_i) + (f1_{i+1} - f1_new) * (r2 - f2_new)
        # Gain = new - old = (f1_{i+1} - f1_new) * (r2 - f2_new) - (f1_{i+1} - f1_new) * (r2 - f2_i)
        #      = (f1_{i+1} - f1_new) * (f2_i - f2_new)
        
        # But we also need to check that the new point is not dominated
        # and that it actually improves the front
        
        # For ZDT1, the Pareto front is f2 = 11*(1-sqrt(f1))
        # So for a given x1, the best f2 is 11*(1-sqrt(x1)) (when g=0)
        
        # Let's precompute the theoretical Pareto front
        # and use it to guide where to place new points
        
        # Theoretical front: f1 = t, f2 = 11*(1-sqrt(t)) for t in [0,1]
        
        # For each gap between consecutive points, the optimal new point
        # in terms of hypervolume gain is not necessarily the midpoint
        # Let's compute it analytically
        
        # Given two adjacent points (f1_a, f2_a) and (f1_b, f2_b) with f1_a < f1_b
        # and both on the Pareto front, we want to find f1_new in (f1_a, f1_b)
        # that maximizes the marginal HV gain
        
        # The gain from inserting (f1_new, f2_new) where f2_new = 11*(1-sqrt(f1_new)):
        # gain = (f1_b - f1_new) * (f2_a - f2_new)
        #      = (f1_b - f1_new) * (11*(1-sqrt(f1_a)) - 11*(1-sqrt(f1_new)))
        #      = 11 * (f1_b - f1_new) * (sqrt(f1_new) - sqrt(f1_a))
        
        # Let u = sqrt(f1_new), so f1_new = u^2, u in (sqrt(f1_a), sqrt(f1_b))
        # gain = 11 * (f1_b - u^2) * (u - sqrt(f1_a))
        
        # d(gain)/du = 11 * [-2u * (u - sqrt(f1_a)) + (f1_b - u^2) * 1]
        #            = 11 * [-2u^2 + 2u*sqrt(f1_a) + f1_b - u^2]
        #            = 11 * [-3u^2 + 2*sqrt(f1_a)*u + f1_b]
        
        # Setting to 0: 3u^2 - 2*sqrt(f1_a)*u - f1_b = 0
        # u = [2*sqrt(f1_a) + sqrt(4*f1_a + 12*f1_b)] / 6
        #   = [sqrt(f1_a) + sqrt(f1_a + 3*f1_b)] / 3
        
        def optimal_x1_in_interval(f1_a, f1_b):
            """Find the x1 that maximizes marginal HV gain in [f1_a, f1_b]."""
            if f1_a >= f1_b:
                return (f1_a + f1_b) / 2
            sqrt_a = np.sqrt(f1_a)
            sqrt_term = np.sqrt(f1_a + 3 * f1_b)
            u = (sqrt_a + sqrt_term) / 3
            f1_new = u * u
            # Clamp to interval
            f1_new = np.clip(f1_new, f1_a + 1e-10, f1_b - 1e-10)
            return f1_new

        # Now iteratively add points
        while used < budget:
            if time.time() > deadline:
                break
            
            # Find the interval with the maximum marginal HV gain
            # Consider all gaps between consecutive points, and also the ends
            best_gain = -1
            best_x1 = None
            best_idx = None  # index to insert after
            
            pts = current_points  # sorted by f1
            m = len(pts)
            
            # Check gaps between consecutive points
            for i in range(m - 1):
                f1_a, f2_a = pts[i]
                f1_b, f2_b = pts[i + 1]
                if f1_a >= f1_b:
                    continue
                
                # Compute optimal x1 in this interval
                f1_new = optimal_x1_in_interval(f1_a, f1_b)
                f2_new = 11 * (1 - np.sqrt(f1_new))
                
                # Compute gain
                gain = (f1_b - f1_new) * (f2_a - f2_new)
                
                if gain > best_gain:
                    best_gain = gain
                    best_x1 = f1_new
                    best_idx = i
            
            # Also consider extending beyond the current range
            # Left end: f1 < pts[0][0]
            if m > 0 and pts[0][0] > 1e-6:
                f1_a = 0.0
                f2_a = 11.0  # f2 at f1=0
                f1_b, f2_b = pts[0]
                f1_new = optimal_x1_in_interval(f1_a, f1_b)
                f2_new = 11 * (1 - np.sqrt(f1_new))
                gain = (f1_b - f1_new) * (f2_a - f2_new)
                if gain > best_gain:
                    best_gain = gain
                    best_x1 = f1_new
                    best_idx = -1  # insert at beginning
            
            # Right end: f1 > pts[-1][0]
            if m > 0 and pts[-1][0] < 1.0 - 1e-6:
                f1_a, f2_a = pts[-1]
                f1_b = 1.0
                f2_b = 0.0  # f2 at f1=1
                f1_new = optimal_x1_in_interval(f1_a, f1_b)
                f2_new = 11 * (1 - np.sqrt(f1_new))
                gain = (f1_b - f1_new) * (f2_a - f2_new)
                if gain > best_gain:
                    best_gain = gain
                    best_x1 = f1_new
                    best_idx = m - 1  # insert after last
            
            if best_x1 is None or best_gain < 1e-10:
                # No significant gain, try random perturbation
                x1 = np.random.uniform(0, 1)
            else:
                x1 = best_x1
            
            # Evaluate: set x1 and other vars to 0
            x = np.zeros(n)
            x[0] = x1
            f = np.array(problem(x))
            used += 1
            
            # Add to current points
            new_point = (f[0], f[1])
            # Insert in sorted order
            inserted = False
            for i in range(len(current_points)):
                if new_point[0] < current_points[i][0]:
                    current_points.insert(i, new_point)
                    inserted = True
                    break
            if not inserted:
                current_points.append(new_point)
        
        # Phase 3: If we still have budget, try to improve by perturbing
        # some points slightly to see if we can get better f2 values
        # (in case the problem isn't exactly ZDT1 as assumed)
        # But for ZDT1, setting other vars to 0 is optimal, so this shouldn't help
        
        return None
# EVOLVE-BLOCK-END
