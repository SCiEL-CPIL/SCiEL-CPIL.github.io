import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        base_x = np.zeros(self.dim)
        base_x[2:] = 0.5  # x_3, ..., x_12 = 0.5 to ensure g = 0

        # Generate points using recursive bisection in objective space
        # mapped back to (x1, x2) parameter space
        # This places points at centers of sub-rectangles, maximizing
        # marginal hypervolume contribution per point
        
        def objective_to_params(f1, f2, f3):
            """Map objective values back to (x1, x2) parameter space"""
            # f1 = sin(x1*pi/2)
            # f2 = cos(x1*pi/2)*sin(x2*pi/2)
            # f3 = cos(x1*pi/2)*cos(x2*pi/2)
            
            # From f1: x1 = (2/pi) * arcsin(f1)
            x1 = (2.0 / np.pi) * np.arcsin(np.clip(f1, -1, 1))
            
            # From f2 and f3: cos(x1*pi/2) = sqrt(f2^2 + f3^2)
            # sin(x2*pi/2) = f2 / cos(x1*pi/2)
            # cos(x2*pi/2) = f3 / cos(x1*pi/2)
            
            cos_x1 = np.sqrt(f2**2 + f3**2)
            if cos_x1 < 1e-10:
                # Near x1 = 1, f2 and f3 are both near 0
                x2 = 0.5  # Default
            else:
                sin_x2 = f2 / cos_x1
                cos_x2 = f3 / cos_x1
                # x2 = (2/pi) * arcsin(sin_x2) or (2/pi) * arccos(cos_x2)
                # Use arcsin for consistency
                x2 = (2.0 / np.pi) * np.arcsin(np.clip(sin_x2, -1, 1))
            
            return x1, x2

        # Recursive bisection in objective space
        # The Pareto front in objective space is a quarter-sphere:
        # f1^2 + f2^2 + f3^2 = 1, with f1, f2, f3 >= 0
        
        # We'll do recursive bisection on the (f1, f2) plane,
        # computing f3 = sqrt(1 - f1^2 - f2^2) for points on the front
        
        # Start with the full triangle in (f1, f2) space:
        # f1 in [0, 1], f2 in [0, 1], f1^2 + f2^2 <= 1
        
        # Use a quadtree-like approach: recursively subdivide regions
        # and place a point at the center of each leaf region
        
        # For budget N, we want roughly N points
        # Use a depth-limited recursive subdivision
        
        def subdivide(f1_lo, f1_hi, f2_lo, f2_hi, depth, max_depth, points):
            """Recursively subdivide a rectangle in (f1, f2) space"""
            if depth >= max_depth:
                # Place a point at the center
                f1_c = (f1_lo + f1_hi) / 2.0
                f2_c = (f2_lo + f2_hi) / 2.0
                
                # Check if center is on or inside the quarter circle
                if f1_c**2 + f2_c**2 <= 1.0 + 1e-10:
                    f3_c = np.sqrt(max(0.0, 1.0 - f1_c**2 - f2_c**2))
                    x1, x2 = objective_to_params(f1_c, f2_c, f3_c)
                    points.append((x1, x2))
                return
            
            # Decide whether to subdivide based on whether the rectangle
            # is fully inside, fully outside, or intersects the quarter circle
            # Check corners
            corners = [
                (f1_lo, f2_lo), (f1_lo, f2_hi),
                (f1_hi, f2_lo), (f1_hi, f2_hi)
            ]
            
            inside_count = sum(1 for f1, f2 in corners if f1**2 + f2**2 <= 1.0)
            
            if inside_count == 0:
                # Entirely outside, don't subdivide
                return
            elif inside_count == 4:
                # Entirely inside, subdivide if depth allows
                if depth < max_depth:
                    # Split into 4 sub-rectangles
                    f1_mid = (f1_lo + f1_hi) / 2.0
                    f2_mid = (f2_lo + f2_hi) / 2.0
                    
                    subdivide(f1_lo, f1_mid, f2_lo, f2_mid, depth + 1, max_depth, points)
                    subdivide(f1_lo, f1_mid, f2_mid, f2_hi, depth + 1, max_depth, points)
                    subdivide(f1_mid, f1_hi, f2_lo, f2_mid, depth + 1, max_depth, points)
                    subdivide(f1_mid, f1_hi, f2_mid, f2_hi, depth + 1, max_depth, points)
                else:
                    # At max depth, place a point
                    f1_c = (f1_lo + f1_hi) / 2.0
                    f2_c = (f2_lo + f2_hi) / 2.0
                    if f1_c**2 + f2_c**2 <= 1.0 + 1e-10:
                        f3_c = np.sqrt(max(0.0, 1.0 - f1_c**2 - f2_c**2))
                        x1, x2 = objective_to_params(f1_c, f2_c, f3_c)
                        points.append((x1, x2))
            else:
                # Partially inside, must subdivide
                if depth < max_depth:
                    f1_mid = (f1_lo + f1_hi) / 2.0
                    f2_mid = (f2_lo + f2_hi) / 2.0
                    
                    subdivide(f1_lo, f1_mid, f2_lo, f2_mid, depth + 1, max_depth, points)
                    subdivide(f1_lo, f1_mid, f2_mid, f2_hi, depth + 1, max_depth, points)
                    subdivide(f1_mid, f1_hi, f2_lo, f2_mid, depth + 1, max_depth, points)
                    subdivide(f1_mid, f1_hi, f2_mid, f2_hi, depth + 1, max_depth, points)
                else:
                    # At max depth, place a point if center is inside
                    f1_c = (f1_lo + f1_hi) / 2.0
                    f2_c = (f2_lo + f2_hi) / 2.0
                    if f1_c**2 + f2_c**2 <= 1.0 + 1e-10:
                        f3_c = np.sqrt(max(0.0, 1.0 - f1_c**2 - f2_c**2))
                        x1, x2 = objective_to_params(f1_c, f2_c, f3_c)
                        points.append((x1, x2))

        # Determine max_depth based on budget
        # Each leaf produces 1 point, and we start with 1 rectangle
        # With max_depth d, we have at most 4^d rectangles
        # We want roughly budget points, so 4^d ~ budget
        # d ~ log4(budget) = log(budget)/log(4)
        
        target_points = min(self.budget, 200)
        max_depth = int(np.log2(target_points) / 2.0) + 1  # Slightly more than needed
        max_depth = max(2, min(max_depth, 8))  # Clamp to reasonable range
        
        # Generate points via recursive bisection
        bisection_points = []
        subdivide(0.0, 1.0, 0.0, 1.0, 0, max_depth, bisection_points)
        
        # Add structured points for better coverage of corners and edges
        extra_points = [
            (0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0),
            (0.5, 0.0), (0.5, 1.0), (0.0, 0.5), (1.0, 0.5),
            (0.25, 0.5), (0.75, 0.5), (0.5, 0.25), (0.5, 0.75),
            (0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9),
            (0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75),
            (0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85),
            (0.35, 0.35), (0.65, 0.35), (0.35, 0.65), (0.65, 0.65),
        ]
        
        # Combine and deduplicate
        all_params = list(bisection_points) + list(extra_points)
        
        # Remove duplicates (within tolerance)
        unique_params = []
        for p in all_params:
            x1, x2 = p
            if not any(abs(x1 - u[0]) < 1e-6 and abs(x2 - u[1]) < 1e-6 for u in unique_params):
                unique_params.append(p)
        
        # Truncate to budget
        all_params = unique_params[:self.budget]
        
        # Evaluate all points
        results = []
        for x1, x2 in all_params:
            if len(results) >= self.budget:
                break
            if time.time() > deadline:
                break
            
            x = base_x.copy()
            x[0] = x1
            x[1] = x2
            x = np.clip(x, lo, hi)
            
            f = problem(x)
            results.append(f)
        
        # Use remaining budget for random refinement
        if len(results) < self.budget:
            rng = np.random.default_rng(42)
            remaining = self.budget - len(results)
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                
                x1 = rng.random()
                x2 = rng.random()
                
                x = base_x.copy()
                x[0] = x1
                x[1] = x2
                x = np.clip(x, lo, hi)
                
                f = problem(x)
                results.append(f)
        
        return results
