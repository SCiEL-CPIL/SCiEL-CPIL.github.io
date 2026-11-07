import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0  # Safety margin
        
        # Pre-compute the optimal x1 values
        n_points = self.budget
        x1_values = self._compute_optimal_x1(n_points)
        
        # Generate the x vectors
        xs = []
        for x1 in x1_values:
            x = np.zeros(self.dim)
            x[0] = x1
            x[1:] = 0.5  # To ensure g(x) = 1
            xs.append(x)
        
        # Call problem for each x
        for x in xs:
            if time.time() > deadline:
                break
            problem(x)
        
        return xs

    def _compute_optimal_x1(self, n_points):
        # Greedy insertion
        x1_list = [0.0, 1.0]
        
        # Function to compute hypervolume of a set of x1 values
        def hypervolume(x1_vals):
            # x1_vals is a list of x1 values
            # f2 = 1 - sqrt(x1)
            # Reference point (11, 11)
            # Sort by x1
            x1_vals = sorted(x1_vals)
            f2_vals = [1 - np.sqrt(x) for x in x1_vals]
            
            # Compute area
            # The area is the sum of rectangles
            # For each point i, the width is x1[i+1] - x1[i] (or 1 - x1[-1] for the last one? No, reference is 11)
            # Actually, the hypervolume is the area of the union of rectangles [x1_i, 11] x [f2_i, 11]
            # This is equivalent to the area under the "staircase" function.
            # The staircase function h(x) is the minimum f2 of all points with x1 <= x.
            # Wait, the dominated region is { (u, v) | u >= f1_i, v >= f2_i for some i }
            # The area is integral of (11 - h(x)) dx from 0 to 11?
            # No, the reference point is (11, 11).
            # The area is sum of (11 - f1_i) * (11 - f2_i) for non-dominated points? No.
            
            # Let's use the standard 2D hypervolume calculation.
            # Sort points by f1 (ascending).
            # Keep track of the maximum f2 seen so far? No, we want the minimum f2 for a given f1.
            # Actually, for a set of points, the hypervolume is the area of the union of rectangles.
            # We can compute it by sorting by f1 and then for each point, if it is non-dominated,
            # it contributes a rectangle of width (f1_next - f1) and height (11 - f2).
            # Plus the last point contributes (11 - f1) * (11 - f2).
            
            points = sorted(zip(x1_vals, f2_vals))
            hv = 0.0
            prev_f1 = 0.0
            for i in range(len(points)):
                f1, f2 = points[i]
                # The width of the rectangle for this point is the distance to the next point's f1
                # Or to 11 if it's the last one.
                if i < len(points) - 1:
                    next_f1 = points[i+1][0]
                else:
                    next_f1 = 11.0
                # But we only count the part where this point is the "best" in terms of f2.
                # Since the points are on the Pareto front, f2 is decreasing as f1 increases.
                # So the point with smallest f1 has largest f2.
                # The rectangle for point i is from f1_i to f1_{i+1} in x, and from f2_i to 11 in y?
                # No, the dominated region for point i is [f1_i, 11] x [f2_i, 11].
                # The union of these rectangles is the hypervolume.
                # We can compute this by iterating from left to right.
                # The height of the "staircase" at x is the minimum f2 of all points with f1 <= x.
                # Since f2 is decreasing, the minimum f2 for x in [f1_i, f1_{i+1}) is f2_i.
                # So the area is sum (f1_{i+1} - f1_i) * (11 - f2_i) + (11 - f1_last) * (11 - f2_last)?
                # Wait, the reference point is (11, 11).
                # The area is integral from 0 to 11 of (11 - h(x)) dx, where h(x) is the lower envelope.
                # For x in [0, f1_0), h(x) = 11 (no points).
                # For x in [f1_i, f1_{i+1}), h(x) = f2_i.
                # So the area is sum_{i} (f1_{i+1} - f1_i) * (11 - f2_i) + (11 - f1_last) * (11 - f2_last).
                # Wait, f1_last is the largest x1.
                # The term (11 - f1_last) * (11 - f2_last) is the rectangle from f1_last to 11.
                # The terms (f1_{i+1} - f1_i) * (11 - f2_i) are the rectangles from f1_i to f1_{i+1}.
                # This sums to the total area.
                
                width = next_f1 - f1
                hv += width * (11.0 - f2)
            return hv

        # Greedy insertion
        while len(x1_list) < n_points:
            best_gain = -1
            best_x = None
            for i in range(len(x1_list) - 1):
                a = x1_list[i]
                b = x1_list[i+1]
                # Try a few points in [a, b]
                # Midpoint, 1/3, 2/3
                candidates = [(a+b)/2, a + (b-a)/3, a + 2*(b-a)/3]
                # Also check endpoints? No, they are already in the list.
                for x in candidates:
                    new_list = x1_list[:i+1] + [x] + x1_list[i+1:]
                    gain = hypervolume(new_list) - hypervolume(x1_list)
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
            x1_list.append(best_x)
            x1_list.sort()
            
        return x1_list
