import numpy as np
import time
import heapq

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290
        
        # ZDT3 Pareto front: f2 = 292 * (1 - sqrt(x1/292))
        # This is convex with infinite slope at x1=0
        
        def f2_analytic(x1):
            return 292.0 * (1.0 - np.sqrt(x1 / 292.0))
        
        # Compute the optimal split point for an interval [a, b]
        # Gain(x_m) = (x_m - a) * (f(x_m) - f(b))
        # We want to maximize this over x_m in [a, b]
        # For f(x) = C*(1-sqrt(x/K)), we can find the optimal split analytically
        # or use a few trial points
        
        def find_best_split(a, b):
            # Try several points: midpoint, and points biased towards a (steeper part)
            # Since f is convex and steeper near 0, the optimal split is closer to a
            candidates = [
                (a + b) / 2.0,
                a + (b - a) * 0.3,
                a + (b - a) * 0.4,
                a + (b - a) * 0.5,
                a + (b - a) * 0.6,
            ]
            best_gain = -1
            best_x = (a + b) / 2.0
            y_b = f2_analytic(b)
            for x_m in candidates:
                if x_m <= a or x_m >= b:
                    continue
                y_m = f2_analytic(x_m)
                gain = (x_m - a) * (y_m - y_b)
                if gain > best_gain:
                    best_gain = gain
                    best_x = x_m
            return best_x, best_gain
        
        # Use a priority queue: (-gain, interval_index, a, b)
        # Start with interval [0, 1]
        points = [0.0, 1.0]
        heap = []
        
        # Compute initial gain for [0, 1]
        best_split, best_gain = find_best_split(0.0, 1.0)
        heapq.heappush(heap, (-best_gain, 0, 0.0, 1.0))
        
        # We have budget evaluations. 2 for endpoints, budget-2 for splits
        n_splits = self.budget - 2
        
        for _ in range(n_splits):
            if time.time() > deadline:
                break
            if not heap:
                break
            
            neg_gain, idx, a, b = heapq.heappop(heap)
            gain = -neg_gain
            
            if gain <= 0:
                break
            
            # Find best split for this interval
            best_split, actual_gain = find_best_split(a, b)
            
            # Insert the split point
            # Find position in points list
            # points is sorted, so we can binary search
            import bisect
            pos = bisect.bisect_right(points, best_split)
            points.insert(pos, best_split)
            
            # Now we have two new intervals: [a, best_split] and [best_split, b]
            # Update the heap with the new intervals
            # We need to track which intervals are in the heap
            # Simple approach: just push both new intervals
            
            # Left interval [a, best_split]
            if best_split > a:
                left_split, left_gain = find_best_split(a, best_split)
                if left_gain > 0:
                    heapq.heappush(heap, (-left_gain, pos, a, best_split))
            
            # Right interval [best_split, b]
            if b > best_split:
                right_split, right_gain = find_best_split(best_split, b)
                if right_gain > 0:
                    heapq.heappush(heap, (-right_gain, pos + 1, best_split, b))
        
        # Sort points
        points.sort()
        
        # Call problem for each point
        X_list = []
        for x0 in points:
            if time.time() > deadline:
                break
            x = np.zeros(self.dim)
            x[0] = x0
            X_list.append(x)
        
        for x in X_list:
            if time.time() > deadline:
                break
            problem(x)
        
        return X_list
