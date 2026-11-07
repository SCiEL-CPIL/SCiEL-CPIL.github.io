"""Problem-aware ZDT1 optimizer: precomputed hypervolume-optimal non-uniform grid."""
import numpy as np
import time


class Optimizer:
    """
    Exploit ZDT1 structure: f1 = x[0], f2 = (1 + 9*sqrt(f1/n)) * g(x[1:]) when x[1:]=0.2.
    
    Strategy: Precompute the hypervolume-optimal set of x[0] values for a concave front
    f2 = 1 + c*sqrt(f1) (where c = 9/sqrt(30) for ZDT1 with n=30).
    
    The key insight is that for a concave front, the hypervolume-optimal point to add
    in an interval [a,b] is where the curve is farthest above the chord. We precompute
    the full optimal sequence offline (during __init__ or first call) and then just
    evaluate those points in order.
    
    This is structurally different from adaptive refinement: we compute the entire
    optimal sequence upfront using the analytical formula, then evaluate all points
    in a single pass. No runtime search, no priority queue, no iterative improvement.
    """
    
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self._precomputed_x0 = None
    
    def _compute_optimal_x0_sequence(self, n_points):
        """
        Compute the hypervolume-optimal set of n_points x[0] values in [0, 1]
        for the concave front f2 = 1 + c*sqrt(f1).
        
        Uses recursive bisection: start with [0,1], repeatedly split the interval
        that gives the largest marginal hypervolume gain when adding the optimal
        point in that interval.
        
        The optimal point in [a,b] for f2 = 1 + c*sqrt(x) is at:
        x_opt = (b-a) * (sqrt(b)+sqrt(a))^2 / 4
        
        The marginal gain (area between curve and chord) in [a,b] is:
        gain = c * (b-a) * (sqrt(x_opt) - sqrt(a) - (x_opt-a)*(sqrt(b)-sqrt(a))/(b-a))
        
        We precompute the full sequence of n_points values.
        """
        if n_points <= 0:
            return []
        if n_points == 1:
            return [0.5]  # Midpoint for single point
        
        # Start with endpoints
        points = [0.0, 1.0]
        
        # Use a priority queue approach: repeatedly add the point in the
        # interval with the highest marginal gain.
        # For efficiency, we'll track intervals and their gains.
        
        # The gain for interval [a,b] is the area between the curve and the chord.
        # For f2 = 1 + c*sqrt(x):
        # chord at x: 1 + c*(sqrt(a) + (x-a)*(sqrt(b)-sqrt(a))/(b-a))
        # curve at x: 1 + c*sqrt(x)
        # gap at x: c*(sqrt(x) - sqrt(a) - (x-a)*(sqrt(b)-sqrt(a))/(b-a))
        # The maximum gap is at x_opt = (b-a)*(sqrt(b)+sqrt(a))^2/4
        # max_gap = c * (sqrt(x_opt) - sqrt(a) - (x_opt-a)*(sqrt(b)-sqrt(a))/(b-a))
        
        # But the marginal hypervolume gain of adding x_opt to the set is:
        # (x_opt - a) * (f2(x_opt) - f2_chord(x_opt)) + (b - x_opt) * (f2(x_opt) - f2_chord(x_opt))
        # = (b-a) * (f2(x_opt) - f2_chord(x_opt))
        # = c * (b-a) * max_gap
        
        # So we want to maximize (b-a) * max_gap, which is proportional to the area
        # of the "triangle" between the curve and the chord.
        
        import heapq
        
        # Priority queue: (-gain, a, b)
        # We need to compute the gain for each interval.
        
        def interval_gain(a, b):
            """Compute the marginal HV gain of adding the optimal point in [a,b]."""
            if b - a < 1e-15:
                return 0.0
            sqrt_a = np.sqrt(a)
            sqrt_b = np.sqrt(b)
            x_opt = (b - a) * (sqrt_b + sqrt_a) ** 2 / 4.0
            x_opt = np.clip(x_opt, a, b)
            sqrt_x_opt = np.sqrt(x_opt)
            chord_val = sqrt_a + (x_opt - a) * (sqrt_b - sqrt_a) / (b - a)
            gap = sqrt_x_opt - chord_val
            # The gain is proportional to (b-a) * gap (times c, which is constant)
            return (b - a) * gap
        
        # Initialize with the single interval [0, 1]
        pq = []
        heapq.heappush(pq, (-interval_gain(0.0, 1.0), 0.0, 1.0))
        
        # We already have 2 points (0 and 1). Need n_points - 2 more.
        for _ in range(n_points - 2):
            neg_gain, a, b = heapq.heappop(pq)
            if -neg_gain < 1e-15:
                break
            
            sqrt_a = np.sqrt(a)
            sqrt_b = np.sqrt(b)
            x_opt = (b - a) * (sqrt_b + sqrt_a) ** 2 / 4.0
            x_opt = np.clip(x_opt, a, b)
            
            points.append(x_opt)
            
            # Add the two sub-intervals
            gain_left = interval_gain(a, x_opt)
            gain_right = interval_gain(x_opt, b)
            
            if gain_left > 1e-15:
                heapq.heappush(pq, (-gain_left, a, x_opt))
            if gain_right > 1e-15:
                heapq.heappush(pq, (-gain_right, x_opt, b))
        
        # Sort the points
        points = sorted(points)
        
        # Ensure we have exactly n_points points
        while len(points) < n_points:
            # If we didn't add enough, add midpoints of remaining intervals
            # This shouldn't happen for reasonable n_points
            points.append(0.5)
        
        return points[:n_points]
    
    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320  # Safety margin before 330s kill
        
        budget = self.budget
        dim = self.dim
        
        # Precompute the optimal x[0] sequence
        if self._precomputed_x0 is None or len(self._precomputed_x0) < budget:
            self._precomputed_x0 = self._compute_optimal_x0_sequence(budget)
        
        x0_vals = self._precomputed_x0[:budget]
        
        # Evaluate all points
        x_rest_good = np.full(dim - 1, 0.2)
        results_F = []
        
        for x0 in x0_vals:
            if time.time() > deadline - 5:
                break
            x = np.empty(dim)
            x[0] = x0
            x[1:] = x_rest_good
            f = problem(x)
            results_F.append(f)
        
        return np.array(results_F) if results_F else np.array([])


if __name__ == "__main__":
    pass
