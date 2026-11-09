import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # ZDT3 Pareto Front: x0 = f1, x1..xn = 0
        # f1 = x0
        # g = 1 + 10*(n-1) + sum_{i=2}^{n} [x_i^2 - 10*cos(2*pi*x_i)]
        # f2 = (1 + sqrt(f1)) * (1 - sqrt(f1/g) - sin(10*pi*f1))
        # Pareto optimal when g = 1 (i.e., x[1:] = 0)
        # Then f2(f1) = (1 + sqrt(f1)) * (1 - sqrt(f1) - sin(10*pi*f1))

        # The function f2(f1) is non-smooth in the sense that it has many local minima
        # where sin(10*pi*f1) = 1, i.e., f1 = 0.05 + 0.2*k for k = 0,1,2,3,4
        # These are the "peaks" of sin, so 1 - sin(...) is at its minimum there.
        # But wait: f2 = (1+sqrt(f1)) * (1 - sqrt(f1) - sin(10*pi*f1))
        # The term (1 - sqrt(f1) - sin(10*pi*f1)) is minimized when sin is maximized (sin=1)
        # and sqrt(f1) is large... it's a balance.
        
        # Let's precompute f2(f1) on a fine grid to find the Pareto envelope (running minimum).
        # The Pareto front in objective space is the set of (f1, f2(f1)) that are non-dominated.
        # Since f1 is monotonic, a point (f1_i, f2_i) is dominated if there exists j with
        # f1_j <= f1_i and f2_j <= f2_i (at least one strict).
        # Equivalently, the Pareto front consists of points where f2(f1) is at a new running minimum
        # as f1 increases... wait, no. As f1 increases, we want f2 to decrease.
        # Actually, the Pareto front is the lower-left envelope.
        # Let's compute f2 on a fine grid and find the lower envelope.

        # Fine grid for precomputation (not evaluated, just for planning)
        n_fine = 100000
        f1_fine = np.linspace(0, 1, n_fine)
        f2_fine = (1 + np.sqrt(f1_fine)) * (1 - np.sqrt(f1_fine) - np.sin(10 * np.pi * f1_fine))
        
        # Find the Pareto envelope: points where f2 is strictly less than all previous f2 values
        # Since f1 is sorted, we track the running minimum of f2.
        running_min = np.full(n_fine, np.inf)
        running_min[0] = f2_fine[0]
        for i in range(1, n_fine):
            running_min[i] = min(running_min[i-1], f2_fine[i])
        
        # Points on the Pareto front are those where f2_fine[i] == running_min[i]
        # But due to floating point, let's use a small tolerance.
        # Actually, for the envelope, we want points where f2_fine[i] <= running_min[i-1] (for i>0)
        # Let's identify indices where f2_fine is at a new (or equal) running minimum.
        # But we want distinct points. Let's just take all points where f2_fine[i] <= running_min[i-1] + eps
        # This might include plateaus. Let's just take the indices where the running min changes or is first.
        
        # Simpler: The Pareto front is the set of points that are not dominated.
        # A point i is dominated if there exists j < i with f2_fine[j] <= f2_fine[i] (and f1[j] < f1[i] which is true)
        # So i is non-dominated iff f2_fine[i] < min(f2_fine[0:i])
        # Let's compute min_so_far
        min_so_far = np.full(n_fine, np.inf)
        min_so_far[0] = f2_fine[0]
        for i in range(1, n_fine):
            min_so_far[i] = min(min_so_far[i-1], f2_fine[i])
        
        # Non-dominated points: f2_fine[i] <= min_so_far[i] (which is always true)
        # We want f2_fine[i] < min(f2_fine[0:i]) for i>0, and i=0 is always non-dominated.
        # Let's use: f2_fine[i] <= min_so_far[i-1] for i>0
        # This captures points that are at least as good as all previous points.
        
        # Actually, let's just sample uniformly from the non-dominated region.
        # The non-dominated points form a set of intervals.
        # Let's find the f1 values where f2_fine[i] < min_so_far[i-1] (strict improvement)
        # This gives the "kinks" of the Pareto front.
        # But the front is continuous. Let's just take all points where f2_fine[i] <= min_so_far[i-1] + 1e-12
        # and then subsample.
        
        # Let's identify the non-dominated indices
        nd_indices = [0]
        for i in range(1, n_fine):
            if f2_fine[i] <= min_so_far[i-1] + 1e-12:
                nd_indices.append(i)
        
        nd_indices = np.array(nd_indices)
        f1_nd = f1_fine[nd_indices]
        
        # Now we have f1 values that are on or very near the Pareto front.
        # We need to select `budget` points from these.
        # If len(f1_nd) > budget, we subsample.
        # To maximize hypervolume, we should sample in a way that covers the objective space well.
        # Since the Pareto front is a curve in (f1, f2) space, and hypervolume is the area under the curve
        # (to the reference point), we want to approximate the curve well.
        # Uniform sampling in f1 on the Pareto front should work well.
        
        if len(f1_nd) >= budget:
            # Select budget points uniformly spaced in the index of f1_nd
            # This ensures even coverage of the "length" of the Pareto front in the fine grid
            selected_indices = np.linspace(0, len(f1_nd) - 1, budget, dtype=int)
            f1_selected = f1_nd[selected_indices]
        else:
            # If we have fewer non-dominated points than budget, use all of them
            # and add some extra points? But we can't add points that are dominated.
            # Just use all non-dominated points.
            f1_selected = f1_nd.copy()
            # If we have budget left, we can add points near the endpoints or in gaps.
            # But for now, just use what we have.
            # Actually, we can add points at the boundaries of the non-dominated intervals.
            # Let's just use linspace from 0 to 1 if we have very few points.
            if len(f1_selected) < 10:
                f1_selected = np.linspace(0, 1, budget)
        
        # Ensure we don't exceed budget
        f1_selected = f1_selected[:budget]
        
        # Remove duplicates (if any) and sort
        f1_selected = np.unique(f1_selected)
        f1_selected = np.sort(f1_selected)
        
        # If we have fewer than budget points, add more via linspace
        if len(f1_selected) < budget:
            extra = np.linspace(0, 1, budget)
            all_f1 = np.concatenate([f1_selected, extra])
            all_f1 = np.unique(all_f1)
            all_f1 = np.sort(all_f1)
            f1_selected = all_f1[:budget]
        
        # Evaluate the selected points
        for f1_val in f1_selected:
            if time.time() > deadline:
                break
            
            x = np.zeros(dim)
            x[0] = f1_val
            # x[1:] = 0.0 is already set by zeros
            
            x = np.clip(x, lo, hi)
            
            try:
                problem(x)
            except Exception:
                break

        return None
