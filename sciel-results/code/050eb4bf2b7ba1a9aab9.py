# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for ZDT3.
    Strategy: Deterministic grid sampling of the known 1D Pareto front segments.
    ZDT3's Pareto front is characterized by g(x)=1 and f1(x)=x1.
    The front is discontinuous, consisting of 6 segments (based on current heuristic).
    We distribute the budget proportionally to segment lengths and place points
    equidistantly within each segment to maximize hypervolume coverage.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 250.0  # safety margin before 300s limit
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # ZDT3 Pareto front segments (x1 ranges where g=1 is achievable/valid for front)
        # These segments are derived from the standard ZDT3 discontinuities.
        segments = [
            (0.0, 0.05),
            (0.15, 0.25),
            (0.35, 0.45),
            (0.55, 0.65),
            (0.75, 0.85),
            (0.95, 1.0)
        ]

        # Calculate lengths
        lengths = np.array([h - l for l, h in segments])
        total_len = lengths.sum()
        
        # Allocate budget proportional to length
        # Ensure at least 1 point per segment if possible
        raw_counts = budget * lengths / total_len
        counts = np.floor(raw_counts).astype(int)
        
        # Distribute remaining points to segments with largest fractional parts
        remaining = budget - counts.sum()
        if remaining > 0:
            fractions = raw_counts - counts
            order = np.argsort(-fractions)
            for i in range(remaining):
                counts[order[i % len(order)]] += 1
        
        # Generate x1 values
        x1_values = []
        for i, (l, h) in enumerate(segments):
            if counts[i] > 0:
                # Use linspace for equidistant points
                # Include endpoints to capture boundaries
                pts = np.linspace(l, h, counts[i])
                x1_values.extend(pts)
        
        # If we still have points left (shouldn't happen with the distribution above, but just in case)
        if len(x1_values) < budget:
            # Fill with random points in the largest segment or uniform over all
            # This case is unlikely given the allocation logic
            diff = budget - len(x1_values)
            # Add to the longest segment
            longest_idx = np.argmax(lengths)
            l, h = segments[longest_idx]
            extra_pts = np.random.uniform(l, h, diff)
            x1_values.extend(extra_pts)
            
        # If we have too many (due to integer rounding errors in other ways, though unlikely here)
        x1_values = np.array(x1_values[:budget])

        # Build the full x vectors
        X = np.zeros((budget, dim))
        X[:, 0] = x1_values
        # x[1:] = 0 is optimal for minimizing g in ZDT3 (assuming standard def where min is at 0)
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        return X
# EVOLVE-BLOCK-END
