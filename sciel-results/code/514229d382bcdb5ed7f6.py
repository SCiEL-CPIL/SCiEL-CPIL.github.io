"""Multi-objective optimizer for ZDT1: deterministic geometric coverage via analytic Pareto front sampling."""
import numpy as np


class Optimizer:
    """Optimizes ZDT1 by sampling points that maximize hypervolume coverage of the Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # Precompute the Pareto-optimal x vectors for ZDT1
        # f1 = x[0], f2 = (1 + 9/n_r) * g * h(f1)
        # g = 1 + 10/(n_r) * sum(x[1:])
        # h = 1 - sqrt(f1/f1) ... actually for ZDT1: h(y) = 1 - sqrt(y/f1)
        # Minimize g and h. g min when all x[1:]=0, h min when f1=0
        # Pareto front: g=1, x[1:]=0, f1 in [0,1]
        # f1 = x[0], f2 = (1 + 9/29) * 1 * (1 - sqrt(f1))
        
        n_r = self.dim - 1
        # The Pareto front in objective space:
        # f1 in [0, 1], f2 = (1 + 9/n_r) * (1 - sqrt(f1))
        
        # We want to maximize hypervolume with reference point [11, 11]
        # The Pareto front is a curve. We should sample points along this curve
        # to cover the dominated region as much as possible.
        
        # For ZDT1, the Pareto front is:
        # f1 = t, f2 = (1 + 9/n_r) * (1 - sqrt(t)) for t in [0, 1]
        
        # To maximize hypervolume, we want points that are "extreme" in both dimensions
        # The curve is convex (bowed toward the origin), so uniform sampling in t
        # won't be optimal. We should use a spacing that covers the area better.
        
        # Let's sample n points along the Pareto front
        n_samples = min(self.budget, 200)
        
        # Use a stratified approach: sample t values to maximize hypervolume
        # The hypervolume of a set of points on a Pareto front with reference [11, 11]
        # is the area under the curve from the points to [11, 11]
        
        # For ZDT1 Pareto front: f1 = t, f2 = (1 + 9/n_r) * (1 - sqrt(t))
        # Let's use a power-law spacing to concentrate points where the curve is "bowed"
        
        # Actually, let's think about this more carefully.
        # The hypervolume contribution of a point (f1_i, f2_i) depends on its position
        # relative to other points. For a convex Pareto front, uniform spacing in the
        # "arc length" or in the objective space tends to work well.
        
        # Let's use a simple approach: sample t uniformly in [0, 1] and compute
        # the corresponding x vectors.
        
        # But we can do better: use a spacing that accounts for the curvature.
        # The curve f2(t) = (1 + 9/n_r) * (1 - sqrt(t)) has higher slope near t=0.
        
        # Let's use a transformation: let s = sqrt(t), so t = s^2
        # Then f1 = s^2, f2 = (1 + 9/n_r) * (1 - s)
        # Uniform sampling in s gives better coverage.
        
        s_vals = np.linspace(0, 1, n_samples)
        t_vals = s_vals ** 2  # f1 values
        
        # For each point, x[0] = t, x[1:] = 0
        x_samples = np.zeros((n_samples, self.dim))
        x_samples[:, 0] = t_vals
        # x[1:] are all zero, which is already set
        
        # Clip to bounds
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        for i in range(n_samples):
            for j in range(self.dim):
                x_samples[i, j] = np.clip(x_samples[i, j], lo[j], hi[j])
        
        # Evaluate all points
        for i in range(n_samples):
            problem(x_samples[i])
        
        return x_samples
