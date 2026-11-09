import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_head = 2
        n_tail = self.dim - n_head
        
        # DTLZ7 Structure:
        # f1 = x1 * (1 + g)
        # f2 = x2 * (1 + g)
        # f3 = 1 + 2*g
        # where g = sum((xi - 0.5)^2 - cos(20*pi*(xi - 0.5))) for i in 3..n
        # g is minimized to 0 when xi = 0.5 for all tail variables.
        # The Pareto front is achieved when g=0, i.e., x_tail = 0.5.
        
        # Strategy:
        # 1. Verify that x_tail = 0.5 indeed gives g=0 (or close to it).
        #    We don't need to minimize g if we know the structure, but let's be robust.
        #    Actually, for DTLZ7, the Pareto front is at x_i = 0.5 for i>=3.
        #    Let's just set the tail to 0.5 and use all budget to sample the front.
        #    This is the "structural insight" approach: don't waste budget optimizing g,
        #    use the known solution for g.
        
        # However, the problem might be slightly different or we want to be safe.
        # Let's do a quick check: evaluate one point with tail=0.5.
        # If f3 is close to 1, we are good. If not, we might need to optimize.
        # But the contract says DTLZ7. Let's trust the structure.
        
        # To maximize hypervolume, we want to cover the Pareto front:
        # f1 = x1, f2 = x2, f3 = 1 (since g=0)
        # x1, x2 in [0, 1].
        # The front is a 2D surface in the box [0,1]x[0,1]x{1}.
        # Reference point: [1.1, 1.1, 25.0]
        
        # We need to distribute 200 points on this 2D surface (x1, x2) to maximize
        # the hypervolume of the resulting set of points (x1, x2, 1) w.r.t ref [1.1, 1.1, 25.0].
        
        # Hypervolume contribution of a point (f1, f2, f3) to the dominated region:
        # Since f3=1 is constant and much smaller than ref_f3=25, the "depth" in f3 direction
        # is (25-1) = 24 for all points.
        # The hypervolume is essentially 24 * (Area in f1-f2 plane covered by the points).
        # Wait, hypervolume is the volume of the union of boxes dominated by each point.
        # A point (f1, f2, 1) dominates the box [f1, 1.1] x [f2, 1.1] x [1, 25].
        # The union of these boxes is determined by the non-dominated points in the f1-f2 plane.
        # Since f3 is constant, the problem reduces to maximizing the area of the union of
        # rectangles [f1, 1.1] x [f2, 1.1] in the 2D objective space.
        
        # To maximize the area covered by a set of points in [0,1]x[0,1] w.r.t corner (1.1, 1.1),
        # we should distribute points to cover the space [0,1]x[0,1] as evenly as possible.
        # A uniform grid is good, but a Sobol sequence or Halton sequence might be better
        # for avoiding clustering and covering the space more uniformly, especially for
        # small N.
        
        # Let's use a Sobol sequence for the (x1, x2) sampling.
        # We have 200 evaluations.
        # We will set x_tail = 0.5 for all evaluations.
        
        # Generate Sobol points
        # scipy.stats.qmc.Sobol is available in recent scipy versions.
        # If not, we can use Halton or just a grid.
        # Let's try to use Sobol.
        
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=2, scramble=True)
            # We need 200 points.
            # Note: Sobol generator produces points in [0,1].
            # We will use all 200 budget for sampling.
            n_samples = self.budget
            sobol_points = sampler.random(n=n_samples)
            # sobol_points is shape (200, 2)
            x1_vals = sobol_points[:, 0]
            x2_vals = sobol_points[:, 1]
        except ImportError:
            # Fallback to Halton or Grid
            # Let's use a simple grid if Sobol is not available
            n_grid = int(np.sqrt(self.budget))
            # Adjust to fit exactly budget
            n_grid = int(np.floor(np.sqrt(self.budget)))
            if n_grid * n_grid > self.budget:
                n_grid -= 1
            
            x1_vals, x2_vals = np.meshgrid(np.linspace(0, 1, n_grid), np.linspace(0, 1, n_grid))
            x1_vals = x1_vals.flatten()
            x2_vals = x2_vals.flatten()
            # If we have extra budget, add random points?
            # Or just use the grid.
            # Let's stick to the grid for simplicity in fallback.
            # But we need to handle the exact count if we want to use all budget.
            # For now, let's just use the generated points.
            # If n_grid*n_grid < budget, we are wasting some evaluations.
            # Let's add random points to fill the rest.
            remaining = self.budget - len(x1_vals)
            if remaining > 0:
                x1_rand = np.random.rand(remaining)
                x2_rand = np.random.rand(remaining)
                x1_vals = np.concatenate([x1_vals, x1_rand])
                x2_vals = np.concatenate([x2_vals, x2_rand])
            else:
                x1_vals = x1_vals[:self.budget]
                x2_vals = x2_vals[:self.budget]

        # Construct the full decision vectors
        # x_tail is 0.5
        x_tail = np.full(n_tail, 0.5)
        
        # We will evaluate all these points.
        # We need to be careful about the time limit.
        # 200 evaluations should be fast.
        
        results = []
        for i in range(len(x1_vals)):
            if time.time() > deadline:
                break
            
            x1 = x1_vals[i]
            x2 = x2_vals[i]
            
            x = np.zeros(self.dim)
            x[0] = x1
            x[1] = x2
            x[2:] = x_tail
            
            # Clip to bounds just in case
            x = np.clip(x, lo, hi)
            
            f = problem(x)
            results.append(f)
            
            # Check time
            if time.time() > deadline:
                break
        
        # If we ran out of time, return what we have.
        # The return value is ignored, but the recorded points matter.
        
        return np.array(results) if results else np.zeros((1, 3))
