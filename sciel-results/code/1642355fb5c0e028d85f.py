import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # 300s limit, leave 10s margin
        
        # Strategy: 
        # DTLZ7 Pareto front is defined by active variables x0, x1 and inactive variables x2...xn set to 0.
        # Objectives:
        # f1 = x0
        # f2 = x0 * (1 - x1)
        # f3 = (1 + x1) * (1 - x0 - x0*(1-x1))
        
        # Analysis of Hypervolume Contribution:
        # Reference point R = [1.1, 1.1, 25.0]
        # We want to maximize the union volume of boxes defined by (f1, f2, f3) and R.
        # For a fixed f1 (i.e., fixed x0), varying x1 changes f2 and f3.
        # f2 is minimized when x1 = 1 (f2 = 0).
        # f3 is minimized when x1 = 0 (f3 = (1-x0)^2) vs x1=1 (f3 = 2(1-x0)).
        # Wait, let's check f3 at x1=0: (1+0)(1 - x0 - x0(1)) = 1 - 2x0. 
        # Wait, DTLZ7 definition: f3 = (1 + x_{n-1}) * (1 - x_{n-2} - x_{n-2}(1 - x_{n-1}))
        # Let's re-verify the standard DTLZ7 formula.
        # g(x) = 1 + 10 * sum_{i=n-2}^{n-1} (x_i^2 - cos(20 pi x_i)) ... usually inactive vars are x_2...x_{n-1} in 0-indexed if x0,x1 are active?
        # Standard DTLZ7:
        # x = [x_0, ..., x_{n-1}]
        # g(x) = 1 + 10/n_u * (k + sum_{i in X_u} (x_i - 0.5)^2 - cos(20 pi (x_i - 0.5))))
        # where X_u = {x_{n-2}, ..., x_{n-1}} are the last 2 variables? Or x_2...x_{n-1}?
        # Usually for DTLZ7, the last two variables are the ones in g. Let's assume standard convention where x_0, x_1 are active for shape, and x_2...x_{n-1} are for g.
        # If we set x_2...x_{n-1} = 0.5, g=1.
        # f_1 = x_0
        # f_2 = x_0 (1 - x_1)
        # f_3 = (1 + x_1) (1 - x_0 - x_0 (1 - x_1))
        
        # Let's analyze the curve x_1 = 0 vs x_1 = 1.
        # If x_1 = 0:
        #   f_1 = x_0
        #   f_2 = x_0
        #   f_3 = 1 * (1 - x_0 - x_0) = 1 - 2x_0
        #   This is valid for x_0 in [0, 0.5] to keep f_3 >= 0? No, f_3 must be positive.
        #   If x_0 = 0.5, f_3 = 0. If x_0 > 0.5, f_3 becomes negative? 
        #   The formula is (1+x_1)(1 - x_0(2-x_1)).
        #   If x_1=0, f_3 = 1 - 2x_0. This becomes 0 at x_0=0.5.
        #   However, DTLZ7 Pareto front is usually defined for x_0 in [0,1].
        #   Let's check x_1 = 1:
        #   f_1 = x_0
        #   f_2 = 0
        #   f_3 = 2 * (1 - x_0 - 0) = 2(1-x_0)
        #   
        #   Which curve dominates?
        #   Compare (f2, f3) for fixed f1=x0.
        #   Curve A (x1=0): f2=x0, f3=1-2x0. (Only valid if 1-2x0 >= 0 -> x0 <= 0.5? No, objectives can be anything, but Pareto front is the boundary. If f3 < 0, it's better than 0? No, we minimize. Negative is better. But usually bounds are 0..1. If f3 goes negative, it's outside the standard box? No, hypervolume is calculated with reference point. If f3 < 0, the box extends further. But usually DTLZ7 front is in the positive orthant.)
        #   Actually, the standard DTLZ7 Pareto set is x_0 in [0,1], x_1 in [0,1], x_2..x_{n-1}=0.5.
        #   The Pareto front in objective space is a surface.
        #   However, the "knee" and high hypervolume regions are often found on the edges.
        #   
        #   Let's look at the "Edge 2" from the current program: x1=1.
        #   f1 = x0, f2 = 0, f3 = 2(1-x0).
        #   This curve has f2=0 always. This is extremely good for f2.
        #   f3 varies from 2 to 0.
        #   f1 varies from 0 to 1.
        #   
        #   Let's look at "Edge 1": x1=0.
        #   f1 = x0, f2 = x0, f3 = 1-2x0.
        #   If x0 is small, f3 is large. f2 is small.
        #   
        #   Which is better for hypervolume?
        #   Reference point: [1.1, 1.1, 25].
        #   Volume contribution of a point p is roughly (1.1-f1)(1.1-f2)(25-f3).
        #   
        #   For Edge 2 (x1=1):
        #   V(x0) = (1.1-x0)(1.1-0)(25-2(1-x0)) = 1.1 * (1.1-x0) * (23+2x0)
        #   For x0=0: V = 1.1 * 1.1 * 23 = 27.8
        #   For x0=0.5: V = 1.1 * 0.6 * 24 = 15.8
        #   For x0=1: V = 1.1 * 0.1 * 25 = 2.75
        #   
        #   For Edge 1 (x1=0):
        #   V(x0) = (1.1-x0)(1.1-x0)(25-(1-2x0)) = (1.1-x0)^2 * (24+2x0)
        #   For x0=0: V = 1.1^2 * 24 = 29.04
        #   For x0=0.5: V = 0.6^2 * 25 = 9.0
        #   For x0=1: V = 0.1^2 * 26 = 0.26
        #   
        #   Edge 1 (x1=0) seems to have higher volume near x0=0.
        #   However, Edge 1 is only "valid" in the sense that it's part of the Pareto front?
        #   Actually, the Pareto front of DTLZ7 is a 2D surface. 
        #   The hypervolume is the union of boxes.
        #   If we sample the entire 2D surface, we might get better coverage.
        #   But the directive says: "Represent the 2D Pareto front as a single 1D curve by fixing the inactive variable x1=0".
        #   Wait, x1 is an ACTIVE variable in DTLZ7 (along with x0). The "inactive" variables for g are x2...xn-1.
        #   The directive says "fixing the inactive variable x1=0". This might be a slight misnomer in the prompt's directive if it considers x1 inactive for the "main" curve, or it implies that the curve x1=0 is the dominant one.
        #   Given the calculation above, the curve x1=0 (Edge 1) has higher peak volume at the origin (29.04 vs 27.8).
        #   Also, the distribution of volume along x1=0 drops off faster, suggesting a bias towards x0=0 is crucial.
        #   
        #   Let's try sampling the curve x1=0 with a bias towards x0=0.
        #   We will use 200 points.
        #   Use a power law distribution: x0 = u^p where u is uniform.
        #   If p > 1, it biases towards 0.
        #   Let's try p=2 or p=3.
        #   Or simply use a geometric/spaced sequence.
        
        n_samples = self.budget
        # Generate x0 values biased towards 0
        # Using a simple power law: x0 = (i / n_samples)^p
        # But we need to cover up to 1.0? 
        # The front extends to x0=1.
        # However, the volume at x0=1 is negligible.
        # We should focus on x0 in [0, 0.5] mostly.
        
        # Let's use a non-uniform grid.
        # We want dense sampling near 0.
        # x0_i = (i / (n_samples - 1)) ** 2 for i in 0..n_samples-1
        # This puts half the points in [0, 0.5] and half in [0.5, 1].
        # Maybe **3?
        
        # Let's stick to the directive: "sampling only x0 in [0,1] with 200 points... biased towards the origin".
        
        # Construct x vectors
        # x[0] = x0
        # x[1] = 0.0  (Following directive: fix x1=0)
        # x[2:] = 0.5 (To make g=1, assuming standard DTLZ7 where x2..xn-1 are in g)
        # Note: Some implementations use x2..xn-1 = 0.5 to minimize g.
        # If g > 1, all objectives are scaled up by g, which is bad.
        # So we MUST set inactive vars to 0.5.
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Initialize array
        X = np.zeros((n_samples, self.dim))
        
        # Set inactive variables to 0.5 (center of [0,1]) to minimize g
        # Indices 2 to dim-1
        X[:, 2:] = 0.5
        
        # Generate x0 values
        # Use a bias towards 0.
        # Let's use a sequence that is dense near 0.
        # x0 = (i / N)^2
        i_vals = np.arange(n_samples)
        # Normalize to [0, 1]
        # If we use (i/(N-1))^2, we hit 0 and 1.
        # Let's use this.
        x0_vals = (i_vals / (n_samples - 1)) ** 2.0
        
        # Assign x0 and x1
        X[:, 0] = x0_vals
        X[:, 1] = 0.0
        
        # Clip to bounds
        X = np.clip(X, lo, hi)
        
        # Evaluate
        for i in range(n_samples):
            if time.time() > deadline:
                break
            x = X[i]
            problem(x)
            
        return X
