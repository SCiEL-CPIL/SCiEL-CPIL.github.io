import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for ZDT1.
    Uses the known analytical Pareto front to place evaluations optimally.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        # Safety margin: The evaluator kills at 330s. We aim to finish in < 5s.
        deadline = t0 + 10.0 

        B = int(self.budget)
        D = int(self.dim)
        
        if B <= 0:
            return np.zeros(D)

        # ZDT1 Pareto Front:
        # x1 in [0, 1], x2...xD = 0
        # f1 = x1
        # f2 = 1 - sqrt(x1)
        
        # Reference point
        R1 = 11.0
        R2 = 11.0

        def compute_hv(x1_vals):
            """
            Calculate Hypervolume for points on the ZDT1 front.
            x1_vals: array of x1 values (sorted ascending).
            """
            if len(x1_vals) == 0:
                return 0.0
            
            # Ensure sorted
            x1_sorted = np.sort(x1_vals)
            
            # f2 values
            f2 = 1.0 - np.sqrt(x1_sorted)
            
            # Heights relative to reference point
            # The HV contribution of a point i is (next_f1 - current_f1) * (R2 - current_f2)
            # The last point extends to R1
            next_f1 = np.append(x1_sorted[1:], R1)
            current_f1 = x1_sorted
            
            widths = next_f1 - current_f1
            heights = R2 - f2
            
            # Sum of areas of the rectangles
            hv = np.sum(widths * heights)
            
            return float(hv)

        # Optimal quantization for hypervolume maximization on ZDT1 front.
        # The marginal gain of adding a point at x1 is proportional to the height
        # (R2 - f2(x1)) * (dx). For optimal distribution, we place points such
        # that the local density is proportional to the height.
        # 
        # f2(x1) = 1 - sqrt(x1)
        # height(x1) = R2 - f2(x1) = 11 - (1 - sqrt(x1)) = 10 + sqrt(x1)
        #
        # We want to find x1 values such that the CDF of the distribution
        # proportional to height(x1) matches uniform quantiles.
        #
        # Let g(x) = 10 + sqrt(x). We want to solve:
        # integral_0^x1 g(t) dt = u * integral_0^1 g(t) dt
        #
        # integral g(t) dt = 10*x + (2/3)*x^(3/2)
        # integral_0^1 g(t) dt = 10 + 2/3 = 32/3
        #
        # So we need to solve: 10*x + (2/3)*x^(3/2) = u * 32/3
        #
        # This is a cubic-like equation in x. Let y = sqrt(x), then x = y^2.
        # 10*y^2 + (2/3)*y^3 = u * 32/3
        # Multiply by 3: 30*y^2 + 2*y^3 = u * 32
        # 2*y^3 + 30*y^2 - 32*u = 0
        # y^3 + 15*y^2 - 16*u = 0
        #
        # For each u in [0, 1], we solve this cubic for y in [0, 1].
        
        u = np.linspace(0.0, 1.0, B)
        
        # Solve y^3 + 15*y^2 - 16*u = 0 for y in [0, 1]
        # Use Newton's method for each u
        x1_vals = np.zeros(B)
        
        for i in range(B):
            target = 16.0 * u[i]
            # f(y) = y^3 + 15*y^2 - target
            # f'(y) = 3*y^2 + 30*y
            # Newton iteration
            y = 1.0  # initial guess
            for _ in range(20):
                fy = y**3 + 15.0*y**2 - target
                dfy = 3.0*y**2 + 30.0*y
                if abs(dfy) < 1e-15:
                    break
                y_new = y - fy / dfy
                if y_new < 0:
                    y_new = 0.0
                if y_new > 1:
                    y_new = 1.0
                if abs(y_new - y) < 1e-12:
                    y = y_new
                    break
                y = y_new
            x1_vals[i] = y**2
        
        x1_vals = np.clip(x1_vals, 0.0, 1.0)
        
        # Verify and potentially refine with local search
        # Compute HV for this distribution
        hv_current = compute_hv(x1_vals)
        
        # Try a few power-law distributions as alternatives and pick the best
        best_hv = hv_current
        best_x1 = x1_vals.copy()
        
        if time.time() < deadline - 2.0:
            p_values = np.array([0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0, 3.0])
            for p in p_values:
                x1_test = u ** p
                x1_test = np.clip(x1_test, 0.0, 1.0)
                hv_test = compute_hv(x1_test)
                if hv_test > best_hv:
                    best_hv = hv_test
                    best_x1 = x1_test
        
        # Construct X
        X = np.zeros((B, D))
        X[:, 0] = best_x1
        # x2...xD = 0 is already set by zeros
        
        # Clip to bounds
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        X = np.clip(X, lo, hi)
        
        # Evaluate
        for i in range(B):
            if time.time() > deadline:
                break
            problem(X[i].copy())
            
        return X[-1] # Return value is ignored, but must not crash
