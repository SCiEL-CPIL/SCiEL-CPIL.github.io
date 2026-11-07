"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer for DTLZ4 using data-driven variable identification
    followed by structured sampling of the identified Pareto front.
    
    Strategy:
    1. Use a small fraction of the budget to perform a one-at-a-time (OAT) sensitivity
       analysis to determine which variables affect the objective values (position/angle
       variables) and which only affect the distance to the Pareto front (g-variables).
    2. Once identified, fix the g-variables to their optimal values (typically 0.5 for
       DTLZ4) and sample the position variables using a low-discrepancy sequence
       (Halton/Sobol) to cover the Pareto front uniformly.
    3. This avoids the brittle assumption about which variables are position vs g.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _halton_sequence(self, num_points, dim):
        """Generate a Halton low-discrepancy sequence."""
        points = []
        for n in range(1, num_points + 1):
            point = []
            for base in range(2, dim + 2):
                f = 1.0
                i = n
                while i > 0:
                    f /= base
                    i = (i - 1) // base
                point.append(f)
            points.append(point)
        return np.array(points)

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        mid = 0.5 * (lo + hi)
        
        # Phase 1: Variable sensitivity analysis
        # Use ~25% of budget for this
        n_sens = max(10, min(self.dim * 2, self.budget // 4))
        
        # Baseline: all variables at midpoint
        x_base = mid.copy()
        f_base = np.array(problem(x_base))
        
        # For each variable, evaluate at low and high bounds to see effect on objectives
        # We'll track the variance in objective values when each variable changes
        var_effects = np.zeros(self.dim)
        
        for i in range(self.dim):
            if time.time() > deadline - 10:
                break
            
            x_low = mid.copy()
            x_low[i] = lo[i]
            f_low = np.array(problem(x_low))
            
            x_high = mid.copy()
            x_high[i] = hi[i]
            f_high = np.array(problem(x_high))
            
            # Measure the change in objective values
            # If a variable is a g-variable, changing it should change all objectives
            # by a similar factor (scaling). If it's a position variable, it changes
            # the direction/shape of the objectives.
            
            # Simple metric: sum of squared differences from baseline
            # For g-variables, all objectives scale together
            # For position variables, objectives change in different ways
            
            diff_low = f_low - f_base
            diff_high = f_high - f_base
            
            # Use the norm of the difference
            var_effects[i] = np.sum(diff_low ** 2) + np.sum(diff_high ** 2)
        
        # Heuristic: In DTLZ4, g-variables typically have larger effects on the
        # magnitude of all objectives, while position variables affect the ratios.
        # A more robust approach: check if the objective vectors remain proportional
        # when a variable changes (g-variable) or change direction (position variable).
        
        # Let's use a different metric: the cosine similarity between the change vectors
        # For g-variables, f_low and f_high should be roughly proportional to f_base
        # For position variables, the direction changes
        
        pos_vars = []
        g_vars = []
        
        for i in range(self.dim):
            if time.time() > deadline - 10:
                g_vars.append(i)
                continue
                
            x_low = mid.copy()
            x_low[i] = lo[i]
            f_low = np.array(problem(x_low))
            
            x_high = mid.copy()
            x_high[i] = hi[i]
            f_high = np.array(problem(x_high))
            
            # Check if the objective vectors are proportional
            # Compute the ratio of each objective to the first
            if abs(f_base[0]) > 1e-10:
                ratio_base = f_base / f_base[0]
                ratio_low = f_low / f_low[0] if abs(f_low[0]) > 1e-10 else np.zeros_like(f_base)
                ratio_high = f_high / f_high[0] if abs(f_high[0]) > 1e-10 else np.zeros_like(f_base)
                
                # If ratios are similar, it's likely a g-variable
                # If ratios differ significantly, it's a position variable
                diff_ratio_low = np.sum((ratio_low - ratio_base) ** 2)
                diff_ratio_high = np.sum((ratio_high - ratio_base) ** 2)
                
                if diff_ratio_low < 0.1 and diff_ratio_high < 0.1:
                    g_vars.append(i)
                else:
                    pos_vars.append(i)
            else:
                # Fallback: use magnitude change
                if var_effects[i] > np.median(var_effects) * 1.5:
                    g_vars.append(i)
                else:
                    pos_vars.append(i)
        
        # Ensure we have at least 1 position variable
        if len(pos_vars) == 0:
            # Fallback: assume first 2 variables are position
            pos_vars = [0, 1]
            g_vars = list(range(2, self.dim))
        
        # Limit position variables to reasonable number (typically 2 for 3 objectives)
        if len(pos_vars) > 3:
            # Keep the ones with smallest ratio difference (most likely position)
            # For now, just take the first 3
            pos_vars = pos_vars[:3]
            g_vars = [i for i in range(self.dim) if i not in pos_vars]
        
        # Phase 2: Sample the Pareto front
        # Fix g-variables to their optimal values (midpoint for DTLZ4)
        # Sample position variables using Halton sequence
        
        remaining_budget = self.budget - n_sens - self.dim * 2  # Approximate
        if remaining_budget < 10:
            remaining_budget = max(10, self.budget // 2)
        
        # Generate Halton points for position variables
        n_pos = len(pos_vars)
        n_points = min(remaining_budget, 200)
        
        # If n_pos == 2, use 2D Halton; if n_pos == 3, use 3D Halton
        if n_pos == 1:
            # 1D: just linspace
            pos_vals = np.linspace(0.01, 0.99, n_points).reshape(-1, 1)
        elif n_pos == 2:
            pos_vals = self._halton_sequence(n_points, 2)
            pos_vals = 0.01 + 0.98 * pos_vals  # Avoid boundaries
        elif n_pos == 3:
            pos_vals = self._halton_sequence(n_points, 3)
            pos_vals = 0.01 + 0.98 * pos_vals
        else:
            # Fallback to random if too many position variables
            pos_vals = np.random.rand(n_points, n_pos)
            pos_vals = 0.01 + 0.98 * pos_vals
        
        # Create population
        X = np.zeros((n_points, self.dim))
        
        # Set g-variables to midpoint
        for i in g_vars:
            X[:, i] = mid[i]
        
        # Set position variables from Halton sequence
        for j, var_idx in enumerate(pos_vars):
            X[:, var_idx] = pos_vals[:, j]
        
        # Ensure within bounds
        for i in range(self.dim):
            X[:, i] = np.clip(X[:, i], lo[i], hi[i])
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        # Check time
        if time.time() > deadline:
            return X
        
        # Optional: Add a few random points for robustness
        n_random = min(5, self.budget - n_points)
        if n_random > 0:
            X_rand = np.random.rand(n_random, self.dim)
            X_rand = lo + X_rand * (hi - lo)
            X = np.vstack([X, X_rand])
        
        return X
# EVOLVE-BLOCK-END
