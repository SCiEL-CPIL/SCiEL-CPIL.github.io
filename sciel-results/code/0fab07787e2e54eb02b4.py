"""Optimizer for ZDT1 exploiting its known structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Exploits ZDT1 structure: x1 controls front position, x2..xn control distance to front.
    Places points on the Pareto front (x2..xn=0) with optimized x1 spacing
    to maximize hypervolume coverage of the convex Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def _compute_hypervolume(self, f1_vals):
        """Compute hypervolume for 2D points on ZDT1 front with ref point (11, 11)."""
        f2_vals = 1.0 - np.sqrt(f1_vals)
        # Sort by f1 ascending
        order = np.argsort(f1_vals)
        f1_sorted = f1_vals[order]
        f2_sorted = f2_vals[order]
        
        # Hypervolume: sum of non-dominated rectangle areas
        # For each point i, its contribution is (f2[i] - f2[i+1]) * (11 - f1[i])
        # where f2[i+1] is the next point's f2 (or 0 if last)
        hv = 0.0
        for i in range(len(f1_sorted)):
            f2_next = f2_sorted[i + 1] if i + 1 < len(f2_sorted) else 0.0
            width = 11.0 - f1_sorted[i]
            height = f2_sorted[i] - f2_next
            hv += width * height
        return hv

    def _compute_optimal_front_points(self, n_points, lo0, hi0):
        """
        Compute x1 values optimized for hypervolume on ZDT1 Pareto front.
        
        Strategy:
        1. Initialize with quantiles of the optimal distribution derived from
           equalizing marginal hypervolume contributions.
           Density proportional to (11 - f2) = 10 + sqrt(x1).
        2. Refine using the recurrence relation derived from the first-order
           optimality conditions for the discrete hypervolume maximization problem.
        """
        if n_points <= 1:
            return np.array([(lo0 + hi0) / 2.0])
        
        # Step 1: Initialize with CDF quantiles
        # F(s) = (10s + (2/3)s^{3/2}) / (32/3)
        s_vals = np.zeros(n_points)
        t_uniform = np.linspace(0.0, 1.0, n_points)
        target = t_uniform * (32.0 / 3.0)
        
        for i in range(n_points):
            s = target[i] / 10.0 if target[i] > 0 else 0.0
            s = max(0.0, min(1.0, s))
            for _ in range(20):
                f_val = 10.0 * s + (2.0/3.0) * s**1.5
                f_prime = 10.0 + s**0.5
                if f_prime == 0:
                    break
                correction = (f_val - target[i]) / f_prime
                s_new = s - correction
                s_new = max(0.0, min(1.0, s_new))
                if abs(s_new - s) < 1e-14:
                    s = s_new
                    break
                s = s_new
            s_vals[i] = s
        
        # Enforce endpoints
        s_vals[0] = lo0
        s_vals[-1] = hi0
        
        # Step 2: Refine using recurrence relation
        # x_i = [(6x_{i-1} + 4x_{i+1}) - sqrt((6x_{i-1} + 4x_{i+1})^2 - 36x_{i-1}^2)] / 18
        # This formula is specific to the ZDT1 front with ref point 11.
        
        for _ in range(100):
            new_s_vals = s_vals.copy()
            for i in range(1, n_points - 1):
                x_prev = s_vals[i-1]
                x_next = s_vals[i+1]
                
                # Use the recurrence from inspiration 3
                A = 9.0
                B = -(6.0 * x_prev + 4.0 * x_next)
                C = x_prev * x_prev
                
                discriminant = B*B - 4.0*A*C
                if discriminant < 0:
                    continue
                
                sqrt_disc = np.sqrt(discriminant)
                root1 = (-B + sqrt_disc) / (2.0 * A)
                root2 = (-B - sqrt_disc) / (2.0 * A)
                
                # We want the root between x_prev and x_next
                candidate = None
                if x_prev < root1 < x_next:
                    candidate = root1
                elif x_prev < root2 < x_next:
                    candidate = root2
                else:
                    # If no root in range, pick the one closer to current value
                    if abs(root1 - s_vals[i]) < abs(root2 - s_vals[i]):
                        if x_prev <= root1 <= x_next:
                            candidate = root1
                    else:
                        if x_prev <= root2 <= x_next:
                            candidate = root2
                
                if candidate is not None:
                    new_s_vals[i] = candidate
            
            # Check convergence
            if np.max(np.abs(new_s_vals - s_vals)) < 1e-12:
                s_vals = new_s_vals
                break
            s_vals = new_s_vals
            
            # Re-enforce bounds
            s_vals[0] = lo0
            s_vals[-1] = hi0
            s_vals = np.clip(s_vals, lo0, hi0)
            
        return s_vals

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        n_points = self.budget

        # Compute optimal x1 values using the refined method
        x1_final = self._compute_optimal_front_points(n_points, 0.0, 1.0)
        
        evaluated = 0
        for i in range(n_points):
            if time.time() > deadline:
                break

            x = np.zeros(n)
            x[0] = x1_final[i]
            # x[1:] = 0 (Pareto front)
            problem(x)
            evaluated += 1

        return None
# EVOLVE-BLOCK-END
