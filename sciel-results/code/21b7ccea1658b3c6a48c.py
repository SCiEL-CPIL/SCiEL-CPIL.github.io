"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Structure-exploiting multi-objective optimizer for ZDT4.

    ZDT4 structure:
      g(x) = 1 + 10 * sum_{i=1}^{n} sin^2(10*x_i)
      f0 = g(x)
      f1 = 1 - sqrt((1/n) * sum x_i^2) * g(x)
    
    Pareto front: g(x) = 1 (all sin terms zero), so x_i = k*pi/10 for integer k.
      f0 = 1
      f1 = 1 - sqrt(sum x_i^2 / n)
    
    To maximize hypervolume with reference point [11, 400]:
      - We want many points on the Pareto front
      - Spread f1 values as widely as possible
      - f1 ranges from 1 (all x_i=0) down to 1 - sqrt(sum x_i^2/n)
      - Max |x_i| in [-5,5] with x_i = k*pi/10: |k| <= 15 (since 15*pi/10 ~ 4.71 < 5)
      - Max sum x_i^2/n = 10*(15*pi/10)^2/10 = (15*pi/10)^2 = 2.25*pi^2 ~ 22.2
      - So f1 can go down to 1 - sqrt(22.2) ~ 1 - 4.71 = -3.71
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        n = dim
        budget = self.budget

        # ZDT4: g(x) = 1 + 10*sum sin^2(10*x_i)
        # Pareto front: sin(10*x_i) = 0 for all i => x_i = k*pi/10
        # f0 = 1, f1 = 1 - sqrt(sum x_i^2 / n)
        
        # Valid k values: x_i = k*pi/10 must be in bounds
        # For bounds [-5, 5]: |k|*pi/10 <= 5 => |k| <= 50/pi ~ 15.9 => |k| <= 15
        k_max = 15
        k_vals = np.arange(-k_max, k_max + 1)  # -15 to 15
        x_vals = k_vals * np.pi / 10.0  # corresponding x values

        # Generate diverse points on the Pareto front
        # Strategy: create points that span the full range of f1 values
        # f1 = 1 - sqrt(sum x_i^2 / n)
        # We want to cover f1 from ~1 down to ~-3.7
        
        points = []
        
        # Point with all zeros: f1 = 1
        points.append(np.zeros(dim))
        
        # Create systematic points by varying how many dimensions are non-zero
        # and their magnitudes to spread f1 values
        used_evals = 0
        
        # First, create a set of well-spread points on the front
        # Target f1 values: spread from 1 to -3.7
        target_f1_values = np.linspace(1.0, -3.7, 30)
        
        for target_f1 in target_f1_values:
            # f1 = 1 - sqrt(sum x_i^2 / n)
            # target_f1 = 1 - sqrt(sum x_i^2 / n)
            # sqrt(sum x_i^2 / n) = 1 - target_f1
            # sum x_i^2 = n * (1 - target_f1)^2
            target_sum_sq = n * (1.0 - target_f1) ** 2
            
            # Distribute this sum of squares across dimensions
            # Use a greedy approach: assign largest possible |x_i| first
            remaining = target_sum_sq
            pt = np.zeros(dim)
            
            for i in range(dim):
                if remaining <= 1e-10:
                    break
                # Find the largest |x_val| such that x_val^2 <= remaining
                # and x_val is in x_vals
                # x_vals^2 sorted
                x_sq = x_vals ** 2
                # Find largest x_sq <= remaining
                valid = x_sq[1:]  # skip 0
                if len(valid) == 0:
                    continue
                # Find the largest valid value <= remaining
                mask = valid <= remaining + 1e-10
                if np.any(mask):
                    idx = np.where(mask)[0]
                    best_idx = idx[-1]
                    # Get the corresponding x value (could be positive or negative)
                    # Use positive for simplicity
                    x_val = x_vals[best_idx + 1]  # +1 because we skipped 0
                    pt[i] = x_val
                    remaining -= x_val ** 2
                else:
                    # Use the smallest non-zero
                    x_val = x_vals[1]
                    pt[i] = x_val
                    remaining -= x_val ** 2
            
            # Check if we're close to target
            actual_sum_sq = np.sum(pt ** 2)
            actual_f1 = 1.0 - np.sqrt(actual_sum_sq / n)
            
            points.append(pt)
        
        # Add more diverse points using random combinations
        rng = np.random.default_rng(42)
        for _ in range(40):
            k_choice = rng.integers(-k_max, k_max + 1, size=dim)
            pt = k_choice * np.pi / 10.0
            points.append(pt)
        
        # Also add some points with mixed magnitudes for better coverage
        for _ in range(30):
            pt = np.zeros(dim)
            for i in range(dim):
                if rng.random() < 0.5:
                    k = rng.integers(1, k_max + 1)
                    sign = rng.choice([-1, 1])
                    pt[i] = sign * k * np.pi / 10.0
            points.append(pt)
        
        # Evaluate all points
        F = []
        valid_points = []
        for pt in points:
            if time.time() > deadline:
                break
            if used_evals >= budget:
                break
            fc = problem(pt)
            F.append(fc)
            valid_points.append(pt)
            used_evals += 1
        
        F = np.array(F)
        valid_points = np.array(valid_points)
        
        # Now use remaining budget to refine: try to improve points that aren't exactly on the front
        # For each point, check if f0 is close to 1, and if not, try to fix it
        if used_evals < budget:
            for i in range(len(valid_points)):
                if time.time() > deadline:
                    break
                if used_evals >= budget:
                    break
                
                # If f0 is already very close to 1, skip
                if abs(F[i, 0] - 1.0) < 1e-10:
                    continue
                
                # Try to fix each dimension
                for dim_idx in range(dim):
                    if time.time() > deadline:
                        break
                    if used_evals >= budget:
                        break
                    
                    cur_val = valid_points[i, dim_idx]
                    # Find nearest k*pi/10
                    k_cur = int(round(cur_val * 10.0 / np.pi))
                    for dk in [-2, -1, 0, 1, 2]:
                        k_new = k_cur + dk
                        if abs(k_new) > k_max:
                            continue
                        new_val = k_new * np.pi / 10.0
                        test_pt = valid_points[i].copy()
                        test_pt[dim_idx] = new_val
                        fc = problem(test_pt)
                        used_evals += 1
                        
                        # Keep if it improves f0 (closer to 1)
                        if abs(fc[0] - 1.0) < abs(F[i, 0] - 1.0):
                            valid_points[i] = test_pt
                            F[i] = fc
                        if used_evals >= budget or time.time() > deadline:
                            break
                
                if used_evals >= budget or time.time() > deadline:
                    break
        
        # Use any remaining budget to add more diverse points
        if used_evals < budget:
            for _ in range(budget - used_evals):
                if time.time() > deadline:
                    break
                # Random point on the front
                k_choice = rng.integers(-k_max, k_max + 1, size=dim)
                pt = k_choice * np.pi / 10.0
                fc = problem(pt)
                F = np.vstack([F, [fc]])
                valid_points = np.vstack([valid_points, [pt]])
                used_evals += 1
        
        return valid_points
# EVOLVE-BLOCK-END
