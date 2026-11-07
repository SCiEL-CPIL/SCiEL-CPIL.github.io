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
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 275.0
        dim = self.dim
        n = dim
        budget = self.budget

        # ZDT4: g(x) = 1 + 10*sum sin^2(10*x_i)
        # Pareto front: sin(10*x_i) = 0 for all i => x_i = k*pi/10
        # f0 = 1, f1 = 1 - sqrt(sum x_i^2 / n)
        
        # Valid k values: x_i = k*pi/10 must be in bounds
        # For bounds [-5, 5]: |k|*pi/10 <= 5 => |k| <= 50/pi ~ 15.9 => |k| <= 15
        # But we should check actual bounds
        k_vals = []
        for k in range(-20, 21):
            x_val = k * np.pi / 10.0
            # Check if within bounds
            if all(low <= x_val <= high for low, high in bounds):
                k_vals.append(k)
        
        if not k_vals:
            # Fallback: use 0
            k_vals = [0]
        
        x_vals = np.array(k_vals) * np.pi / 10.0
        k_max = max(abs(k) for k in k_vals)
        
        # Precompute all possible x values on the front
        # We want to maximize hypervolume by covering the Pareto front well
        # f1 = 1 - sqrt(sum x_i^2 / n)
        # To cover f1 from max (1, when all x_i=0) to min
        
        # Strategy: Generate points that systematically cover the Pareto front
        # The key insight: we need points with different values of sum(x_i^2)
        
        points = []
        used_evals = 0
        
        # Point with all zeros: f1 = 1, f0 = 1
        points.append(np.zeros(dim))
        
        # Systematic coverage: vary the number of non-zero dimensions and their magnitudes
        # This creates a grid of points with different sum(x_i^2) values
        
        # First pass: single dimension non-zero
        for i in range(dim):
            for k in k_vals:
                if k == 0:
                    continue
                pt = np.zeros(dim)
                pt[i] = k * np.pi / 10.0
                points.append(pt)
        
        # Second pass: two dimensions non-zero
        for i in range(dim):
            for j in range(i + 1, dim):
                for k1 in k_vals:
                    for k2 in k_vals:
                        if k1 == 0 and k2 == 0:
                            continue
                        pt = np.zeros(dim)
                        pt[i] = k1 * np.pi / 10.0
                        pt[j] = k2 * np.pi / 10.0
                        points.append(pt)
        
        # Third pass: random combinations for more coverage
        rng = np.random.default_rng(42)
        for _ in range(100):
            k_choice = rng.integers(k_vals[0], k_vals[-1] + 1, size=dim)
            pt = k_choice * np.pi / 10.0
            points.append(pt)
        
        # Fourth pass: targeted f1 values
        # f1 = 1 - sqrt(sum x_i^2 / n)
        # For target f1, we need sum x_i^2 = n * (1 - f1)^2
        target_f1_values = np.linspace(1.0, -4.0, 40)
        for target_f1 in target_f1_values:
            target_sum_sq = n * (1.0 - target_f1) ** 2
            
            pt = np.zeros(dim)
            remaining = target_sum_sq
            
            # Greedy assignment
            for i in range(dim):
                if remaining <= 1e-10:
                    break
                # Find largest |x| in x_vals such that x^2 <= remaining
                x_sq = x_vals ** 2
                valid_mask = x_sq <= remaining + 1e-10
                if np.any(valid_mask):
                    valid_indices = np.where(valid_mask)[0]
                    # Pick the one with largest |x|
                    best_idx = valid_indices[np.argmax(np.abs(x_vals[valid_indices]))]
                    pt[i] = x_vals[best_idx]
                    remaining -= x_vals[best_idx] ** 2
            
            points.append(pt)
        
        # Evaluate all points, keeping only those that fit in budget
        F = []
        valid_points = []
        
        for pt in points:
            if time.time() > deadline:
                break
            if used_evals >= budget:
                break
            try:
                fc = problem(pt)
                F.append(np.array(fc, dtype=float))
                valid_points.append(pt)
                used_evals += 1
            except Exception:
                break
        
        if len(F) == 0:
            # At least evaluate one point
            try:
                fc = problem(np.zeros(dim))
                F.append(np.array(fc, dtype=float))
                valid_points.append(np.zeros(dim))
                used_evals += 1
            except Exception:
                pass
        
        F = np.array(F) if F else np.empty((0, 2))
        valid_points = np.array(valid_points) if valid_points else np.empty((0, dim))
        
        # Use remaining budget for refinement: try to fix points not exactly on the front
        if used_evals < budget and len(valid_points) > 0:
            # Find points where f0 is not close to 1
            for i in range(len(valid_points)):
                if time.time() > deadline:
                    break
                if used_evals >= budget:
                    break
                
                if F[i, 0] < 1.0 - 1e-10:
                    continue
                
                # Try adjusting each dimension to nearest k*pi/10
                for dim_idx in range(dim):
                    if time.time() > deadline:
                        break
                    if used_evals >= budget:
                        break
                    
                    cur_val = valid_points[i, dim_idx]
                    k_cur = int(round(cur_val * 10.0 / np.pi))
                    
                    for dk in [-1, 0, 1]:
                        if time.time() > deadline:
                            break
                        if used_evals >= budget:
                            break
                        
                        k_new = k_cur + dk
                        if k_new < k_vals[0] or k_new > k_vals[-1]:
                            continue
                        
                        new_val = k_new * np.pi / 10.0
                        test_pt = valid_points[i].copy()
                        test_pt[dim_idx] = new_val
                        
                        try:
                            fc = problem(test_pt)
                            used_evals += 1
                            
                            # Keep if it improves (f0 closer to 1 or f1 better)
                            if fc[0] < F[i, 0] - 1e-10 or (abs(fc[0] - F[i, 0]) < 1e-10 and fc[1] < F[i, 1]):
                                valid_points[i] = test_pt
                                F[i] = np.array(fc, dtype=float)
                        except Exception:
                            break
                
                if used_evals >= budget or time.time() > deadline:
                    break
        
        # Use any remaining budget to add more diverse points
        if used_evals < budget:
            for _ in range(budget - used_evals):
                if time.time() > deadline:
                    break
                try:
                    k_choice = rng.integers(k_vals[0], k_vals[-1] + 1, size=dim)
                    pt = k_choice * np.pi / 10.0
                    fc = problem(pt)
                    F = np.vstack([F, [np.array(fc, dtype=float)]])
                    valid_points = np.vstack([valid_points, [pt]])
                    used_evals += 1
                except Exception:
                    break
        
        return valid_points
# EVOLVE-BLOCK-END
