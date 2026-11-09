"""Rewritten optimizer: deterministic multi-scale coordinate descent with time guard."""
import time
import numpy as np


class Optimizer:
    """Deterministic multi-scale coordinate descent for Büche-Rastrigin (f4)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 280.0  # safety margin before 300s kill

        dim = self.dim
        budget = self.budget

        # Phase 1: Structured scan to find a good starting region
        # Use a more comprehensive set of points: corners, center, face centers, edge midpoints
        scan_points = []
        
        # Corners
        for i in range(2 ** dim):
            pt = np.array([[-5.0, 5.0][(i >> d) & 1] for d in range(dim)])
            scan_points.append(pt)
        
        # Center
        scan_points.append(np.zeros(dim))
        
        # Face centers (one axis at ±3, others at 0)
        for d in range(dim):
            for val in [-3.0, 3.0]:
                pt = np.zeros(dim)
                pt[d] = val
                scan_points.append(pt)
        
        # Edge midpoints (two axes non-zero)
        for d1 in range(dim):
            for d2 in range(d1 + 1, dim):
                for v1 in [-2.5, 2.5]:
                    for v2 in [-2.5, 2.5]:
                        pt = np.zeros(dim)
                        pt[d1] = v1
                        pt[d2] = v2
                        scan_points.append(pt)
        
        # Limit scan to reasonable number
        max_scan = 30
        scan_points = scan_points[:max_scan]

        best_x = None
        best_f = np.inf
        calls_used = 0

        for sp in scan_points:
            if calls_used >= budget:
                break
            if time.time() > deadline - 5.0:
                break
            f_val = func(sp)
            calls_used += 1
            if f_val < best_f:
                best_f = f_val
                best_x = sp.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_used += 1

        # Phase 2: Estimate local gradient direction to account for rotation
        # Use finite differences to estimate gradient at best point
        eps = 0.05
        grad = np.zeros(dim)
        for d in range(dim):
            if calls_used >= budget - 2:
                break
            if time.time() > deadline - 5.0:
                break
            x_plus = best_x.copy()
            x_plus[d] += eps
            x_plus = np.clip(x_plus, -5.0, 5.0)
            f_plus = func(x_plus)
            calls_used += 1
            
            x_minus = best_x.copy()
            x_minus[d] -= eps
            x_minus = np.clip(x_minus, -5.0, 5.0)
            f_minus = func(x_minus)
            calls_used += 1
            
            grad[d] = (f_plus - f_minus) / (2 * eps)

        # Normalize gradient direction
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-10:
            grad_dir = grad / grad_norm
        else:
            grad_dir = None

        x = best_x.copy()
        fx = best_f

        # Phase 3: Multi-scale descent using both coordinate and gradient directions
        # Geometrically shrinking step sizes
        base_steps = [4.0, 2.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]

        for step in base_steps:
            if calls_used >= budget:
                break
            if time.time() > deadline - 5.0:
                break
            
            improved = True
            max_iters = 3  # Limit iterations per step size to control budget
            while improved and calls_used < budget and max_iters > 0:
                if time.time() > deadline - 5.0:
                    break
                improved = False
                max_iters -= 1
                
                # Try coordinate directions
                for d in range(dim):
                    if calls_used >= budget:
                        break
                    if time.time() > deadline - 5.0:
                        break
                    for direction in [1.0, -1.0]:
                        if calls_used >= budget:
                            break
                        if time.time() > deadline - 5.0:
                            break
                        x_new = x.copy()
                        x_new[d] += direction * step
                        x_new = np.clip(x_new, -5.0, 5.0)
                        f_new = func(x_new)
                        calls_used += 1
                        if f_new < fx:
                            fx = f_new
                            x = x_new
                            improved = True
                
                # Try gradient direction if available
                if grad_dir is not None and calls_used < budget:
                    if time.time() > deadline - 5.0:
                        break
                    for direction in [1.0, -1.0]:
                        if calls_used >= budget:
                            break
                        if time.time() > deadline - 5.0:
                            break
                        x_new = x + direction * step * grad_dir
                        x_new = np.clip(x_new, -5.0, 5.0)
                        f_new = func(x_new)
                        calls_used += 1
                        if f_new < fx:
                            fx = f_new
                            x = x_new
                            improved = True

        # Phase 4: Fine-tuning with very small steps
        if calls_used < budget:
            fine_steps = [0.0002, 0.0001, 0.00005, 0.00002]
            for step in fine_steps:
                if calls_used >= budget:
                    break
                if time.time() > deadline - 5.0:
                    break
                for d in range(dim):
                    if calls_used >= budget:
                        break
                    if time.time() > deadline - 5.0:
                        break
                    for direction in [1.0, -1.0]:
                        if calls_used >= budget:
                            break
                        if time.time() > deadline - 5.0:
                            break
                        x_new = x.copy()
                        x_new[d] += direction * step
                        x_new = np.clip(x_new, -5.0, 5.0)
                        f_new = func(x_new)
                        calls_used += 1
                        if f_new < fx:
                            fx = f_new
                            x = x_new

        return x, fx
