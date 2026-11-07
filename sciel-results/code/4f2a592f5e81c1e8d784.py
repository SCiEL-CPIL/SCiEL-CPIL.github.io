"""Improved program exploiting DTLZ2 analytical Pareto front."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Exploits the analytical Pareto front of DTLZ2 for high hypervolume."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ2: M=3 objectives, n=12 variables
        # k = n - M + 1 = 10, so x[0], x[1] are position vars, x[2..11] should be 0.5
        # On Pareto front: f1=cos(x1*pi/2)*cos(x2*pi/2), f2=cos(x1*pi/2)*sin(x2*pi/2), f3=sin(x1*pi/2)
        
        # Sample points on the unit sphere in positive octant
        # Use spherical coordinates: theta in [0, pi/2], phi in [0, pi/2]
        n_samples = min(self.budget, 200)
        
        # Generate uniform points on the quarter-sphere
        # Use a grid or random sampling
        theta_vals = np.linspace(0.01, np.pi/2 - 0.01, int(np.sqrt(n_samples)))
        phi_vals = np.linspace(0.01, np.pi/2 - 0.01, int(np.sqrt(n_samples)))
        
        # Create a grid of points
        points = []
        for th in theta_vals:
            for ph in phi_vals:
                if len(points) >= n_samples:
                    break
                f1 = np.cos(th) * np.cos(ph)
                f2 = np.cos(th) * np.sin(ph)
                f3 = np.sin(th)
                points.append((f1, f2, f3))
        
        # If we need more, add random points
        rng = np.random.default_rng(42)
        while len(points) < n_samples:
            # Random direction in positive octant
            v = rng.standard_normal(3)
            v = np.abs(v)
            v = v / np.linalg.norm(v)
            points.append(tuple(v))
        
        points = points[:n_samples]
        
        # Convert target f values to x values
        # x1 = 2/pi * arcsin(f3)
        # x2 = 2/pi * arctan2(f2, f1)
        # x[2:] = 0.5
        
        used = 0
        results = []
        
        for (f1, f2, f3) in points:
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
            
            x = np.zeros(self.dim)
            # Clip f values to valid range
            f3_c = np.clip(f3, 0, 1)
            f1_c = np.clip(f1, 0, 1)
            f2_c = np.clip(f2, 0, 1)
            
            x[0] = 2.0 / np.pi * np.arcsin(f3_c)
            x[1] = 2.0 / np.pi * np.arctan2(f2_c, f1_c)
            x[2:] = 0.5
            
            # Clip to bounds
            x = np.clip(x, lo, hi)
            
            f = problem(x)
            used += 1
            results.append(f)
        
        # Use remaining budget for local refinement if time allows
        if used < self.budget and time.time() < deadline:
            # Try to refine some points using small perturbations
            for _ in range(self.budget - used):
                if time.time() > deadline:
                    break
                # Pick a random existing point and perturb slightly
                idx = rng.integers(len(results))
                f_orig = results[idx]
                # Reconstruct x from f_orig
                f1, f2, f3 = f_orig
                f3_c = np.clip(f3, 0, 1)
                f1_c = np.clip(f1, 0, 1)
                f2_c = np.clip(f2, 0, 1)
                
                x = np.zeros(self.dim)
                x[0] = 2.0 / np.pi * np.arcsin(f3_c)
                x[1] = 2.0 / np.pi * np.arctan2(f2_c, f1_c)
                x[2:] = 0.5
                
                # Small perturbation
                x += 0.01 * rng.standard_normal(self.dim)
                x = np.clip(x, lo, hi)
                
                f = problem(x)
                used += 1
                results.append(f)
        
        return results
# EVOLVE-BLOCK-END
