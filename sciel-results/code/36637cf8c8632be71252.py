"""Optimizer for BBOB f24 (Lunacek bi-Rastrigin) using DFCD multi-resolution grid scan."""
import numpy as np
import time


class Optimizer:
    """Derivative-Free Coordinate Descent with multi-resolution grid scan."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf
        self.deadline = None

    def __call__(self, func):
        self.deadline = time.time() + 290.0  # Safety margin before 330s limit
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        # Phase 1: Coarse grid scan to find good regions
        # Lunacek has two minima; use a coarse grid to locate them
        # Start with a 3x3x3x3x3 grid (243 points) - too many for budget 500
        # Use adaptive: start with fewer points, refine based on results
        
        # Initial: 5 points per dim = 3125, too many
        # Use 3 points per dim = 243, still a lot
        # Use 2 points per dim = 32 for very coarse, then refine
        
        # Strategy: 
        # 1. Evaluate 2^5 = 32 corner-ish points (coarse)
        # 2. Identify best region, do 3^5 = 243 points in that region (medium)
        # 3. Refine with 4^5 = 1024? No, too many. Use 3^5 in smaller region
        # 4. Final polish with coordinate descent
        
        # Actually, let's be smarter:
        # Phase 1: 2 points per dim = 32 evaluations
        # Phase 2: Around best, 3 points per dim in local region = 243 evaluations  
        # Phase 3: Coordinate descent refinement with ~200 evaluations
        # Total: ~475, leaving ~25 for final polish
        
        # Phase 1: Coarse 2-point grid per dimension
        coarse_points = []
        for i in range(2 ** n):
            point = np.array([(i >> d) & 1 for d in range(n)], dtype=float)
            # Map 0,1 to -4, 4 (avoiding boundaries)
            point = point * 8 - 4
            coarse_points.append(point)
        
        for x0 in coarse_points:
            if self.calls >= budget - 300 or time.time() > self.deadline:
                break
            x0 = np.clip(x0, -5, 5)
            fx = func(x0)
            self.calls += 1
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x0.copy()

        if self.best_x is None or time.time() > self.deadline:
            return self.best_x, self.best_f

        # Phase 2: Medium resolution around best point
        # Use 3 points per dim in a local region around best
        center = self.best_x.copy()
        local_radius = 2.0  # Search in [-2, 2] around center
        
        medium_points = []
        for i in range(3 ** n):
            point = np.array([(i >> (d * 2)) & 3 for d in range(n)], dtype=float)
            # Map 0,1,2 to -1, 0, 1
            point = (point - 1) * local_radius
            point = center + point
            medium_points.append(point)
        
        for x0 in medium_points:
            if self.calls >= budget - 100 or time.time() > self.deadline:
                break
            x0 = np.clip(x0, -5, 5)
            fx = func(x0)
            self.calls += 1
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x0.copy()

        if time.time() > self.deadline:
            return self.best_x, self.best_f

        # Phase 3: Coordinate descent refinement
        # Iterate over dimensions, trying small perturbations
        x = self.best_x.copy()
        best_f_cd = self.best_f
        step_sizes = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
        
        for step in step_sizes:
            if self.calls >= budget or time.time() > self.deadline:
                break
            
            improved = True
            while improved and self.calls < budget and time.time() < self.deadline:
                improved = False
                for d in range(n):
                    if self.calls >= budget or time.time() > self.deadline:
                        break
                    
                    # Try positive direction
                    x_plus = x.copy()
                    x_plus[d] += step
                    if x_plus[d] <= 5:
                        x_plus[d] = np.clip(x_plus[d], -5, 5)
                        fp = func(x_plus)
                        self.calls += 1
                        if fp < best_f_cd:
                            best_f_cd = fp
                            x = x_plus.copy()
                            improved = True
                    
                    # Try negative direction
                    if self.calls >= budget or time.time() > self.deadline:
                        break
                    x_minus = x.copy()
                    x_minus[d] -= step
                    if x_minus[d] >= -5:
                        x_minus[d] = np.clip(x_minus[d], -5, 5)
                        fm = func(x_minus)
                        self.calls += 1
                        if fm < best_f_cd:
                            best_f_cd = fm
                            x = x_minus.copy()
                            improved = True
            
            if best_f_cd < self.best_f:
                self.best_f = best_f_cd
                self.best_x = x.copy()

        # Phase 4: Final local refinement with very small steps
        if self.calls < budget and time.time() < self.deadline:
            x = self.best_x.copy()
            best_f_final = self.best_f
            fine_steps = [0.005, 0.002, 0.001, 0.0005, 0.0002]
            
            for step in fine_steps:
                if self.calls >= budget or time.time() > self.deadline:
                    break
                
                for _ in range(3):  # Multiple passes per step size
                    if self.calls >= budget or time.time() > self.deadline:
                        break
                    improved = True
                    while improved and self.calls < budget and time.time() < self.deadline:
                        improved = False
                        for d in range(n):
                            if self.calls >= budget or time.time() > self.deadline:
                                break
                            
                            for direction in [1, -1]:
                                if self.calls >= budget or time.time() > self.deadline:
                                    break
                                x_trial = x.copy()
                                x_trial[d] += direction * step
                                x_trial[d] = np.clip(x_trial[d], -5, 5)
                                ft = func(x_trial)
                                self.calls += 1
                                if ft < best_f_final:
                                    best_f_final = ft
                                    x = x_trial.copy()
                                    improved = True
            
            if best_f_final < self.best_f:
                self.best_f = best_f_final
                self.best_x = x.copy()

        return self.best_x, self.best_f
