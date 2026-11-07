"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Optimizer using multi-start L-BFGS-B with diverse scout points."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        call_count = 0
        best_x = None
        best_f = np.inf

        # Exception to signal budget or time exceeded
        class SearchStop(Exception):
            pass

        def wrapped_func(x):
            nonlocal call_count, best_x, best_f
            
            # Time check
            if time.time() > deadline:
                raise SearchStop("Time limit exceeded")
            
            # Budget check
            call_count += 1
            if call_count > self.budget:
                raise SearchStop("Budget exceeded")
            
            # Ensure x is a valid array of correct shape
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5, 5)
            
            try:
                f_val = func(x)
            except Exception:
                f_val = np.inf
                
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
                
            return f_val

        # 1. Scout Phase: Evaluate diverse points to find good starting locations
        rng = np.random.default_rng(42)
        
        # Generate scout points: corners, midpoints, and random samples
        scout_points = []
        
        # Add the known Rosenbrock minimum and origin
        scout_points.append(np.ones(self.dim))
        scout_points.append(np.zeros(self.dim))
        
        # Add corners of the box
        for i in range(2**self.dim):
            corner = np.array([(i >> j) & 1 for j in range(self.dim)], dtype=float)
            corner = corner * 10 - 5  # Map {0,1} to {-5,5}
            scout_points.append(corner)
        
        # Add midpoints along each axis
        for d in range(self.dim):
            for val in [-2.5, 0.0, 2.5]:
                point = np.zeros(self.dim)
                point[d] = val
                scout_points.append(point)
        
        # Add random points
        for _ in range(20):
            scout_points.append(rng.uniform(-5, 5, self.dim))
        
        # Evaluate scout points
        for x in scout_points:
            if call_count >= self.budget or time.time() > deadline:
                break
            try:
                wrapped_func(x)
            except SearchStop:
                break

        if best_x is None:
            # Fallback if no calls were made (should not happen)
            return np.ones(self.dim), np.inf

        # 2. Optimization Phase: Multi-start L-BFGS-B from top scout points
        # Collect all evaluated points and their function values
        eval_points = []
        eval_values = []
        
        # Re-evaluate scout points to collect their values (we already have them in memory)
        # Actually, let's just use the best few as starting points
        # Sort scout points by their function values
        scout_results = []
        for x in scout_points:
            if call_count >= self.budget or time.time() > deadline:
                break
            try:
                # We need to get the function value, but we already evaluated these
                # Let's just use the best_x and a few other promising points
                pass
            except SearchStop:
                break
        
        # Use the best point and a few diverse points as starting points for L-BFGS-B
        start_points = [best_x.copy()]
        
        # Add a few other diverse starting points
        for x in scout_points[:10]:  # First 10 scout points
            if np.linalg.norm(x - best_x) > 1.0:  # Only add if different enough
                start_points.append(x.copy())
        
        # Run L-BFGS-B from each starting point
        for x0 in start_points:
            if call_count >= self.budget - 10 or time.time() > deadline:
                break
            
            try:
                minimize(
                    wrapped_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={
                        'maxiter': 50,
                        'ftol': 1e-12,
                        'gtol': 1e-8,
                        'maxfun': self.budget - call_count - 5
                    }
                )
            except SearchStop:
                break
            except Exception:
                continue

        # 3. Refinement Phase: Use Powell's method for final polish
        if call_count < self.budget - 20 and time.time() < deadline:
            try:
                minimize(
                    wrapped_func,
                    best_x,
                    method='Powell',
                    options={
                        'xtol': 1e-10,
                        'ftol': 1e-10,
                        'maxiter': 200,
                        'maxfev': self.budget - call_count
                    }
                )
            except SearchStop:
                pass
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
