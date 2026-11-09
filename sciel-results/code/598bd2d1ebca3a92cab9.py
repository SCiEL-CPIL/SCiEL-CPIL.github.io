import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _halton(self, n, base):
        """
        Generate the first n points of the Halton sequence in base 'base'.
        Returns an array of shape (n, 1).
        """
        result = np.zeros(n)
        f = 1.0
        for i in range(1, n + 1):
            f /= base
            result[i - 1] = f * (i / base)
            # Wait, the standard radical inverse algorithm is:
            # result = 0
            # f = 1/base
            # i = n
            # while i > 0:
            #   result += f * (i % base)
            #   i //= base
            #   f /= base
        # Let's implement the correct radical inverse.
        
    def _radical_inverse(self, i, base):
        result = 0.0
        f = 1.0 / base
        while i > 0:
            result += f * (i % base)
            i //= base
            f /= base
        return result

    def _halton_sequence(self, n, dim):
        """
        Generate n points in dim dimensions using Halton sequence.
        """
        # First 'dim' prime numbers
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        points = np.zeros((n, dim))
        for i in range(n):
            for j in range(dim):
                points[i, j] = self._radical_inverse(i + 1, primes[j])
        return points

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0 # 10s safety margin
        
        # Phase 1: Global Search with Halton Sequence
        n_global = 100
        halton_points = self._halton_sequence(n_global, self.dim)
        # Map [0, 1] to [-5, 5]
        halton_points = halton_points * 10.0 - 5.0
        
        best_x = None
        best_f = float('inf')
        
        for i in range(n_global):
            if time.time() > deadline:
                break
            x = halton_points[i]
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
        # Phase 2: Local Refinement
        # Use remaining budget
        used_budget = n_global
        remaining_budget = self.budget - used_budget
        
        if remaining_budget > 0 and best_x is not None:
            # Use a (1+1)-ES for local refinement
            # Initialize sigma
            sigma = 0.5
            x = best_x
            fx = best_f
            
            # We can use a simple (1+1)-ES
            # Or scipy minimize. Let's use scipy minimize with Nelder-Mead
            # as it's robust and handles the "box" via the func clipping implicitly
            # if we don't pass bounds, but func clips. 
            # However, Nelder-Mead might take many iterations. 
            # A controlled (1+1)-ES is often better for fixed budgets.
            
            # Let's stick to a robust (1+1)-ES with adaptive sigma
            # Similar to baseline but starting from a much better point.
            
            rng = np.random.default_rng(42)
            for _ in range(remaining_budget):
                if time.time() > deadline:
                    break
                
                # Adaptive sigma based on success (1/5 rule simplified)
                # Just use a simple decay/growth
                y = x + sigma * rng.standard_normal(self.dim)
                # Clip y to box to be safe, though func does it
                y = np.clip(y, -5, 5)
                fy = func(y)
                
                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.1 # Increase step size slightly on success
                else:
                    sigma *= 0.9 # Decrease step size on failure
                    # Ensure sigma doesn't get too small
                    if sigma < 1e-4:
                        sigma = 1e-4
                        
        return best_x, best_f
