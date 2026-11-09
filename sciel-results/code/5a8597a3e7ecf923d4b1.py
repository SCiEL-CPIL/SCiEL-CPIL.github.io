"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Combined optimizer: Halton coarse-to-fine + adaptive (1+1)-ES with restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    @staticmethod
    def _halton(index, base):
        result = 0.0
        f = 1.0
        i = index
        while i > 0:
            f /= base
            result += f * (i % base)
            i //= base
        return result

    def _halton_sequence(self, count, dim):
        points = np.zeros((count, dim))
        for d in range(dim):
            for i in range(count):
                points[i, d] = self._halton(i + 1, d + 2)
        return points

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        call_count = 0
        dim = self.dim
        budget = self.budget

        def evaluate(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return None
            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Halton coarse-to-fine search
        # Use first 150 calls for systematic coverage
        halton_budget = min(150, budget)
        halton_points = self._halton_sequence(halton_budget, dim)
        # Map [0,1]^dim to [-5,5]^dim
        halton_points = halton_points * 10.0 - 5.0
        
        for i in range(halton_budget):
            if time.time() > deadline:
                break
            evaluate(halton_points[i])

        # Phase 2: Adaptive (1+1)-ES with restarts
        remaining_budget = budget - call_count
        if remaining_budget <= 0:
            return best_x, best_f

        # Initialize from best found so far
        x = best_x.copy()
        fx = best_f
        sigma = 1.0
        no_improve_count = 0
        max_no_improve = 20
        restarts = 0
        max_restarts = 3

        while call_count < budget:
            if time.time() > deadline:
                break
            
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = evaluate(y)
            if fy is None:
                break
                
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.1
                no_improve_count = 0
            else:
                sigma *= 0.95
                no_improve_count += 1
                
            if no_improve_count >= max_no_improve:
                if restarts < max_restarts:
                    # Restart from a new random point or best found
                    if rng.random() < 0.5:
                        x = rng.uniform(-5, 5, dim)
                        fx = evaluate(x)
                        if fx is None:
                            break
                    else:
                        x = best_x.copy()
                        fx = best_f
                    sigma = 2.0
                    no_improve_count = 0
                    restarts += 1
                else:
                    # Try smaller sigma
                    sigma *= 0.5
                    no_improve_count = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
