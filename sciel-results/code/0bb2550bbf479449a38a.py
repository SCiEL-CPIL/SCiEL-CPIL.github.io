"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-scale coordinate descent with time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        
        x = np.zeros(self.dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        calls = 1
        
        def try_update(xcand, fcand):
            nonlocal best_x, best_f, calls
            if fcand < best_f:
                best_x = xcand.copy()
                best_f = fcand
            calls += 1
        
        # Phase 1: Coarse grid scan to find a good basin
        # Scan each dimension with geometrically spaced points
        coarse_steps = [4.0, 3.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.0625]
        
        for step in coarse_steps:
            for d in range(self.dim):
                # Try positive and negative offsets from current best
                for sign in [1.0, -1.0]:
                    if calls >= self.budget or time.time() >= deadline:
                        return best_x, best_f
                    xcand = best_x.copy()
                    xcand[d] += sign * step
                    xcand = np.clip(xcand, -5, 5)
                    fcand = func(xcand)
                    try_update(xcand, fcand)
        
        # Phase 2: Fine coordinate descent with shrinking steps
        fine_steps = [0.05, 0.025, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
        
        for step in fine_steps:
            for d in range(self.dim):
                # Try small perturbations in each dimension
                for sign in [1.0, -1.0]:
                    if calls >= self.budget or time.time() >= deadline:
                        return best_x, best_f
                    xcand = best_x.copy()
                    xcand[d] += sign * step
                    xcand = np.clip(xcand, -5, 5)
                    fcand = func(xcand)
                    try_update(xcand, fcand)
        
        # Phase 3: Very fine refinement
        very_fine_steps = [0.00005, 0.00002, 0.00001]
        
        for step in very_fine_steps:
            for d in range(self.dim):
                for sign in [1.0, -1.0]:
                    if calls >= self.budget or time.time() >= deadline:
                        return best_x, best_f
                    xcand = best_x.copy()
                    xcand[d] += sign * step
                    xcand = np.clip(xcand, -5, 5)
                    fcand = func(xcand)
                    try_update(xcand, fcand)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
