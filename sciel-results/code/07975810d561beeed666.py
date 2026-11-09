"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Deterministic coarse-to-fine search using Halton sequence with local refinement."""

    @staticmethod
    def _halton_sequence_fast(n, dim):
        """Faster Halton sequence generation using vectorized operations."""
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        base = primes[:dim]
        points = np.zeros((n, dim))
        for d in range(dim):
            b = base[d]
            idx = np.arange(1, n + 1)
            f = 1.0 / b
            frac = np.zeros(n)
            i_val = idx.copy()
            while np.any(i_val > 0):
                frac += f * (i_val % b)
                i_val //= b
                f /= b
            points[:, d] = frac
        return points

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.low = -5.0
        self.high = 5.0

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        low, high = self.low, self.high

        calls_made = 0
        best_f = np.inf
        best_x = None
        second_best_f = np.inf
        second_best_x = None

        # Phase 1: Global Coarse Search
        # Use 60 points to cover the space well.
        n_global = 60
        halton_global = self._halton_sequence_fast(n_global, dim)
        x_global = low + (high - low) * halton_global

        for i in range(n_global):
            if calls_made >= budget:
                break
            x = x_global[i]
            fx = func(x)
            calls_made += 1
            
            if fx < best_f:
                if best_f < np.inf:
                    second_best_f = best_f
                    second_best_x = best_x.copy()
                best_f = fx
                best_x = x.copy()
            elif fx < second_best_f:
                second_best_f = fx
                second_best_x = x.copy()

        # Phase 2: Local Refinement with Halton and Restarts
        # We will use the remaining budget for local search.
        # Strategy:
        # 1. Maintain a 'current_center' and 'radius'.
        # 2. Sample Halton points in a box around current_center.
        # 3. If improvement found, shrink radius.
        # 4. If no improvement for a while, shrink radius or restart at second_best.

        current_center = best_x.copy()
        radius = 1.0
        no_improve_count = 0
        local_batch_size = 10
        max_no_improve = 30 # 3 batches of 10
        
        while calls_made < budget:
            # Generate local points
            n_local = min(local_batch_size, budget - calls_made)
            if n_local <= 0:
                break
            
            # Generate Halton in [0,1]^dim and map to [center-radius, center+radius]^dim
            h = self._halton_sequence_fast(n_local, dim)
            x_local = current_center + radius * (2 * h - 1)
            x_local = np.clip(x_local, low, high)
            
            improved_in_batch = False
            for i in range(n_local):
                if calls_made >= budget:
                    break
                x = x_local[i]
                fx = func(x)
                calls_made += 1
                
                if fx < best_f:
                    # Update global best
                    best_f = fx
                    best_x = x.copy()
                    current_center = x.copy()
                    radius *= 0.7 # Shrink search area
                    improved_in_batch = True
                    no_improve_count = 0
                elif fx < second_best_f:
                    # Update second best (useful for restarts)
                    second_best_f = fx
                    second_best_x = x.copy()

            if not improved_in_batch:
                no_improve_count += n_local
                if no_improve_count >= max_no_improve:
                    # Restart logic:
                    if radius > 0.1:
                        # Focus more tightly
                        radius *= 0.5
                    else:
                        # Stuck in local min, try second best
                        if second_best_x is not None:
                            current_center = second_best_x.copy()
                            radius = 1.0
                        else:
                            # No second best, just expand slightly
                            radius = min(radius * 1.5, 5.0)
                    no_improve_count = 0
            else:
                # If we improved, we already handled radius shrinkage inside the loop
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
