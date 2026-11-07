"""Optimizer for BBOB-style Rastrigin f3 with hidden shift/rotation/offset."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart (1+1)-ES with local restarts and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0  # 10s safety margin before 300s limit
        rng = np.random.default_rng(42)

        # Track global best
        best_x = None
        best_f = np.inf
        calls_made = 0
        budget = self.budget

        # Strategy: multi-restart with a mix of deterministic and random starts
        # For Rastrigin, the global min structure has periodicity ~4
        # After hidden shift/rotation, the minimizer is in [-4,4]^5
        
        # Phase 1: Coarse grid to find a good region (uses ~100 evals)
        # Phase 2: (1+1)-ES refinement from best grid point
        # Phase 3: Random restarts for remaining budget

        # Phase 1: Sparse grid search
        # 3 points per dim = 3^5 = 243 evals - too many
        # Use 3 points for 3 dims, 2 for 2 dims = 3*3*3*2*2 = 108
        # Or better: use a small set of strategic points
        
        # Actually, for Rastrigin with 500 budget, let's do:
        # - 30 evals: strategic points (origin, axis-aligned, corners)
        # - 400 evals: multi-restart (1+1)-ES with 8 restarts of 50 evals each
        # - 70 evals: final refinement
        
        # Phase 1: Strategic initial points
        strategic_points = []
        
        # Origin
        strategic_points.append(np.zeros(self.dim))
        
        # Axis-aligned points: for each dim, try +/-2, +/-4
        for d in range(self.dim):
            for val in [-4, -2, 0, 2, 4]:
                p = np.zeros(self.dim)
                p[d] = val
                strategic_points.append(p)
        
        # Some random points
        for _ in range(10):
            strategic_points.append(rng.uniform(-5, 5, self.dim))
        
        # Evaluate strategic points
        for p in strategic_points:
            if calls_made >= budget or time.time() > deadline:
                break
            p_clipped = np.clip(p, -5, 5)
            fp = func(p_clipped)
            calls_made += 1
            if fp < best_f:
                best_f = fp
                best_x = p_clipped
        
        if calls_made >= budget or time.time() > deadline:
            return best_x, best_f
        
        # Phase 2: Multi-restart (1+1)-ES
        # Use remaining budget for 8 restarts
        remaining = budget - calls_made
        n_restarts = 8
        per_restart = remaining // n_restarts
        if per_restart < 10:
            per_restart = 10
            n_restarts = remaining // per_restart
            if n_restarts < 1:
                n_restarts = 1
                per_restart = remaining

        for r in range(n_restarts):
            if calls_made >= budget or time.time() > deadline:
                break
            
            # Choose starting point: best found so far with perturbation, or random
            if r == 0:
                # Start from current best with small perturbation
                x = np.clip(best_x + 0.5 * rng.standard_normal(self.dim), -5, 5)
            elif r < 3:
                # Start near best with moderate perturbation
                x = np.clip(best_x + 2.0 * rng.standard_normal(self.dim), -5, 5)
            else:
                # Random restart
                x = rng.uniform(-5, 5, self.dim)
            
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x
            
            sigma = 1.5
            no_improve = 0
            max_no_improve = 15
            
            for _ in range(per_restart - 1):
                if calls_made >= budget or time.time() > deadline:
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                calls_made += 1
                
                if fy < fx:
                    x, fx = y, fy
                    no_improve = 0
                    sigma = min(sigma * 1.3, 3.0)
                    if fx < best_f:
                        best_f = fx
                        best_x = x
                else:
                    no_improve += 1
                    sigma = max(sigma * 0.85, 0.05)
                
                # Local restart within this restart if stuck too long
                if no_improve >= max_no_improve:
                    # Try a few quick local perturbations
                    found = False
                    for _ in range(5):
                        if calls_made >= budget or time.time() > deadline:
                            break
                        z = np.clip(x + 0.5 * rng.standard_normal(self.dim), -5, 5)
                        fz = func(z)
                        calls_made += 1
                        if fz < fx:
                            x, fx = z, fz
                            sigma = 1.0
                            no_improve = 0
                            if fx < best_f:
                                best_f = fx
                                best_x = x
                            found = True
                            break
                    if not found:
                        # Move to next restart
                        break
        
        if calls_made >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 3: Final refinement from best point
        x = best_x.copy()
        fx = best_f
        sigma = 0.3
        no_improve = 0
        max_no_improve = 20
        
        while calls_made < budget and time.time() < deadline:
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls_made += 1
            
            if fy < fx:
                x, fx = y, fy
                no_improve = 0
                sigma = min(sigma * 1.1, 0.5)
                if fx < best_f:
                    best_f = fx
                    best_x = x
            else:
                no_improve += 1
                sigma = max(sigma * 0.9, 0.01)
            
            if no_improve >= max_no_improve:
                # Try to kick out of local minimum
                y = np.clip(x + 1.0 * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma = 0.5
                    no_improve = 0
                    if fx < best_f:
                        best_f = fx
                        best_x = x
                else:
                    # Give up on this direction, shrink more
                    sigma = max(sigma * 0.8, 0.005)
                    no_improve = 0
        
        return best_x, best_f
# EVOLVE-BLOCK-END
