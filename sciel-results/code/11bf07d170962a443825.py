"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """
    Hybrid optimizer for BBOB f14 (Different Powers) in 5D.
    Phase 1: Random sampling to find a good basin (10% budget).
    Phase 2: Nelder-Mead simplex method for robust convergence.
    Phase 3: Local coordinate-axis search for final polish.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        calls = 0

        # Wrapper to count calls and handle clipping
        def _eval(x):
            nonlocal calls
            if calls >= budget:
                raise RuntimeError("Budget exceeded")
            x_clipped = np.clip(x, -5.0, 5.0)
            calls += 1
            return func(x_clipped)

        rng = np.random.default_rng(42)
        
        # --- Phase 1: Global Random Search ---
        # Use 24 samples to find a good starting point
        n_global = 24
        
        x_best = None
        f_best = np.inf
        
        for i in range(n_global):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            try:
                f = _eval(x)
            except Exception:
                break
            if f < f_best:
                f_best = f
                x_best = x.copy()
                
        if x_best is None:
            x_best = np.zeros(dim)
            try:
                f_best = _eval(x_best)
            except Exception:
                return x_best, np.inf

        # --- Phase 2: Nelder-Mead ---
        # Initialize simplex with adaptive delta
        # Estimate scale from initial samples if possible, otherwise use 0.5
        delta = 0.5
        simplex = np.zeros((dim + 1, dim))
        f_values = np.zeros(dim + 1)
        
        simplex[0] = x_best
        f_values[0] = f_best
        
        for i in range(1, dim + 1):
            simplex[i] = x_best.copy()
            simplex[i][i-1] += delta
            simplex[i] = np.clip(simplex[i], -5, 5)
            try:
                f_values[i] = _eval(simplex[i])
            except Exception:
                f_values[i] = np.inf
            if time.time() > deadline:
                break

        # NM Parameters
        alpha = 1.0   # Reflection
        gamma = 2.0   # Expansion
        rho = 0.5     # Contraction
        sigma_shrink = 0.5   # Shrink
        
        while calls < budget:
            if time.time() > deadline:
                break
            
            # Sort simplex by function value
            order = np.argsort(f_values)
            simplex = simplex[order]
            f_values = f_values[order]
            
            x_worst = simplex[-1]
            x_best_nm = simplex[0]
            x_centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = x_centroid + alpha * (x_centroid - x_worst)
            x_r = np.clip(x_r, -5, 5)
            try:
                f_r = _eval(x_r)
            except Exception:
                f_r = np.inf
            
            if f_r < f_values[0]:
                # Expansion
                x_e = x_centroid + gamma * (x_r - x_centroid)
                x_e = np.clip(x_e, -5, 5)
                try:
                    f_e = _eval(x_e)
                except Exception:
                    f_e = np.inf
                if f_e < f_r:
                    simplex[-1] = x_e
                    f_values[-1] = f_e
                else:
                    simplex[-1] = x_r
                    f_values[-1] = f_r
            elif f_r < f_values[-2]:
                simplex[-1] = x_r
                f_values[-1] = f_r
            else:
                # Contraction
                if f_r < f_values[-1]:
                    x_c = x_centroid + rho * (x_r - x_centroid)
                else:
                    x_c = x_centroid - rho * (x_r - x_centroid)
                x_c = np.clip(x_c, -5, 5)
                try:
                    f_c = _eval(x_c)
                except Exception:
                    f_c = np.inf
                
                if f_c < min(f_r, f_values[-1]):
                    simplex[-1] = x_c
                    f_values[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = x_best_nm + sigma_shrink * (simplex[i] - x_best_nm)
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        try:
                            f_values[i] = _eval(simplex[i])
                        except Exception:
                            f_values[i] = np.inf
                        if calls >= budget:
                            break
                    if calls >= budget:
                        break

            # Update global best
            current_best_f = np.min(f_values)
            if current_best_f < f_best:
                f_best = current_best_f
                x_best = simplex[np.argmin(f_values)].copy()
            
            # Convergence check
            if np.max(f_values) - np.min(f_values) < 1e-12:
                break

        # --- Phase 3: Adaptive Multi-scale Local Polish ---
        # Use remaining budget for adaptive local search
        # Start with larger steps and shrink only when no improvement is found
        step_size = 0.5
        min_step = 1e-8
        while calls < budget:
            if time.time() > deadline:
                break
            improved = False
            for d in range(dim):
                for direction in [1, -1]:
                    if calls >= budget or time.time() > deadline:
                        break
                    x_trial = x_best.copy()
                    x_trial[d] += direction * step_size
                    x_trial = np.clip(x_trial, -5, 5)
                    try:
                        f_trial = _eval(x_trial)
                    except Exception:
                        continue
                    if f_trial < f_best:
                        f_best = f_trial
                        x_best = x_trial
                        improved = True
                    if calls >= budget:
                        break
                if calls >= budget:
                    break
            if not improved:
                step_size *= 0.5
                if step_size < min_step:
                    break
            else:
                # If improved, try slightly larger steps to explore more
                step_size = min(step_size * 1.5, 2.0)

        return x_best, f_best
# EVOLVE-BLOCK-END
