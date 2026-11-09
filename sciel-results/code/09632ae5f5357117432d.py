"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """(1+1)-ES with adaptive step-size, Nelder-Mead polish, and Sobol exploration
    for BBOB-style Schwefel f20."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 285  # Safety margin for 300s limit

        call_count = [0]
        max_calls = budget

        def counted_func(x):
            if call_count[0] >= max_calls:
                return float('inf')
            call_count[0] += 1
            x = np.clip(x, -5.0, 5.0)
            return float(func(x))

        rng = np.random.default_rng(12345)

        # ---- Sobol sequence generation for global exploration ----
        def sobol_point(idx, dim):
            """Generate a point in [0,1)^dim using a simple Sobol-like hash."""
            p = np.zeros(dim)
            for d in range(dim):
                # Use a deterministic quasi-random sequence based on bit-reversal
                val = idx * (d + 1) + 7
                # Use a simple LCG for reproducibility
                val = (val * 1103515245 + 12345) & 0x7fffffff
                val2 = (val * 1103515245 + 12345) & 0x7fffffff
                p[d] = val2 / 0x7fffffff
            return p

        # Phase 1: Global exploration with Sobol sequence
        # Use first ~30 calls to sample well-distributed points
        n_global = min(30, max_calls // 10)
        
        best_x = None
        best_f = float('inf')

        for i in range(n_global):
            if call_count[0] >= max_calls:
                break
            s = sobol_point(i + 1, dim)
            x = s * 10.0 - 5.0  # Map to [-5, 5]
            f = counted_func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 2: (1+1)-ES with adaptive step size
        if best_x is None:
            best_x = rng.uniform(-5.0, 5.0, dim)
            best_f = counted_func(best_x)

        x = best_x.copy()
        f_x = best_f
        sigma = 0.8  # Initial step size

        # Track success rate with a sliding window
        window_size = 15
        successes = []

        while call_count[0] < max_calls and time.time() < deadline:
            # Generate candidate
            noise = rng.standard_normal(dim) * sigma
            x_trial = x + noise
            x_trial = np.clip(x_trial, -5.0, 5.0)
            
            f_trial = counted_func(x_trial)
            
            if f_trial < f_x:
                x = x_trial
                f_x = f_trial
                best_x = x.copy()
                best_f = f_trial
                successes.append(1)
            else:
                successes.append(0)
            
            # Keep only the last window_size entries
            if len(successes) > window_size:
                successes = successes[-window_size:]
            
            # Compute success rate
            n = len(successes)
            success_rate = sum(successes) / n if n > 0 else 0.5
            
            # Adapt sigma using 1/5 rule
            if success_rate > 0.2:
                sigma *= 1.04
            else:
                sigma /= 1.04
            
            sigma = np.clip(sigma, 1e-6, 8.0)

            # If sigma is very small, we might be converged
            # Try a restart with larger sigma if stuck
            if sigma < 1e-4 and call_count[0] < max_calls - 10:
                # Nudge with a larger random jump
                noise = rng.standard_normal(dim) * 2.0
                x_trial = x + noise
                x_trial = np.clip(x_trial, -5.0, 5.0)
                f_trial = counted_func(x_trial)
                if f_trial < f_x:
                    x = x_trial
                    f_x = f_trial
                    best_x = x.copy()
                    best_f = f_trial
                sigma = 0.5

        # Phase 3: Nelder-Mead polish using remaining budget
        remaining = max_calls - call_count[0]
        if remaining > 10 and time.time() < deadline:
            try:
                from scipy.optimize import minimize
                
                # Set up Nelder-Mead
                x0 = best_x.copy()
                
                # Create initial simplex
                simplex = [x0.copy()]
                for d in range(dim):
                    pt = x0.copy()
                    step = max(0.01, sigma * 0.5)
                    pt[d] += step
                    simplex.append(pt)
                
                # Evaluate initial simplex
                simplex_f = []
                for pt in simplex:
                    pt = np.clip(pt, -5.0, 5.0)
                    f_val = counted_func(pt)
                    simplex_f.append(f_val)
                
                # Manual Nelder-Mead implementation
                # This is a simplified version to avoid scipy overhead
                
                alpha, gamma, rho = 1.0, 2.0, -0.5
                
                nm_iter = 0
                max_nm_iter = min(remaining - 5, 200)
                
                while nm_iter < max_nm_iter and call_count[0] < max_calls and time.time() < deadline:
                    # Sort by function value
                    order = np.argsort(simplex_f)
                    simplex = [simplex[i] for i in order]
                    simplex_f = [simplex_f[i] for i in order]
                    
                    # Compute centroid of all but worst
                    centroid = np.mean(simplex[:-1], axis=0)
                    
                    # Reflection
                    x_ref = centroid + alpha * (centroid - simplex[-1])
                    x_ref = np.clip(x_ref, -5.0, 5.0)
                    f_ref = counted_func(x_ref)
                    
                    if f_ref < simplex_f[0]:
                        # Expansion
                        x_exp = centroid + gamma * (x_ref - centroid)
                        x_exp = np.clip(x_exp, -5.0, 5.0)
                        f_exp = counted_func(x_exp)
                        if f_exp < f_ref:
                            simplex[-1] = x_exp
                            simplex_f[-1] = f_exp
                        else:
                            simplex[-1] = x_ref
                            simplex_f[-1] = f_ref
                    elif f_ref < simplex_f[-2]:
                        # Accept reflection
                        simplex[-1] = x_ref
                        simplex_f[-1] = f_ref
                    else:
                        # Contraction
                        x_con = centroid + rho * (simplex[-1] - centroid)
                        x_con = np.clip(x_con, -5.0, 5.0)
                        f_con = counted_func(x_con)
                        if f_con < simplex_f[-1]:
                            simplex[-1] = x_con
                            simplex_f[-1] = f_con
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = (simplex[0] + simplex[i]) / 2.0
                                simplex_f[i] = counted_func(simplex[i])
                    
                    nm_iter += 1
                    if call_count[0] >= max_calls:
                        break
                    
                    # Update best
                    min_idx = np.argmin(simplex_f)
                    if simplex_f[min_idx] < best_f:
                        best_f = simplex_f[min_idx]
                        best_x = simplex[min_idx].copy()
                    
                    # Check convergence
                    spread = np.max(simplex_f) - np.min(simplex_f)
                    if spread < 1e-8:
                        break

            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
