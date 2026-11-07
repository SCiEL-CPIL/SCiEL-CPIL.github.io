"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer for BBOB f15 (Rastrigin rotated).
    Strategy:
    1. Sobol quasi-random scan for systematic basin coverage.
    2. Adaptive variance reduction: dynamically shift sampling center to
       current best while scaling step size to exploit Rastrigin's known
       local minima periodicity (period ~2.0 in each dimension).
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s limit
        budget = self.budget
        dim = self.dim

        calls_used = 0

        def f(x):
            nonlocal calls_used
            if calls_used >= budget:
                return 1e10
            calls_used += 1
            return func(x)

        # --- Phase 1: Sobol Quasi-Random Scan ---
        def sobol_sequence(n, dim, seed=42):
            try:
                from scipy.stats import qmc
                sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
                points = sampler.random(n)
                return points
            except ImportError:
                def halton(index, base):
                    f = 1.0
                    r = 0.0
                    while index > 0:
                        f /= base
                        r += f * (index % base)
                        index //= base
                    return r
                points = np.zeros((n, dim))
                for d in range(dim):
                    base = d + 2
                    for i in range(n):
                        points[i, d] = halton(i + 1, base)
                return points

        n_sobol = min(40, budget // 12)
        if n_sobol < 10:
            n_sobol = 10

        sobol_points = sobol_sequence(n_sobol, dim)
        sobol_points = sobol_points * 10.0 - 5.0

        candidates = []
        for pt in sobol_points:
            if calls_used >= budget or time.time() > deadline:
                break
            pt_clipped = np.clip(pt, -5, 5)
            fp = f(pt_clipped)
            candidates.append((fp, pt_clipped.copy()))

        if not candidates:
            best_x = np.zeros(dim)
            best_f = f(best_x)
            candidates = [(best_f, best_x)]

        candidates.sort(key=lambda x: x[0])

        global_best_x = candidates[0][1].copy()
        global_best_f = candidates[0][0]

        # --- Phase 2: Adaptive Variance Reduction with Periodicity Exploitation ---
        # Key insight: Rastrigin has local minima at integer points (period 2.0).
        # We dynamically shift the sampling center toward the current best and
        # scale the step size to match the local structure.
        
        rng = np.random.default_rng(12345)
        
        # Initial center: best Sobol point
        center = global_best_x.copy()
        best_f = global_best_f
        
        # Initial step size: moderate, will adapt
        # Rastrigin period is ~2.0, so start with step ~1.0 to explore neighboring basins
        step_size = 1.2
        
        # Success/failure tracking for adaptive step size
        # Track recent improvements to modulate step size
        recent_improvements = []  # last N step outcomes (1=improved, 0=not)
        window_size = 8
        
        # Periodicity-aware perturbation: bias toward integer lattice points
        # since Rastrigin minima are at integers
        def periodicity_bias(point):
            """Nudge point toward nearest integer lattice point with small probability."""
            if rng.random() < 0.3:
                # Snap one random coordinate to nearest integer
                d = rng.integers(0, dim)
                point[d] = np.round(point[d])
            return point
        
        # Main adaptive loop
        while calls_used < budget and time.time() < deadline:
            # Generate candidate around current center
            # Use a mixture: Gaussian perturbation + periodicity bias
            y = center + step_size * rng.standard_normal(dim)
            y = periodicity_bias(y)
            y = np.clip(y, -5, 5)
            
            fy = f(y)
            
            if fy < best_f:
                # Improvement: move center toward the new best
                best_f = fy
                global_best_f = fy
                global_best_x = y.copy()
                
                # Shift center partially toward the improvement (momentum-like)
                center = 0.7 * center + 0.3 * y
                center = np.clip(center, -5, 5)
                
                # Increase step size slightly to explore more broadly
                step_size *= 1.15
                step_size = min(step_size, 4.0)
                
                recent_improvements.append(1)
            else:
                # No improvement: shrink step size to refine locally
                step_size *= 0.92
                step_size = max(step_size, 0.15)
                
                recent_improvements.append(0)
            
            # Trim window
            if len(recent_improvements) > window_size:
                recent_improvements.pop(0)
            
            # Adaptive step size based on success rate in window
            if len(recent_improvements) >= window_size:
                success_rate = sum(recent_improvements) / len(recent_improvements)
                if success_rate > 0.6:
                    # Too many successes: increase exploration
                    step_size *= 1.2
                    step_size = min(step_size, 4.0)
                elif success_rate < 0.2:
                    # Too few successes: decrease to exploit
                    step_size *= 0.85
                    step_size = max(step_size, 0.15)
            
            # Periodic global exploration: every ~30 calls, try a Sobol-like
            # jump to cover other parts of the domain
            if calls_used % 30 == 0 and calls_used < budget - 20:
                # Generate a few random exploratory points
                for _ in range(3):
                    if calls_used >= budget or time.time() > deadline:
                        break
                    x_explore = rng.uniform(-5, 5, dim)
                    x_explore = periodicity_bias(x_explore)
                    x_explore = np.clip(x_explore, -5, 5)
                    f_explore = f(x_explore)
                    if f_explore < best_f:
                        best_f = f_explore
                        global_best_f = f_explore
                        global_best_x = x_explore.copy()
                        center = x_explore.copy()
                        step_size = 1.5

        return global_best_x, global_best_f

# EVOLVE-BLOCK-END
