"""Rastrigin-aware optimizer: periodic structure exploitation + adaptive local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Exploit Rastrigin periodicity with structured exploration and adaptive Powell refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin for 330s limit

        rng = np.random.default_rng(42)
        
        calls_made = 0
        best_x = np.zeros(dim)
        best_f = np.inf
        
        # Store top candidates for multi-restart
        candidates = []
        
        def counted_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            calls_made += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            # Keep top 10 candidates
            candidates.append((fx, x.copy()))
            candidates.sort(key=lambda t: t[0])
            if len(candidates) > 10:
                candidates = candidates[:10]
            return fx

        # Phase 1: Structured exploration exploiting Rastrigin periodicity
        # Rastrigin has period 2*pi ≈ 6.283, local minima at x_i = k*2*pi
        # The search box [-5,5]^5 covers roughly one period in each dimension.
        
        # Try origin first (strong prior for unshifted Rastrigin)
        counted_func(np.zeros(dim))
        
        # Sample structured points: mix of uniform, Gaussian-biased, and periodic-aware
        n_phase1 = 120
        for i in range(1, n_phase1 + 1):
            if time.time() > deadline or calls_made >= budget:
                break
            if i % 4 == 0:
                # Uniform in [-4.5, 4.5]
                x = rng.uniform(-4.5, 4.5, dim)
            elif i % 4 == 1:
                # Biased toward 0 with moderate spread
                x = rng.normal(0, 1.2, dim)
                x = np.clip(x, -4.5, 4.5)
            elif i % 4 == 2:
                # Periodic-aware: near multiples of pi
                base = rng.choice([-np.pi, 0, np.pi], size=dim)
                x = base + rng.normal(0, 0.4, dim)
                x = np.clip(x, -5, 5)
            else:
                # Fine-grained search near current best
                x = best_x + rng.normal(0, 0.8, dim)
                x = np.clip(x, -5, 5)
            counted_func(x)

        # Phase 2: Adaptive local refinement using Powell's method
        # Run multiple Powell instances from top candidates and perturbations
        
        # Build a list of starting points for local search
        start_points = []
        
        # Add top candidates from Phase 1
        for fx, x in candidates[:5]:
            start_points.append(x)
        
        # Add perturbations of the global best at multiple scales
        for sigma in [1.5, 0.8, 0.4, 0.2]:
            for _ in range(3):
                pert = best_x + rng.normal(0, sigma, dim)
                pert = np.clip(pert, -5, 5)
                start_points.append(pert)
        
        # Add a few random points in the promising region
        for _ in range(5):
            x = rng.normal(0, 1.0, dim)
            x = np.clip(x, -4, 4)
            start_points.append(x)
        
        # Run Powell from each starting point, with adaptive budget allocation
        max_local_runs = min(len(start_points), 12)
        run_count = 0
        for x_start in start_points:
            if calls_made >= budget or time.time() > deadline - 15:
                break
            if run_count >= max_local_runs:
                break
            
            # Allocate more budget to earlier runs (they're more likely to be good)
            remaining = budget - calls_made
            if run_count == 0:
                maxfev = min(120, max(30, remaining // 3))
            elif run_count < 3:
                maxfev = min(80, max(25, remaining // 4))
            else:
                maxfev = min(50, max(15, remaining // 5))
            
            try:
                res = minimize(
                    counted_func, 
                    x_start, 
                    method='Powell', 
                    options={'maxiter': 100, 'maxfev': maxfev}
                )
            except Exception:
                pass
            run_count += 1

        # Phase 3: Final high-precision polish
        # Use Nelder-Mead with tight tolerances to squeeze out last bit of precision
        if calls_made < budget and time.time() < deadline - 5:
            try:
                res = minimize(
                    counted_func, 
                    best_x, 
                    method='Nelder-Mead', 
                    options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-10}
                )
            except Exception:
                pass
        
        # Phase 4: One more polish with even tighter settings if budget allows
        if calls_made < budget - 5 and time.time() < deadline - 2:
            try:
                res = minimize(
                    counted_func, 
                    best_x, 
                    method='Powell', 
                    options={'maxiter': 50, 'maxfev': 30}
                )
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
