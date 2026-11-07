"""Optimized optimizer for BBOB-style f18 (Schaffers F7 ill-conditioned)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Pattern search with adaptive step size and multi-start."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _pattern_search(self, func, rng, x0, max_calls, deadline):
        """
        Pattern search (Hooke-Jeeves style) with adaptive step size.
        Explores orthogonal and diagonal directions systematically.
        """
        dim = self.dim
        x = np.array(x0, dtype=float)
        fx = func(x)
        calls = 1
        
        # Initial step size
        step = 1.5
        
        # Directions: standard basis + diagonals (all combinations of +/-1 normalized)
        directions = []
        # Standard basis
        for i in range(dim):
            e = np.zeros(dim)
            e[i] = 1.0
            directions.append(e)
        # Diagonal directions (all 2^dim sign combinations, but that's too many for dim=5: 32)
        # Instead, use a set of well-spread diagonal directions
        # For dim=5, use the 5 "main" diagonals and some mixed ones
        # Actually, let's use a smarter set: all directions with entries in {-1, 0, 1}
        # but normalized. For dim=5, that's 3^5 = 243 directions, too many.
        # Let's use: standard basis (5) + all pairwise diagonals (C(5,2)*2 = 20) 
        # + a few full diagonals
        # Total: 5 + 20 + 4 = 29 directions, manageable
        
        # Add pairwise diagonals
        for i in range(dim):
            for j in range(i + 1, dim):
                d1 = np.zeros(dim)
                d1[i] = 1.0
                d1[j] = 1.0
                d1 /= np.linalg.norm(d1)
                directions.append(d1)
                
                d2 = np.zeros(dim)
                d2[i] = 1.0
                d2[j] = -1.0
                d2 /= np.linalg.norm(d2)
                directions.append(d2)
        
        # Add a few full diagonal directions
        for signs in [(1, 1, 1, 1, 1), (1, 1, 1, -1, -1), (1, -1, -1, 1, -1), (-1, 1, -1, -1, 1)]:
            d = np.array(signs, dtype=float)
            d /= np.linalg.norm(d)
            directions.append(d)
        
        n_dirs = len(directions)
        
        # Track progress
        no_improve_count = 0
        max_no_improve = 30
        
        while calls < max_calls:
            if time.time() > deadline - 5.0:
                break
            
            improved = False
            
            # Try each direction
            for d in directions:
                if calls >= max_calls or time.time() > deadline - 5.0:
                    break
                
                # Try positive direction
                x_new = np.clip(x + step * d, -5, 5)
                if np.allclose(x_new, x):
                    continue
                fx_new = func(x_new)
                calls += 1
                
                if fx_new < fx - 1e-10:
                    x = x_new
                    fx = fx_new
                    improved = True
                    no_improve_count = 0
                    # Extend in this direction
                    for _ in range(3):
                        if calls >= max_calls or time.time() > deadline - 5.0:
                            break
                        x_try = np.clip(x + step * d, -5, 5)
                        if np.allclose(x_try, x):
                            break
                        fx_try = func(x_try)
                        calls += 1
                        if fx_try < fx - 1e-10:
                            x = x_try
                            fx = fx_try
                        else:
                            break
                    break  # Restart direction scan after improvement
                
                # Try negative direction
                x_new_neg = np.clip(x - step * d, -5, 5)
                if np.allclose(x_new_neg, x):
                    continue
                fx_new_neg = func(x_new_neg)
                calls += 1
                
                if fx_new_neg < fx - 1e-10:
                    x = x_new_neg
                    fx = fx_new_neg
                    improved = True
                    no_improve_count = 0
                    # Extend in negative direction
                    for _ in range(3):
                        if calls >= max_calls or time.time() > deadline - 5.0:
                            break
                        x_try = np.clip(x - step * d, -5, 5)
                        if np.allclose(x_try, x):
                            break
                        fx_try = func(x_try)
                        calls += 1
                        if fx_try < fx - 1e-10:
                            x = x_try
                            fx = fx_try
                        else:
                            break
                    break  # Restart direction scan after improvement
            
            if not improved:
                no_improve_count += 1
                # Reduce step size
                step *= 0.7
                if no_improve_count >= max_no_improve:
                    break
            
            # Cap step size
            step = min(step, 10.0)
            step = max(step, 1e-8)
        
        return x, fx, calls

    def _adaptive_es(self, func, rng, x0, sigma0, max_calls, deadline):
        """
        Adaptive (1+1)-ES with 1/5 success rule.
        """
        dim = self.dim
        x = np.array(x0, dtype=float)
        fx = func(x)
        sigma = sigma0
        calls = 1
        successes = 0
        trials = 0
        
        while calls < max_calls:
            if time.time() > deadline - 3.0:
                break
            
            # Generate one candidate
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1
            trials += 1
            
            if fy < fx - 1e-10:
                x = y
                fx = fy
                successes += 1
            
            # 1/5 rule: after 10 trials, adjust sigma
            if trials >= 10:
                success_rate = successes / trials
                if success_rate > 0.4:
                    sigma *= 1.25
                elif success_rate < 0.1:
                    sigma *= 0.8
                else:
                    sigma *= 1.05
                successes = 0
                trials = 0
            
            sigma = max(1e-8, min(sigma, 20.0))
        
        return x, fx, calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        
        dim = self.dim
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        total_calls = 0
        budget = self.budget
        
        # Phase 1: Multi-start with pattern search
        # Use 3-4 diverse starting points
        n_starts = 4
        calls_per_start = max(30, budget // (n_starts + 2))
        
        # Starting points: center, and diverse random points
        start_points = [
            rng.uniform(-1, 1, dim),  # Near center
            rng.uniform(-4, 4, dim),  # Broad
            rng.uniform(-5, 5, dim),  # Full range
            rng.uniform(-3, 3, dim),  # Medium
        ]
        
        for i, x0 in enumerate(start_points):
            if total_calls >= budget or time.time() > deadline - 30.0:
                break
            
            x, fx, calls_used = self._pattern_search(
                func, rng, x0,
                min(calls_per_start, budget - total_calls),
                deadline
            )
            total_calls += calls_used
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Phase 2: Refine best point with (1+1)-ES
        if best_x is not None and total_calls < budget:
            remaining = budget - total_calls
            if remaining > 10:
                x, fx, calls_used = self._adaptive_es(
                    func, rng, best_x, 0.5,
                    remaining, deadline
                )
                total_calls += calls_used
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
        
        # Ensure we've made at least one call
        if total_calls == 0:
            x = rng.uniform(-5, 5, dim)
            best_f = func(x)
            best_x = x
            total_calls = 1
        
        return best_x, best_f
# EVOLVE-BLOCK-END
