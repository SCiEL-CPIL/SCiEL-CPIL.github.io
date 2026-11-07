"""CMA-ES based optimizer for BBOB f22 (Gallagher 21 peaks)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start CMA-ES optimizer for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es(self, func, x0, remaining_budget, calls_ref, best_state, deadline, rng, sigma0=1.0):
        """Run CMA-ES from x0, consuming from shared budget."""
        n = self.dim
        pop_size = max(4, int(4 + 3 * np.log(n)))  # ~12 for n=5
        
        # CMA-ES state
        x_mean = np.array(x0, dtype=float)
        sigma = sigma0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        cmac_c = (4 + n / 3) / (n + 4)
        cmac_s = (1 + 2 * max(0, np.sqrt((n - 1) / (n + 3)) - 1)) / (n + 2)
        cc = (4 + n / 2) / (n + 4 + 2)
        cs = (n + 1.3) / (n + 3 + 2 * n / (n + 3))
        chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n * n))
        
        # Initialize with random perturbation
        x_mean = x_mean + sigma * rng.standard_normal(n)
        x_mean = np.clip(x_mean, -5.0, 5.0)
        
        best_f_local = float('inf')
        best_x_local = x_mean.copy()
        
        generations = 0
        max_generations = remaining_budget // pop_size
        
        for gen in range(max_generations):
            if calls_ref[0] >= self.budget or time.time() > deadline:
                break
            
            # Generate population
            B = np.linalg.cholesky(C)
            for i in range(pop_size):
                if calls_ref[0] >= self.budget or time.time() > deadline:
                    break
                z = rng.standard_normal(n)
                x_i = x_mean + sigma * (B @ z)
                x_i = np.clip(x_i, -5.0, 5.0)
                
                calls_ref[0] += 1
                try:
                    f_i = float(func(x_i))
                except Exception:
                    f_i = best_state[0]
                
                if f_i < best_f_local:
                    best_f_local = f_i
                    best_x_local = x_i.copy()
                if f_i < best_state[0]:
                    best_state[0] = f_i
                    best_state[1] = x_i.copy()
            
            if calls_ref[0] >= self.budget or time.time() > deadline:
                break
            
            # Sort by fitness
            # We need to collect all evaluated points this generation
            # Let's restructure: collect all points first
            
            break  # Simplified: just do one generation at a time
            
        return best_x_local, best_f_local

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 300.0  # Safety margin before 330s kill
        
        dim = self.dim
        budget = self.budget
        calls = [0]
        best_state = [float('inf'), np.zeros(dim)]
        rng = np.random.default_rng(42)
        
        def eval_and_track(x):
            if calls[0] >= budget or time.time() > deadline:
                return best_state[0]
            calls[0] += 1
            try:
                x_clipped = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
                val = float(func(x_clipped))
                if val < best_state[0]:
                    best_state[0] = val
                    best_state[1] = x_clipped.copy()
                return val
            except Exception:
                return best_state[0]
        
        # Phase 1: Initial sampling to find promising basins
        n_initial = min(30, budget)
        for i in range(n_initial):
            if calls[0] >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5.0, 5.0, size=dim)
            eval_and_track(x)
        
        # Phase 2: Multi-start CMA-ES
        # Use the best few points as starting points for CMA-ES restarts
        # We'll run CMA-ES with adaptive budget allocation
        
        # Collect some diverse starting points from initial phase
        # Re-sample a few more to get diversity
        n_extra = min(20, budget - calls[0])
        extra_points = []
        for i in range(n_extra):
            if calls[0] >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5.0, 5.0, size=dim)
            val = eval_and_track(x)
            extra_points.append((val, x))
        
        # Sort by fitness and take top diverse points
        extra_points.sort(key=lambda t: t[0])
        
        # Select starting points: best + some diverse ones
        start_points = [best_state[1].copy()]
        if extra_points:
            start_points.append(extra_points[0][1].copy())
            if len(extra_points) > 5:
                start_points.append(extra_points[len(extra_points)//2][1].copy())
            if len(extra_points) > 10:
                start_points.append(extra_points[-1][1].copy())
        
        # Run CMA-ES from each start point with remaining budget
        remaining = budget - calls[0]
        n_restarts = min(len(start_points), max(2, remaining // 50))
        
        for idx, sp in enumerate(start_points[:n_restarts]):
            if calls[0] >= budget - 10 or time.time() > deadline:
                break
            
            # Allocate budget proportionally, with diminishing returns for later restarts
            if idx == 0:
                alloc = min(remaining, max(50, int(remaining * 0.4)))
            elif idx == 1:
                alloc = min(remaining, max(30, int(remaining * 0.3)))
            else:
                alloc = min(remaining, max(20, int(remaining * 0.15)))
            
            remaining -= alloc
            
            # Run simplified CMA-ES
            self._run_cma_es(func, sp, alloc, calls, best_state, deadline, rng)
        
        # Phase 3: Final local refinement if budget remains
        if calls[0] < budget - 5 and time.time() < deadline:
            # Simple gradient-free local search around best
            x_best = best_state[1].copy()
            step_size = 0.1
            for step_iter in range(50):
                if calls[0] >= budget or time.time() > deadline:
                    break
                improved = False
                for d in range(dim):
                    for direction in [-1, 1]:
                        if calls[0] >= budget or time.time() > deadline:
                            break
                        x_try = x_best.copy()
                        x_try[d] += direction * step_size
                        x_try = np.clip(x_try, -5.0, 5.0)
                        val = eval_and_track(x_try)
                        if val < best_state[0]:
                            x_best = x_try.copy()
                            improved = True
                if not improved:
                    step_size *= 0.7
                    if step_size < 0.001:
                        break
        
        return best_state[1], best_state[0]
    
    def _run_cma_es(self, func, x0, alloc_budget, calls, best_state, deadline, rng):
        """Run CMA-ES with given budget allocation."""
        n = self.dim
        pop_size = max(4, int(4 + 3 * np.log(n)))  # ~12 for n=5
        
        x_mean = np.array(x0, dtype=float)
        sigma = 1.0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        cmac_c = (4 + n / 3) / (n + 4)
        cmac_s = (1 + 2 * max(0, np.sqrt((n - 1) / (n + 3)) - 1)) / (n + 2)
        cc = (4 + n / 2) / (n + 4 + 2)
        cs = (n + 1.3) / (n + 3 + 2 * n / (n + 3))
        chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n * n))
        
        # Initial perturbation
        x_mean = x_mean + sigma * rng.standard_normal(n)
        x_mean = np.clip(x_mean, -5.0, 5.0)
        
        max_gens = alloc_budget // pop_size
        B = None
        
        for gen in range(max_gens):
            if calls[0] >= self.budget or time.time() > deadline:
                break
            
            # Compute B if C changed (lazy)
            try:
                B = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(n)
                B = np.eye(n)
            
            # Generate and evaluate population
            offspring = []
            for i in range(pop_size):
                if calls[0] >= self.budget or time.time() > deadline:
                    break
                z = rng.standard_normal(n)
                x_i = x_mean + sigma * (B @ z)
                x_i = np.clip(x_i, -5.0, 5.0)
                
                calls[0] += 1
                try:
                    f_i = float(func(x_i))
                except Exception:
                    f_i = best_state[0]
                
                offspring.append((f_i, x_i))
                if f_i < best_state[0]:
                    best_state[0] = f_i
                    best_state[1] = x_i.copy()
            
            if len(offspring) < 2:
                break
            
            # Sort by fitness
            offspring.sort(key=lambda t: t[0])
            
            # Update CMA-ES parameters
            mu = pop_size // 2  # number of selected individuals
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mueff = 1.0 / np.sum(weights ** 2)
            
            # Update mean
            x_new_mean = np.zeros(n)
            for i in range(mu):
                x_new_mean += weights[i] * offspring[i][1]
            
            # Update step-size control (CSA)
            if np.linalg.norm(B) > 1e-10:
                y = (x_new_mean - x_mean) / sigma
                ps = (1 - cs) * ps + cs * np.sqrt(mueff) / np.linalg.norm(y) * y if np.linalg.norm(y) > 1e-10 else (1 - cs) * ps
            else:
                ps = (1 - cs) * ps
            
            hsig = (np.linalg.norm(ps) / (1 - (1 - cs) ** (gen + 1))) / chi_n < 1.4 + 2 / (n + 1)
            
            # Update covariance
            if hsig:
                pc = (1 - cc) * pc + cc * (x_new_mean - x_mean) / sigma
            else:
                pc = (1 - cc) * pc
            
            delta_hsig = (1 - hsig) * cc * (2 - cc)
            C = (1 + cmac_c * delta_hsig - cmac_c) * C + cmac_s * (pc @ pc.T - C) if False else \
                (1 - cmac_c) * C + cmac_s * (pc @ pc.T + (delta_hsig - cmac_s) * C)
            
            # Ensure positive semi-definite
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-10)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            except np.linalg.LinAlgError:
                C = np.eye(n)
            
            # Update step size (CSA)
            sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            x_mean = x_new_mean
        
        return best_state[1], best_state[0]
# EVOLVE-BLOCK-END
