import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.start_time = time.time()
        self.best_x = np.zeros(dim)
        self.best_f = np.inf
        self.timeout = 290.0  # Safety margin from 300s limit
        self.stagnation_count = 0
        self.best_f_history = []

    def _evaluate(self, x, func):
        if time.time() - self.start_time > self.timeout:
            raise TimeoutError("Time limit exceeded")
        
        if self.calls >= self.budget:
            raise ValueError("Budget exceeded")
            
        self.calls += 1
        x = np.clip(x, -5.0, 5.0)
        f = func(x)
        
        if f < self.best_f:
            self.best_f = f
            self.best_x = x.copy()
            self.stagnation_count = 0
            self.best_f_history.append(f)
        else:
            self.stagnation_count += 1
            
        return f

    def _nelder_mead(self, x0, func, max_iters=100, initial_step=0.5):
        """
        Improved Nelder-Mead with adaptive step size and proper shrink handling.
        """
        n = self.dim
        # Initialize simplex with adaptive step size
        simplex = np.zeros((n + 1, n))
        simplex[0] = x0
        for i in range(1, n + 1):
            simplex[i] = x0.copy()
            simplex[i][i - 1] = initial_step
        
        # Evaluate initial simplex
        fvals = np.zeros(n + 1)
        for i in range(n + 1):
            try:
                fvals[i] = self._evaluate(simplex[i], func)
            except (TimeoutError, ValueError):
                return

        # Track best in this local search
        local_best_f = fvals.min()
        local_best_x = simplex[np.argmin(fvals)].copy()
        no_improve_count = 0
        max_no_improve = 20
        
        # Main loop
        for _ in range(max_iters):
            # Check time/budget
            if time.time() - self.start_time > self.timeout or self.calls >= self.budget:
                return

            # Sort simplex
            idx = np.argsort(fvals)
            simplex = simplex[idx]
            fvals = fvals[idx]

            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            worst = simplex[-1]
            worst_f = fvals[-1]
            best_f = fvals[0]

            # Check for improvement
            if best_f < local_best_f - 1e-10:
                local_best_f = best_f
                local_best_x = simplex[0].copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            if no_improve_count >= max_no_improve:
                break

            # Reflection
            alpha = 1.0
            x_ref = centroid + alpha * (centroid - worst)
            try:
                f_ref = self._evaluate(x_ref, func)
            except (TimeoutError, ValueError):
                return

            if f_ref < best_f:
                # Expansion
                gamma = 2.0
                x_exp = centroid + gamma * (centroid - worst)
                try:
                    f_exp = self._evaluate(x_exp, func)
                except (TimeoutError, ValueError):
                    return

                if f_exp < f_ref:
                    simplex[-1] = x_exp
                    fvals[-1] = f_exp
                else:
                    simplex[-1] = x_ref
                    fvals[-1] = f_ref
            elif f_ref < fvals[-2]:
                # Accept reflection
                simplex[-1] = x_ref
                fvals[-1] = f_ref
            else:
                # Contraction
                beta = 0.5
                x_con = centroid + beta * (worst - centroid)
                try:
                    f_con = self._evaluate(x_con, func)
                except (TimeoutError, ValueError):
                    return

                if f_con < worst_f:
                    simplex[-1] = x_con
                    fvals[-1] = f_con
                else:
                    # Shrink: move all points except best toward best
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        try:
                            fvals[i] = self._evaluate(simplex[i], func)
                        except (TimeoutError, ValueError):
                            return
                    # Don't re-evaluate simplex[0], it's already the best

            # Convergence check
            if np.std(fvals) < 1e-10:
                break

    def _adaptive_cma_es(self, x0, func, population_size=15, max_generations=50):
        """
        Simple CMA-ES variant for local refinement.
        """
        n = self.dim
        rng = np.random.default_rng(42)
        
        # Initialize
        mean = x0.copy()
        sigma = 0.3
        C = np.eye(n)
        pc = np.zeros(n)
        cs = 1.0 / (n + 3)
        c1 = 2 / ((n + 1) ** 2 + 10)
        cmu = min(0.4 - c1, 2 / ((n + 1) ** 2 + 5))
        
        # Precompute
        sqrt_C = np.linalg.cholesky(C)
        
        best_f = self.best_f
        best_x = self.best_x.copy()
        
        for gen in range(max_generations):
            if time.time() - self.start_time > self.timeout or self.calls >= self.budget:
                break
            
            # Generate population
            population = np.zeros((population_size, n))
            fvals = np.zeros(population_size)
            
            for i in range(population_size):
                z = rng.standard_normal(n)
                x = mean + sigma * (sqrt_C @ z)
                try:
                    fvals[i] = self._evaluate(x, func)
                except (TimeoutError, ValueError):
                    break
                
                population[i] = x
            
            # Sort by fitness
            idx = np.argsort(fvals)
            population = population[idx]
            fvals = fvals[idx]
            
            # Update best
            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = population[0].copy()
            
            # Update mean
            mu_eff = int(population_size // 2)
            weights = np.zeros(mu_eff)
            for i in range(mu_eff):
                weights[i] = np.log((mu_eff + 1) / 2) - np.log(i + 1)
            weights /= weights.sum()
            
            old_mean = mean.copy()
            mean = np.sum(weights[:, None] * population[:mu_eff], axis=0)
            
            # Update step size
            y = (mean - old_mean) / sigma
            pc = (1 - cs) * pc + cs * y
            sigma *= np.exp((1 / (n + 3)) * (np.linalg.norm(pc) / np.sqrt(1 - (1 - cs) ** 2) - np.sqrt(n - 1)))
            sigma = max(sigma, 1e-6)
            
            # Update covariance
            v = y / np.linalg.norm(y) if np.linalg.norm(y) > 0 else np.zeros(n)
            rank1 = np.outer(v, v)
            rankmu = np.sum(weights[:, None, None] * np.outer(population[:mu_eff] - old_mean, population[:mu_eff] - old_mean) / (sigma ** 2), axis=0)
            C = (1 + c1 * (2 - c1 - cmu) - c1 - cmu) * C + c1 * rank1 + cmu * rankmu
            
            # Ensure positive definite
            try:
                sqrt_C = np.linalg.cholesky(C)
            except:
                C = C + 1e-6 * np.eye(n)
                sqrt_C = np.linalg.cholesky(C)
            
            # Convergence
            if np.std(fvals[:mu_eff]) < 1e-10:
                break
        
        self.best_f = best_f
        self.best_x = best_x
    
    def __call__(self, func):
        # Phase 1: Enhanced Latin Hypercube Sampling
        n_lhs = 50
        rng = np.random.default_rng(42)
        
        lhs_points = np.zeros((n_lhs, self.dim))
        for d in range(self.dim):
            intervals = np.linspace(-5, 5, n_lhs + 1)
            points_d = rng.uniform(intervals[:-1], intervals[1:])
            rng.shuffle(points_d)
            lhs_points[:, d] = points_d

        lhs_fvals = np.zeros(n_lhs)
        for i in range(n_lhs):
            try:
                lhs_fvals[i] = self._evaluate(lhs_points[i], func)
            except (TimeoutError, ValueError):
                break

        # Phase 2: Multi-stage local optimization
        sorted_idx = np.argsort(lhs_fvals)
        
        # Stage 1: Nelder-Mead on top 5 candidates
        top_k_nm = min(5, n_lhs)
        for k in range(top_k_nm):
            if self.calls >= self.budget:
                break
            if time.time() - self.start_time > self.timeout:
                break
            
            start_x = lhs_points[sorted_idx[k]]
            self._nelder_mead(start_x, func, max_iters=80, initial_step=0.3)
        
        # Stage 2: CMA-ES on top 3 candidates if budget remains
        if self.calls < self.budget * 0.7:
            top_k_cma = min(3, n_lhs)
            for k in range(top_k_cma):
                if self.calls >= self.budget:
                    break
                if time.time() - self.start_time > self.timeout:
                    break
                
                start_x = lhs_points[sorted_idx[k]]
                self._adaptive_cma_es(start_x, func, population_size=12, max_generations=30)
        
        # Stage 3: Final Nelder-Mead polish on best point
        if self.calls < self.budget and time.time() - self.start_time < self.timeout:
            self._nelder_mead(self.best_x, func, max_iters=30, initial_step=0.1)

        return self.best_x, self.best_f
