import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es_run(self, func, x0, sigma0, budget, rng, deadline, best_f_tracker):
        """
        Run CMA-ES.
        Returns: best_x, best_f, calls_used
        best_f_tracker: dict with 'x' and 'f' keys that get updated in-place
        """
        d = self.dim
        lam = 4 + int(3 * np.log(d))
        
        weights = np.log(lam / 2 + 1) - np.log(np.arange(1, lam + 1))
        weights /= weights.sum()
        mu = int(lam / 2)
        w = weights[:mu]
        w = w / w.sum()
        mueff = 1.0 / np.sum(w**2)
        
        c_sigma = (mueff + 2) / (d + mueff + 5)
        c_c = (4 - mueff + c_sigma) / (d + mueff + 9)
        c_1 = 2 / ((d + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1/mueff) / ((d + 2)**2 + mueff))
        
        x_mean = x0.copy()
        sigma = sigma0
        
        C = np.eye(d)
        B = np.eye(d)
        D = np.ones(d)
        
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)
        
        f0 = func(x0)
        best_x = x0.copy()
        best_f = f0
        calls = 1
        
        if f0 < best_f_tracker['f']:
            best_f_tracker['f'] = f0
            best_f_tracker['x'] = x0.copy()
        
        chi_d = np.sqrt(d) * (1 - 1/(4*d) + 1/(21*d**2))
        
        no_improve_generations = 0
        max_no_improve = 12
        
        while calls < budget:
            if time.time() > deadline:
                break
            
            remaining = budget - calls
            if remaining < lam:
                break
            
            ys = [B @ (D * rng.standard_normal(d)) for _ in range(lam)]
            
            xs = []
            fs = []
            for y in ys:
                x_new = x_mean + sigma * y
                f_new = func(x_new)
                xs.append(x_new)
                fs.append(f_new)
                calls += 1
                
                if f_new < best_f:
                    best_f = f_new
                    best_x = x_new.copy()
                    no_improve_generations = 0
                if f_new < best_f_tracker['f']:
                    best_f_tracker['f'] = f_new
                    best_f_tracker['x'] = x_new.copy()
            
            if len(fs) < lam:
                break

            idx = np.argsort(fs)
            xs_sorted = np.array(xs)[idx]
            fs_sorted = np.array(fs)[idx]
            
            if fs_sorted[0] >= best_f - 1e-10:
                no_improve_generations += 1
            else:
                no_improve_generations = 0
            
            if no_improve_generations >= max_no_improve:
                break
            
            x_mean_new = np.sum(w[:, None] * xs_sorted[:mu], axis=0)
            y_sigma = (x_mean_new - x_mean) / sigma
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * y_sigma
            
            if np.linalg.norm(p_sigma) < (1.4 + 2/(d+1)) * chi_d:
                h_sigma = 1.0
            else:
                h_sigma = 0.0
            
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * y_sigma
            
            y_sorted = (xs_sorted[:mu] - x_mean) / sigma
            delta_h = (1 - h_sigma**2) * c_c * (2 - c_c)
            
            C_new = (1 - c_1 - c_mu) * C
            C_new += c_1 * (np.outer(p_c, p_c) + delta_h * C)
            C_new += c_mu * (y_sorted.T @ (w[:, None] * y_sorted))
            
            C_new = (C_new + C_new.T) / 2
            
            eigenvalues, B_new = np.linalg.eigh(C_new)
            eigenvalues = np.maximum(eigenvalues, 1e-16)
            D_new = np.sqrt(eigenvalues)
            
            x_mean = x_mean_new
            C = C_new
            B = B_new
            D = D_new
            
            sigma = sigma * np.exp(c_sigma * (np.linalg.norm(p_sigma) / chi_d - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)
            
            if sigma < 1e-6:
                break
        
        return best_x, best_f, calls

    def _nelder_mead(self, func, x0, budget, deadline, best_f_tracker):
        """
        Nelder-Mead local optimization.
        Returns: best_x, best_f, calls_used
        """
        d = self.dim
        calls = 0
        x_best = x0.copy()
        f_best = np.inf
        
        # Initialize simplex
        simplex = []
        fvals = []
        for i in range(d + 1):
            if i == 0:
                x = x0.copy()
            else:
                x = x0.copy()
                x[i-1] += 0.05 if x0[i-1] < 4.95 else -0.05
            f = func(x)
            calls += 1
            simplex.append(x)
            fvals.append(f)
            if f < f_best:
                f_best = f
                x_best = x.copy()
            if f < best_f_tracker['f']:
                best_f_tracker['f'] = f
                best_f_tracker['x'] = x.copy()
        
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma_nm = 0.5
        
        max_iter = budget - calls
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            if calls >= budget:
                break
            
            # Sort
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            # Check convergence
            if abs(fvals[0] - fvals[-1]) < 1e-12:
                break
            
            x_worst = simplex[-1]
            x_best_s = simplex[0]
            f_worst = fvals[-1]
            f_best_s = fvals[0]
            
            # Centroid of all but worst
            x_centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_reflect = x_centroid + alpha * (x_centroid - x_worst)
            f_reflect = func(x_reflect)
            calls += 1
            if f_reflect < best_f_tracker['f']:
                best_f_tracker['f'] = f_reflect
                best_f_tracker['x'] = x_reflect.copy()
            
            if f_reflect < fvals[0]:
                # Expansion
                x_expand = x_centroid + gamma * (x_reflect - x_centroid)
                f_expand = func(x_expand)
                calls += 1
                if f_expand < best_f_tracker['f']:
                    best_f_tracker['f'] = f_expand
                    best_f_tracker['x'] = x_expand.copy()
                if f_expand < f_reflect:
                    simplex[-1] = x_expand
                    fvals[-1] = f_expand
                else:
                    simplex[-1] = x_reflect
                    fvals[-1] = f_reflect
            elif f_reflect < fvals[-2]:
                simplex[-1] = x_reflect
                fvals[-1] = f_reflect
            else:
                # Contraction
                if f_reflect < f_worst:
                    # Outside contraction
                    x_contract = x_centroid + 0.5 * (x_reflect - x_centroid)
                else:
                    # Inside contraction
                    x_contract = x_centroid - 0.5 * (x_worst - x_centroid)
                f_contract = func(x_contract)
                calls += 1
                if f_contract < best_f_tracker['f']:
                    best_f_tracker['f'] = f_contract
                    best_f_tracker['x'] = x_contract.copy()
                
                if f_contract < min(f_reflect, f_worst):
                    simplex[-1] = x_contract
                    fvals[-1] = f_contract
                else:
                    # Shrink
                    for i in range(1, d + 1):
                        simplex[i] = x_best_s + sigma_nm * (simplex[i] - x_best_s)
                        fvals[i] = func(simplex[i])
                        calls += 1
                        if fvals[i] < best_f_tracker['f']:
                            best_f_tracker['f'] = fvals[i]
                            best_f_tracker['x'] = simplex[i].copy()
            
            if f_reflect < f_best:
                f_best = f_reflect
                x_best = x_reflect.copy()
            if fvals[0] < f_best:
                f_best = fvals[0]
                x_best = simplex[0].copy()
        
        return x_best, f_best, calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0
        
        d = self.dim
        total_budget = self.budget
        rng = np.random.default_rng(42)
        
        best_f_tracker = {'x': None, 'f': np.inf}
        calls_used = 0
        
        # Phase 1: Multiple CMA-ES runs with adaptive restarts
        # Use ~65% of budget for global search with restarts
        global_budget = int(total_budget * 0.65)
        
        # Initial points: center + Halton-based spread
        initial_points = [np.zeros(d)]
        # Add a few well-spread points
        for i in range(1, 4):
            point = np.zeros(d)
            for j in range(d):
                # Use Halton-like spread
                point[j] = -4 + 8 * ((i * (j+1) + 1) / 5 % 1.0)
            initial_points.append(np.clip(point, -4.5, 4.5))
        
        # Add random points
        for _ in range(2):
            initial_points.append(rng.uniform(-4, 4, d))
        
        for i, x_init in enumerate(initial_points):
            if calls_used >= global_budget or time.time() > deadline:
                break
            
            remaining_budget = global_budget - calls_used
            if remaining_budget < 25:
                break
            
            # Adaptive sigma
            dist_from_center = np.linalg.norm(x_init)
            sigma_init = max(1.2, min(2.5, 1.8 + 0.05 * dist_from_center))
            
            x_best, f_best, calls_this_run = self._cma_es_run(
                func, x_init, sigma_init, remaining_budget, rng, deadline, best_f_tracker
            )
            
            calls_used += calls_this_run
            
            if f_best < best_f_tracker['f']:
                best_f_tracker['f'] = f_best
                best_f_tracker['x'] = x_best.copy()
        
        # Phase 2: Local polish using Nelder-Mead
        if best_f_tracker['x'] is not None and calls_used < total_budget:
            polish_budget = total_budget - calls_used
            max_polish_calls = min(polish_budget, 120)
            
            x_best, f_best, polish_calls = self._nelder_mead(
                func, best_f_tracker['x'], max_polish_calls, deadline, best_f_tracker
            )
            calls_used += polish_calls
            
            if f_best < best_f_tracker['f']:
                best_f_tracker['f'] = f_best
                best_f_tracker['x'] = x_best.copy()
        
        # If we haven't made any calls (shouldn't happen), make one
        if calls_used == 0:
            x0 = np.zeros(d)
            f0 = func(x0)
            best_f_tracker['x'] = x0
            best_f_tracker['f'] = f0
            calls_used = 1
        
        return best_f_tracker['x'], best_f_tracker['f']
