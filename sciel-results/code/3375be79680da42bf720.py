import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _halton_sequence(self, n, dim):
        """Generate Halton sequence points in [0, 1]^dim."""
        points = []
        for i in range(1, n + 1):
            point = []
            for base in range(2, 2 + dim):
                f = 1.0
                r = 0.0
                i_temp = i
                while i_temp > 0:
                    f /= base
                    r += f * (i_temp % base)
                    i_temp //= base
                point.append(r)
            points.append(point)
        return np.array(points)

    def _cma_es_run(self, func, x0, sigma0, max_calls, rng, deadline, global_best_tracker):
        """
        Run CMA-ES with adaptive restarts.
        Returns: best_x, best_f, calls_used
        global_best_tracker: list with one element, the current global best f-value
        """
        d = self.dim
        # Tuned for 5D
        lam = 10
        mu = lam // 2
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        
        # Strategy parameters
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
        
        if best_f < global_best_tracker[0]:
            global_best_tracker[0] = best_f
        
        chi_d = np.sqrt(d) * (1 - 1/(4*d) + 1/(21*d**2))
        
        no_improve_generations = 0
        max_no_improve = 8
        
        while calls < max_calls:
            if time.time() > deadline:
                break
            
            remaining = max_calls - calls
            if remaining < lam:
                break
            
            # Sample
            ys = rng.standard_normal((lam, d))
            xs = x_mean + sigma * (ys @ (B * D)[None, :, :])
            
            xs_list = []
            fs_list = []
            
            for i in range(lam):
                x_new = xs[i]
                f_new = func(x_new)
                xs_list.append(x_new)
                fs_list.append(f_new)
                calls += 1
                
                if f_new < best_f:
                    best_f = f_new
                    best_x = x_new.copy()
                    no_improve_generations = 0
                    if best_f < global_best_tracker[0]:
                        global_best_tracker[0] = best_f
            
            if len(fs_list) < lam:
                break

            idx = np.argsort(fs_list)
            xs_sorted = np.array(xs_list)[idx]
            fs_sorted = np.array(fs_list)[idx]
            
            # Check stagnation
            if fs_sorted[0] >= best_f - 1e-10:
                no_improve_generations += 1
            else:
                no_improve_generations = 0
            
            if no_improve_generations >= max_no_improve:
                # Reset state for adaptive restart
                x_mean = best_x.copy()
                sigma = sigma * 1.5
                C = np.eye(d)
                B = np.eye(d)
                D = np.ones(d)
                p_sigma = np.zeros(d)
                p_c = np.zeros(d)
                no_improve_generations = 0
                
                if sigma > 10.0:
                    break
                continue

            # Update mean
            x_mean_new = np.sum(weights[:, None] * xs_sorted[:mu], axis=0)
            y_sigma = (x_mean_new - x_mean) / sigma
            
            # Update p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * y_sigma
            
            if np.linalg.norm(p_sigma) < (1.4 + 2/(d+1)) * chi_d:
                h_sigma = 1.0
            else:
                h_sigma = 0.0
            
            # Update p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * y_sigma
            
            # Update C
            y_sorted = (xs_sorted[:mu] - x_mean) / sigma
            delta_h = (1 - h_sigma**2) * c_c * (2 - c_c)
            
            C_new = (1 - c_1 - c_mu) * C
            C_new += c_1 * (np.outer(p_c, p_c) + delta_h * C)
            C_new += c_mu * (y_sorted.T @ (weights[:, None] * y_sorted))
            
            C_new = (C_new + C_new.T) / 2
            
            # Eigen decomposition
            eigenvalues, B_new = np.linalg.eigh(C_new)
            eigenvalues = np.maximum(eigenvalues, 1e-16)
            D_new = np.sqrt(eigenvalues)
            
            x_mean = x_mean_new
            C = C_new
            B = B_new
            D = D_new
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma * (np.linalg.norm(p_sigma) / chi_d - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)
            
            if sigma < 1e-6:
                break
        
        return best_x, best_f, calls

    def _nelder_mead(self, func, x0, max_calls, deadline, global_best_tracker):
        """
        Simple Nelder-Mead implementation with call budget.
        Returns: best_x, best_f, calls_used
        """
        d = self.dim
        x = x0.copy()
        f0 = func(x)
        calls = 1
        best_x = x.copy()
        best_f = f0
        
        if best_f < global_best_tracker[0]:
            global_best_tracker[0] = best_f
        
        # Initial simplex
        simplex = [x.copy()]
        fs = [f0]
        
        for i in range(d):
            xi = x.copy()
            xi[i] += 0.05
            fi = func(xi)
            calls += 1
            simplex.append(xi)
            fs.append(fi)
            if fi < best_f:
                best_f = fi
                best_x = xi.copy()
                if best_f < global_best_tracker[0]:
                    global_best_tracker[0] = best_f
        
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        while calls < max_calls and time.time() < deadline:
            # Sort simplex
            order = np.argsort(fs)
            simplex = [simplex[i].copy() for i in order]
            fs = [fs[i] for i in order]
            
            # Update best
            if fs[0] < best_f:
                best_f = fs[0]
                best_x = simplex[0].copy()
                if best_f < global_best_tracker[0]:
                    global_best_tracker[0] = best_f
            
            # Centroid of all but worst
            centroid = np.mean(simplex[:d], axis=0)
            
            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            f_r = func(x_r)
            calls += 1
            
            if f_r < fs[0]:
                # Expansion
                x_e = centroid + gamma * (centroid - simplex[-1])
                f_e = func(x_e)
                calls += 1
                
                if f_e < f_r:
                    simplex[-1] = x_e
                    fs[-1] = f_e
                else:
                    simplex[-1] = x_r
                    fs[-1] = f_r
            elif f_r < fs[-2]:
                simplex[-1] = x_r
                fs[-1] = f_r
            else:
                # Contraction
                x_c = centroid + rho * (simplex[-1] - centroid)
                f_c = func(x_c)
                calls += 1
                
                if f_c < fs[-1]:
                    simplex[-1] = x_c
                    fs[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, d + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fs[i] = func(simplex[i])
                        calls += 1
                        if fs[i] < best_f:
                            best_f = fs[i]
                            best_x = simplex[i].copy()
                            if best_f < global_best_tracker[0]:
                                global_best_tracker[0] = best_f
                    
                    if calls >= max_calls:
                        break
            
            # Check convergence
            if max(fs) - min(fs) < 1e-10:
                break
        
        return best_x, best_f, calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        
        d = self.dim
        total_budget = self.budget
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls_used = 0
        global_best_tracker = [np.inf]
        
        # Phase 1: Initial Sampling with Halton + special points
        n_halton = 15
        halton_pts = self._halton_sequence(n_halton, d)
        
        special_points = [
            np.zeros(d),
            np.full(d, 2.0),
            np.full(d, -2.0),
            np.array([2.0, -2.0, 2.0, -2.0, 2.0] if d == 5 else [2.0] * d),
            np.array([-2.0, 2.0, -2.0, 2.0, -2.0] if d == 5 else [-2.0] * d),
        ]
        
        initial_candidates = []
        for pt in special_points:
            initial_candidates.append(pt)
        
        for pt in halton_pts:
            # Map [0,1] to [-4,4]
            x = -4 + 8 * pt
            initial_candidates.append(x)
        
        # Evaluate initial candidates
        for x in initial_candidates:
            if calls_used >= total_budget or time.time() > deadline:
                break
            x_clipped = np.clip(x, -5, 5)
            f = func(x_clipped)
            calls_used += 1
            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()
                global_best_tracker[0] = f
        
        if best_x is None:
            # Fallback: single call at origin
            x0 = np.zeros(d)
            f0 = func(x0)
            best_x = x0
            best_f = f0
            calls_used = 1
            global_best_tracker[0] = f0
        
        # Phase 2: CMA-ES runs from multiple starting points
        # Use top 3 points from initial sampling
        # Re-evaluate to get exact values (we have them, but let's track properly)
        # Actually, we just use best_x and a few random perturbations
        
        # Select starting points: best, and 2 random points near best
        start_points = [best_x.copy()]
        for _ in range(2):
            pert = best_x + rng.normal(0, 1.0, d)
            pert = np.clip(pert, -5, 5)
            start_points.append(pert)
        
        cma_budget = int(total_budget * 0.55)
        cma_calls_remaining = cma_budget - calls_used
        
        for i, x_init in enumerate(start_points):
            if calls_used >= cma_budget or time.time() > deadline:
                break
            
            remaining_for_cma = cma_budget - calls_used
            if remaining_for_cma < 20:
                break
            
            # Adaptive sigma
            sigma_init = 1.0 if i == 0 else 0.5
            
            x_best, f_best, calls_this_run = self._cma_es_run(
                func, x_init, sigma_init, remaining_for_cma, rng, deadline, global_best_tracker
            )
            
            calls_used += calls_this_run
            
            if f_best < best_f:
                best_f = f_best
                best_x = x_best.copy()
        
        # Phase 3: Nelder-Mead polish
        if calls_used < total_budget and time.time() < deadline:
            polish_budget = min(total_budget - calls_used, 100)
            if polish_budget > 10:
                x_best, f_best, calls_polish = self._nelder_mead(
                    func, best_x, polish_budget, deadline, global_best_tracker
                )
                calls_used += calls_polish
                
                if f_best < best_f:
                    best_f = f_best
                    best_x = x_best.copy()
        
        # Final safety check
        if calls_used == 0:
            x0 = np.zeros(d)
            f0 = func(x0)
            best_x = x0
            best_f = f0
            calls_used = 1
        
        return best_x, best_f
