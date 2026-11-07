"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES based optimizer with adaptive restarts and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s limit
        
        best_x = None
        best_f = np.inf
        call_count = 0
        rng = np.random.default_rng(42)

        def make_call(x):
            nonlocal call_count
            if call_count >= self.budget:
                return None
            fx = func(x)
            call_count += 1
            return fx

        def cmaes_search(x0, sigma0, max_calls):
            """
            Runs a CMA-ES instance starting from x0 with initial sigma.
            Returns (best_x, best_f, calls_made)
            """
            nonlocal best_x, best_f, call_count
            
            n = self.dim
            x = x0.copy()
            sigma = sigma0
            
            # CMA-ES parameters
            # Standard parameters from Hansen
            # lambda: population size
            # Typically 4 + floor(3 ln n)
            lam = max(4 + int(3 * np.log(n)), 4)
            # mu: number of weights
            mu = lam // 2
            # Weights
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / np.sum(w)
            # Variance reduction factor
            mu_eff = 1.0 / np.sum(w**2)
            # Learning rates
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((n + 2)**2 + mu_eff))
            # Damping
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
            # Initial state
            C = np.eye(n)
            C_sqrt = np.eye(n)
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            
            # Initial population
            pop = []
            for i in range(lam):
                z = rng.standard_normal(n)
                xi = x + sigma * (C_sqrt @ z)
                xi = np.clip(xi, -5, 5)
                pop.append(xi)
                
            # Evaluate initial population
            fitness = []
            calls_in_run = 0
            for i in range(lam):
                fx = make_call(pop[i])
                if fx is None:
                    return x, np.inf, calls_in_run
                calls_in_run += 1
                if not np.isfinite(fx):
                    fx = np.inf
                fitness.append(fx)
                
            # Update best
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_f:
                best_f = fitness[min_idx]
                best_x = pop[min_idx].copy()
                
            # Loop
            while calls_in_run < max_calls:
                if call_count >= self.budget:
                    break
                if time.time() > deadline - 2.0:
                    break
                    
                # Sort population by fitness
                sorted_idx = np.argsort(fitness)
                y_sorted = [pop[i] for i in sorted_idx]
                f_sorted = [fitness[i] for i in sorted_idx]
                
                # Update mean
                x_new = sum(w[i] * y_sorted[i] for i in range(mu))
                
                # Update covariance matrix adaptation
                y_inv = C_sqrt @ (x_new - x)
                h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(-2 * calls_in_run))) / np.sqrt(1 - (1 - c_sigma)**(-2 * calls_in_run))
                # Correction for boundary clipping is ignored for simplicity, assuming box is large enough or clipping is rare in early stages
                # Actually, clipping can distort CMA-ES. We should be careful.
                # For now, standard update.
                
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_inv
                h_sig = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(-2 * calls_in_run))
                
                p_c = (1 - c_c) * p_c + h_sig * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_inv
                delta_h = (1 - c_c) * (1 - (1 - c_c)**(-2 * calls_in_run))
                
                rank_mu = np.zeros((n, n))
                for i in range(mu):
                    y_i = C_sqrt @ (y_sorted[i] - x)
                    rank_mu += w[i] * np.outer(y_i, y_i)
                    
                C = (1 + c_1 * (delta_h - 1) + c_mu) * C + c_1 * np.outer(p_c, p_c) - c_mu * rank_mu
                
                # Ensure symmetry
                C = (C + C.T) / 2
                
                # Update sigma
                sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(-2 * calls_in_run)) - np.sqrt(n)))
                
                # Update mean
                x = x_new
                
                # Update C_sqrt using eigendecomposition (expensive but robust for small n)
                # For n=5, this is fast.
                evals, evecs = np.linalg.eigh(C)
                # Clip eigenvalues to be positive
                evals = np.clip(evals, 1e-12, None)
                C_sqrt = evecs @ np.diag(np.sqrt(evals))
                
                # Generate new population
                pop = []
                for i in range(lam):
                    z = rng.standard_normal(n)
                    xi = x + sigma * (C_sqrt @ z)
                    xi = np.clip(xi, -5, 5)
                    pop.append(xi)
                    
                fitness = []
                for i in range(lam):
                    fx = make_call(pop[i])
                    if fx is None:
                        return x, np.inf, calls_in_run
                    calls_in_run += 1
                    if not np.isfinite(fx):
                        fx = np.inf
                    fitness.append(fx)
                    
                # Update best
                min_idx = np.argmin(fitness)
                if fitness[min_idx] < best_f:
                    best_f = fitness[min_idx]
                    best_x = pop[min_idx].copy()
                    
                # Check stagnation
                if calls_in_run > 100 and not any(f < best_f - 1e-8 for f in fitness):
                    # Simple stagnation check: if best fitness hasn't improved in last few generations
                    # This is a bit crude, but helps restart.
                    pass

            return x, np.min(fitness), calls_in_run

        # Strategy:
        # 1. Start from the center (origin) as a baseline.
        # 2. Run CMA-ES from there.
        # 3. If stagnation occurs or budget remains, restart from best found with a new random perturbation.

        # Initial start: center
        x_start = np.zeros(self.dim)
        fx_start = make_call(x_start)
        if fx_start is None:
            return best_x, best_f
        if np.isfinite(fx_start) and fx_start < best_f:
            best_f = fx_start
            best_x = x_start.copy()

        # First CMA-ES run
        sigma_init = 2.0
        # Allocate most of the budget to the first run, leaving some for potential restarts
        max_calls_first = 400
        x_best, f_best, calls_made = cmaes_search(x_start, sigma_init, max_calls_first)

        # If we have budget left, do additional restarts
        # Use the best point found so far as the base for new starts
        restart_count = 0
        max_restarts = 2

        while call_count < self.budget - 20 and restart_count < max_restarts:
            if time.time() > deadline - 5.0:
                break

            # Create a new starting point: best_x + random perturbation
            if best_x is not None:
                perturbation = rng.standard_normal(self.dim) * 0.5
                x_new_start = best_x + perturbation
                x_new_start = np.clip(x_new_start, -5, 5)
            else:
                x_new_start = rng.uniform(-3, 3, self.dim)

            # Use a slightly smaller sigma for exploitation
            sigma_restart = 1.0 if restart_count == 0 else 0.5
            max_calls_restart = min(50, self.budget - call_count - 20)

            if max_calls_restart <= 10:
                break

            x_best_r, f_best_r, calls_made_r = cmaes_search(x_new_start, sigma_restart, max_calls_restart)
            restart_count += 1

        # Final fallback
        if best_x is None:
            x = np.zeros(self.dim)
            fx = make_call(x)
            if fx is not None and np.isfinite(fx):
                best_x = x.copy()
                best_f = fx

        return best_x, best_f
# EVOLVE-BLOCK-END
