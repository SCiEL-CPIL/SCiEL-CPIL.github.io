"""Single-run CMA-ES with adaptive restarts for BBOB f16 (Weierstrass)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES with adaptive restarts for multimodal optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill
        
        dim = self.dim
        budget = self.budget
        
        best_x = np.zeros(dim)
        best_f = np.inf
        n_calls = 0
        
        def check_time():
            return time.time() > deadline

        def call_func(x):
            nonlocal n_calls
            if n_calls >= budget or check_time():
                return None, None
            n_calls += 1
            try:
                x_clipped = np.clip(x, -5, 5)
                f = float(func(x_clipped))
                return f, x_clipped
            except Exception:
                return None, None

        def cma_es(x0, sigma0, max_calls, rng):
            """
            Runs CMA-ES.
            Returns (best_x, best_f, calls_used, stagnation_detected).
            """
            n = dim
            # Population size for dim=5: lambda = 4 + 3*ln(5) ~ 9
            lam = max(4, int(4 + 3 * np.log(n)))
            mu = lam // 2
            
            # Weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)
            
            # Learning rates
            cc = (4.0 + mueff / n) / (n + 4.0 + 2.0 * mueff / n)
            cs = (mueff + 2.0) / (n + mueff + 5.0)
            c1 = 2.0 / ((n + 1.3) ** 2 + mueff)
            cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n + 2.0) ** 2 + mueff))
            chi_n = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n ** 2))

            # State
            x = x0.copy()
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            
            local_best_x = x.copy()
            local_best_f = np.inf
            calls_used = 0
            stagnation_counter = 0
            no_improve_threshold = 5 * lam  # No improvement for 5 generations
            
            # Initial population
            pop = np.zeros((lam, n))
            fvals = np.zeros(lam)
            for i in range(lam):
                if calls_used >= max_calls or check_time():
                    return local_best_x, local_best_f, calls_used, True
                z = rng.standard_normal(n)
                x_trial = x + sigma * z
                f, x_eval = call_func(x_trial)
                if f is None:
                    return local_best_x, local_best_f, calls_used, True
                calls_used += 1
                pop[i] = x_eval
                fvals[i] = f
                if f < local_best_f:
                    local_best_f = f
                    local_best_x = x_eval.copy()
            
            # Main loop
            while calls_used < max_calls and not check_time():
                # Sort
                order = np.argsort(fvals)
                pop = pop[order]
                fvals = fvals[order]
                
                # Update mean
                x_new = np.sum(weights[:, None] * pop[:mu], axis=0)
                
                # Compute Cinv_sqrt
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-20, None)
                    Cinv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                except np.linalg.LinAlgError:
                    Cinv_sqrt = np.eye(n)

                # Update ps
                ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) * Cinv_sqrt @ (x_new - x)
                x = x_new

                # Update pc
                hsig = (np.linalg.norm(ps) / np.sqrt(1.0 - (1.0 - cs) ** (2 * (calls_used // lam)))) < (1.4 + 2.0 / (n + 1)) * chi_n
                pc = (1.0 - cc) * pc + hsig * np.sqrt(cc * (2.0 - cc) * mueff) * (x_new - x)

                # Update covariance
                C = (1.0 + c1 * hsig * (1.0 - hsig) - c1 - cmu) * C + c1 * np.outer(pc, pc)
                for i in range(mu):
                    y_i = (pop[i] - x) / sigma
                    C += cmu * weights[i] * np.outer(y_i, y_i)
                C = (C + C.T) / 2.0

                # Update sigma
                sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1.0))
                sigma = np.clip(sigma, 1e-15, 1e6)

                # Generate new population
                for i in range(lam):
                    if calls_used >= max_calls or check_time():
                        break
                    z = rng.standard_normal(n)
                    x_trial = x + sigma * (Cinv_sqrt @ z)
                    f, x_eval = call_func(x_trial)
                    if f is None:
                        return local_best_x, local_best_f, calls_used, True
                    calls_used += 1
                    pop[i] = x_eval
                    fvals[i] = f
                    if f < local_best_f:
                        local_best_f = f
                        local_best_x = x_eval.copy()
                        stagnation_counter = 0
                    else:
                        stagnation_counter += 1

                # Check for stagnation
                if stagnation_counter > no_improve_threshold:
                    return local_best_x, local_best_f, calls_used, True

                # Convergence check
                if sigma < 1e-10:
                    return local_best_x, local_best_f, calls_used, True

            return local_best_x, local_best_f, calls_used, True

        rng = np.random.default_rng(42)
        
        # Phase 1: Initial Sampling
        # Use a few diverse points to find a good starting location
        # Weierstrass has global minimum often near origin but shifted.
        # Let's try origin and a few random points.
        initial_candidates = [
            np.zeros(dim),
            rng.uniform(-2, 2, dim),
            rng.uniform(-2, 2, dim),
            rng.uniform(-2, 2, dim),
            rng.uniform(-4, 4, dim)
        ]
        
        # Evaluate initial candidates
        for x in initial_candidates:
            f, x_eval = call_func(x)
            if f is not None:
                if f < best_f:
                    best_f = f
                    best_x = x_eval.copy()
        
        if best_x is None or np.all(best_x == 0):
             # Fallback if evaluation failed or all zeros
             best_x = np.zeros(dim)
             best_f = 0.0

        # Phase 2: CMA-ES with Adaptive Restarts
        # Start with the best point found
        current_x = best_x.copy()
        current_sigma = 1.5  # Initial sigma
        restart_count = 0
        max_restarts = 5
        
        while n_calls < budget and not check_time() and restart_count < max_restarts:
            # Determine budget for this CMA-ES run
            # Give it a chunk of remaining budget, but leave some for restarts
            remaining = budget - n_calls
            if remaining < 10:
                break
            
            # Allocate 60% of remaining budget to current run, keep 40% for potential restarts
            # If this is the last possible run, use all remaining
            if restart_count == max_restarts - 1:
                alloc = remaining
            else:
                alloc = int(remaining * 0.6)
            
            # Minimum allocation to make sense
            if alloc < 20:
                alloc = remaining
            
            res_x, res_f, used, stagnated = cma_es(current_x, current_sigma, alloc, rng)
            
            if res_f < best_f:
                best_f = res_f
                best_x = res_x.copy()
            
            # Decide whether to restart
            if not stagnated or n_calls >= budget or check_time():
                break
                
            # If stagnated, restart
            restart_count += 1
            # Increase sigma for next restart to explore wider area
            current_sigma *= 1.5
            # Start from best point found so far
            current_x = best_x.copy()
            
            # If we are stuck in a local minimum, maybe try a random jump?
            # For Weierstrass, global min is often not too far.
            # Let's stick to best point with larger sigma first.
            # If restart_count > 2, try a random point
            if restart_count > 2:
                current_x = rng.uniform(-4, 4, dim)
                # Evaluate this new point
                f, x_eval = call_func(current_x)
                if f is not None and f < best_f:
                    best_f = f
                    best_x = x_eval.copy()
                current_sigma = 2.0

        # Phase 3: Local Polish
        # Use remaining budget for small step local search
        if n_calls < budget and not check_time():
            local_sigma = 0.1
            for _ in range(budget - n_calls):
                if check_time() or n_calls >= budget:
                    break
                x_trial = best_x + local_sigma * rng.standard_normal(dim)
                f, x_eval = call_func(x_trial)
                if f is not None:
                    if f < best_f:
                        best_f = f
                        best_x = x_eval.copy()
                        local_sigma *= 0.8
                    else:
                        local_sigma *= 0.95
                        if local_sigma < 1e-4:
                            break

        return best_x, best_f
# EVOLVE-BLOCK-END
