"""Optimized solver for BBOB f18 (Schaffers F7 ill-conditioned)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES with adaptive restarts for Schaffers F7."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin before 330s kill
        
        calls = 0
        best_f = np.inf
        best_x = np.zeros(dim)
        rng = np.random.default_rng(42)

        def counting_func(x):
            nonlocal calls, best_f, best_x
            if calls >= budget:
                return best_f
            calls += 1
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        # Phase 1: Random sampling to find a good initial basin
        n_random = min(20, budget // 10)
        for _ in range(n_random):
            if time.time() > deadline or calls >= budget:
                break
            x = rng.uniform(-5, 5, size=dim)
            counting_func(x)

        # Phase 2: CMA-ES
        # Parameters
        pop_size = 4 + int(3 * np.log(dim))  # Standard lambda
        if pop_size < 4:
            pop_size = 4
        
        # Initialize CMA-ES state
        mean = best_x.copy()
        sigma = 1.0
        C = np.eye(dim)
        C_sqrt = np.eye(dim)
        
        # Strategy parameters
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / (dim ** 1.7 + 10 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / (dim ** 2 + 18))
        
        # Initial step size
        sigma = 1.0
        
        # BFGS update of C and C_sqrt
        def update_C(C, C_sqrt, B, D, p_sigma, p_c):
            # This is a simplified update, standard CMA-ES uses more complex logic
            # For dim=5, we can afford full matrix operations
            pass

        # We will implement a simpler version of CMA-ES steps inline to avoid complex helper functions
        # and ensure strict budget adherence.
        
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        
        while calls < budget and time.time() < deadline:
            # Generate population
            ys = []
            xs = []
            fs = []
            
            for i in range(pop_size):
                if calls >= budget or time.time() > deadline:
                    break
                y = rng.standard_normal(dim)
                # Transform y using C_sqrt
                x = mean + sigma * (C_sqrt @ y)
                fx = counting_func(x)
                ys.append(y)
                xs.append(x)
                fs.append(fx)
            
            if not xs:
                break
                
            # Sort by fitness
            indices = np.argsort(fs)
            xs = [xs[i] for i in indices]
            ys = [ys[i] for i in indices]
            fs = [fs[i] for i in indices]
            
            # Update mean
            new_mean = np.zeros(dim)
            for i in range(mu):
                new_mean += weights[i] * xs[i]
            mean = new_mean
            
            # Update sigma (CSA)
            norm_y = np.linalg.norm(ys[0])
            h_sig = 1 if np.linalg.norm(ps / np.sqrt(1 - (1 - c_sigma) ** 2)) < \
                      (1 + 2 / (dim + 1)) * np.sqrt((mu_eff + 1) / (mu_eff - 1)) - 1 else 0
            ps = (1 - c_sigma) * ps + h_sig * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * \
                 (C_sqrt @ ys[0]) / np.linalg.norm(C_sqrt @ ys[0] + 1e-10)
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(mu_eff) - 1))
            
            # Update covariance
            pc = (1 - c_c) * pc + h_sig * c_c * (mean - new_mean) / sigma # Approximation, usually uses difference
            # Correct update for pc uses the step taken
            # Actually, pc update is: pc = (1-c_c)*pc + h_c * c_c * (mean_new - mean_old)/sigma
            # Let's stick to a simpler rank-one update if full CMA is too complex to implement correctly in short time
            # For dim=5, let's just do a simple adaptation:
            # If we have a lot of budget left, we can just use the population to estimate gradient direction?
            # No, let's implement the standard rank-one and rank-mu updates.
            
            # Re-calculate pc update properly
            # pc = (1 - c_c) * pc + h_c * c_c * (mean - mean_old) / sigma
            # We didn't store mean_old. Let's assume mean changed by step.
            # Actually, standard CMA-ES:
            # y_diff = mean - mean_old
            # pc = (1-c_c)*pc + h_c * c_c * y_diff / sigma
            
            # Since we don't have mean_old easily accessible in this loop structure without storing it,
            # let's store it.
            
            # Let's restart the loop logic with proper state tracking to be safe.
            # The above inline logic is getting messy. Let's use a cleaner structure.
            
            if calls >= budget or time.time() > deadline:
                break

            # Update C (Rank-One + Rank-Mu)
            # This requires storing mean_old. Let's refactor the loop to store it.
            # For now, let's just do a simple restart if stagnation is detected,
            # but CMA-ES is better.
            
            # Given the complexity and risk of bugs in a hand-written CMA-ES within the edit constraint,
            # and the fact that (1+1)-ES scored 1.18, a Multi-Start (1+1)-ES with better restarts
            # or a simpler CMA-ES approximation might be safer.
            # However, Inspiration 1 (CMA-ES) scored 1.65.
            
            # Let's implement a robust, simple CMA-ES.
            break # Exit this messy loop and use a cleaner implementation below?
            # No, I must replace the whole block. Let's rewrite the __call__ method completely with a clean CMA-ES.

        return best_x, best_f

    def _run_cma_es(self, func, start_x, sigma, budget, deadline, calls_counter, best_state, rng):
        dim = self.dim
        pop_size = 4 + int(3 * np.log(dim))
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / (dim ** 1.7 + 10 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / (dim ** 2 + 18))
        
        mean = start_x.copy()
        C = np.eye(dim)
        C_sqrt = np.eye(dim)
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        
        def counting_func(x):
            nonlocal calls_counter, best_state
            if calls_counter[0] >= budget:
                return best_state['f']
            calls_counter[0] += 1
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            if fx < best_state['f']:
                best_state['f'] = fx
                best_state['x'] = x_clipped.copy()
            return fx

        while calls_counter[0] < budget and time.time() < deadline:
            ys = []
            xs = []
            fs = []
            
            for _ in range(pop_size):
                if calls_counter[0] >= budget or time.time() > deadline:
                    break
                y = rng.standard_normal(dim)
                x = mean + sigma * (C_sqrt @ y)
                fx = counting_func(x)
                ys.append(y)
                xs.append(x)
                fs.append(fx)
                
            if not xs:
                break
                
            indices = np.argsort(fs)
            xs = [xs[i] for i in indices]
            ys = [ys[i] for i in indices]
            fs = [fs[i] for i in indices]
            
            mean_old = mean.copy()
            new_mean = np.zeros(dim)
            for i in range(mu):
                new_mean += weights[i] * xs[i]
            mean = new_mean
            
            # Update sigma
            norm_y = np.linalg.norm(ys[0])
            h_sig = 1 if np.linalg.norm(ps / np.sqrt(1 - (1 - c_sigma) ** 2)) < \
                      (1 + 2 / (dim + 1)) * np.sqrt((mu_eff + 1) / (mu_eff - 1)) - 1 else 0
            ps = (1 - c_sigma) * ps + h_sig * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * \
                 (C_sqrt @ ys[0]) / (np.linalg.norm(C_sqrt @ ys[0]) + 1e-10)
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(mu_eff) - 1))
            
            # Update pc
            h_c = 1 if np.linalg.norm(pc / np.sqrt(1 - (1 - c_c) ** 2)) < \
                    (1 + 2 / (dim + 1)) * np.sqrt((mu_eff + 1) / (mu_eff - 1)) - 1 else 0
            pc = (1 - c_c) * pc + h_c * c_c * (mean - mean_old) / (sigma + 1e-10)
            
            # Update C
            C = (1 + c_1 * h_c * (np.outer(pc, pc) - C) + c_mu * np.sum(weights[i] * np.outer(ys[i], ys[i]) for i in range(mu))) * (1 - c_1) + \
                c_1 * (np.outer(pc, pc) - C) + c_mu * np.sum(weights[i] * np.outer(ys[i], ys[i]) for i in range(mu))
            
            # This C update formula is incorrect/simplified. 
            # Standard: C = (1 - c_1 - c_mu) * C + c_1 * (pc*pc' + (1-h_c)*C) + c_mu * sum(w_i * y_i*y_i')
            # Let's use the standard formula:
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + (1 - h_c) * C) + \
                c_mu * np.sum(weights[i] * np.outer(ys[i], ys[i]) for i in range(mu))
            
            # Update C_sqrt (Cholesky or SVD)
            # Use SVD for stability
            U, s, Vt = np.linalg.svd(C)
            C_sqrt = U @ np.diag(np.sqrt(s)) @ Vt
            
            # Stop if sigma is too small
            if sigma < 1e-12:
                break

        return best_state['x'], best_state['f']

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280
        
        calls_counter = [0]
        best_state = {'f': np.inf, 'x': np.zeros(dim)}
        rng = np.random.default_rng(42)

        def counting_func(x):
            if calls_counter[0] >= budget:
                return best_state['f']
            calls_counter[0] += 1
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            if fx < best_state['f']:
                best_state['f'] = fx
                best_state['x'] = x_clipped.copy()
            return fx

        # Phase 1: Random sampling
        n_random = min(20, budget // 10)
        for _ in range(n_random):
            if time.time() > deadline or calls_counter[0] >= budget:
                break
            x = rng.uniform(-5, 5, size=dim)
            counting_func(x)

        # Phase 2: CMA-ES
        if calls_counter[0] < budget and time.time() < deadline:
            self._run_cma_es(func, best_state['x'], 1.0, budget, deadline, calls_counter, best_state, rng)

        return best_state['x'], best_state['f']
# EVOLVE-BLOCK-END
