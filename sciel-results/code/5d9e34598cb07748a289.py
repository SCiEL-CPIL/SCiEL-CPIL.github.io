"""Rewritten optimizer using CMA-ES with coordinate descent refinement for Schaffers F7."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with coordinate descent refinement for BBOB-style f18."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        start_time = time.time()
        time_limit = 300.0  # seconds, with safety margin
        
        rng = np.random.default_rng(42)
        
        # Initial point
        x = np.zeros(d)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        call_count = 1
        
        # CMA-ES parameters
        lambda_ = max(4, int(4 + 3 * np.log(d)))
        # Adjust lambda to fit budget
        max_generations = (budget - 1) // lambda_
        if max_generations < 5:
            lambda_ = max(2, (budget - 1) // 5)
            max_generations = (budget - 1) // lambda_
        
        # Initial sigma
        sigma = 2.0
        mean = x.copy()
        
        # Covariance matrix and evolution paths
        C = np.eye(d)
        Cs = np.zeros(d)
        Cinv = np.eye(d)
        
        # Learning rates
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        cc = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        cs = (mu_eff + 2) / (d + mu_eff + 5)
        c1 = 2 / ((d + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2) ** 2 + mu_eff))
        
        chiN = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))
        
        # Sort indices
        idx = np.arange(lambda_)
        
        for gen in range(max_generations):
            # Check time
            if time.time() - start_time > time_limit - 5.0:
                break
            if call_count >= budget:
                break
            
            # Sample population
            Y = np.zeros((lambda_, d))
            for i in range(lambda_):
                B = np.linalg.cholesky(C)
                z = rng.standard_normal(d)
                Y[i] = mean + sigma * (B @ z)
                Y[i] = np.clip(Y[i], -5, 5)
            
            # Evaluate
            F = np.zeros(lambda_)
            for i in range(lambda_):
                if call_count >= budget:
                    F[i] = F[i-1] if i > 0 else fx
                    break
                F[i] = func(Y[i])
                call_count += 1
                if F[i] < best_fx:
                    best_fx = F[i]
                    best_x = Y[i].copy()
            
            # Sort
            order = np.argsort(F)
            Y = Y[order]
            F = F[order]
            
            # Update mean
            mean = np.sum(weights[:, None] * Y[:mu], axis=0)
            
            # Update evolution paths
            y = (mean - (mean - sigma * np.sum(weights[:, None] * Y[:mu], axis=0) / sigma)) / sigma
            # Actually: y = (mean_new - mean_old) / sigma
            # Let me recompute properly
            mean_old = mean - np.sum(weights[:, None] * Y[:mu], axis=0)
            y = (mean - mean_old) / sigma
            
            Cs = (1 - cs) * Cs + cs * y
            delta_hs = (np.sqrt((1 + 2 * (cs / (1 - cs)) * (1 - cs)) * chiN) - (1 - cs) * chiN)
            hsig = int(np.linalg.norm(Cs) / (1 - (1 - cs) ** (gen + 1)) / chiN < 1.4 + 2 / (d + 1))
            
            C = (1 - cs * (c1 + cmu) + c1 * (2 - c1)) * C \
                + c1 * (np.outer(Cs, Cs) - C) \
                + cmu * np.sum(weights[:, None, None] * np.outer(Y[:mu] - mean_old, Y[:mu] - mean_old), axis=0) / sigma ** 2
            
            # Ensure positive definiteness
            C = (C + C.T) / 2
            try:
                Cinv = np.linalg.inv(C)
            except:
                Cinv = np.eye(d)
            
            # Update sigma
            sigma *= np.exp(cs / chiN * (np.linalg.norm(Cs) / chiN - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # Coordinate descent refinement every few generations
            if gen % 5 == 4 and call_count < budget - 10 and time.time() - start_time < time_limit - 10.0:
                for _ in range(3):
                    if call_count >= budget - 5:
                        break
                    if time.time() - start_time > time_limit - 5.0:
                        break
                    improved = False
                    for i in range(d):
                        if call_count >= budget - 5:
                            break
                        # Try small steps in coordinate i
                        for delta in [0.1, -0.1, 0.05, -0.05]:
                            if call_count >= budget:
                                break
                            x_try = best_x.copy()
                            x_try[i] += delta
                            x_try = np.clip(x_try, -5, 5)
                            fx_try = func(x_try)
                            call_count += 1
                            if fx_try < best_fx - 1e-12:
                                best_x = x_try
                                best_fx = fx_try
                                improved = True
                                break
        
        # Final coordinate descent
        if call_count < budget - 5 and time.time() - start_time < time_limit - 5.0:
            step = 0.01
            for _ in range(10):
                if call_count >= budget - 5 or time.time() - start_time > time_limit - 5.0:
                    break
                improved = False
                for i in range(d):
                    if call_count >= budget - 5:
                        break
                    for delta in [step, -step]:
                        if call_count >= budget:
                            break
                        x_try = best_x.copy()
                        x_try[i] += delta
                        x_try = np.clip(x_try, -5, 5)
                        fx_try = func(x_try)
                        call_count += 1
                        if fx_try < best_fx - 1e-12:
                            best_x = x_try
                            best_fx = fx_try
                            improved = True
                            break
                if not improved:
                    step *= 0.5
                    if step < 1e-4:
                        break
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
