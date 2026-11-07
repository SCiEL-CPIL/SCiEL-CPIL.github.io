"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320  # 10s safety margin before 330s kill
        
        # CMA-ES parameters
        dim = self.dim
        # Population size: 4 + floor(3 * log(dim))
        lam = 4 + int(3 * np.log(dim))
        mu = lam // 2
        
        # Initial step size
        # Heuristic for Rosenbrock: start moderately small to navigate the valley
        sigma = 1.0 
        
        # Initial mean (center of search)
        # Uniform initialization within the box
        rng = np.random.default_rng(0)
        mean = rng.uniform(-5, 5, dim)
        
        # Initial covariance matrix (identity)
        C = np.eye(dim)
        # Eigenvalue decomposition for sqrt(C)
        D, B = np.linalg.eigh(C)
        sqrtC = B @ np.diag(np.sqrt(np.maximum(D, 1e-10)))
        
        # Cumulation parameters
        # Standard CMA-ES weights
        weights = np.log(lam + 0.5) - np.log(np.arange(1, lam + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights[:mu]**2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # State for best solution
        best_x = mean.copy()
        best_f = func(best_x)
        
        calls_made = 1
        
        # Main loop
        while calls_made + lam <= self.budget:
            # Check time
            if time.time() > deadline:
                break
                
            # Generate population
            ys = []
            xs = []
            fs = []
            
            for i in range(lam):
                z = rng.standard_normal(dim)
                y = sqrtC @ z
                x = mean + sigma * y
                x = np.clip(x, -5, 5)
                f = func(x)
                
                ys.append(y)
                xs.append(x)
                fs.append(f)
                
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                    
                calls_made += 1
                
                # If we hit budget exactly mid-iteration, we can't do a full update
                # but we can still record this point. 
                # However, CMA-ES requires a full population to update.
                # If calls_made + (lam - i - 1) > budget, we should break out
                # and not update CMA-ES, just return best.
                if calls_made + (lam - i - 1) > self.budget:
                    break
            
            if calls_made + (lam - len(ys)) > self.budget:
                # We didn't complete the population
                break
            
            # Sort by fitness
            sorted_indices = np.argsort(fs)
            ys = [ys[i] for i in sorted_indices]
            xs = [xs[i] for i in sorted_indices]
            fs = [fs[i] for i in sorted_indices]
            
            # Update mean
            mean = np.sum([weights[i] * xs[i] for i in range(mu)], axis=0)
            
            # Update sigma (CSA)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (sqrtC @ np.mean([weights[i] * ys[i] for i in range(mu)], axis=0))
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_made // lam + 1)))) < (1.4 + 2 / (dim + 1)) * np.trace(C) / dim
            
            # Note: We don't have previous sigma trace easily here without storing history, 
            # but for simplicity and robustness in a short budget, we can use a simplified update
            # or just rely on the standard formula assuming trace(C) ~ dim initially.
            # A more robust check:
            # E[sqrt(C)] is approx sqrt(dim) initially.
            # Let's use a standard heuristic if hsig is true: sigma *= exp(c_sigma/d_sigma * (||ps||/sqrt(dim) - 1))
            
            if hsig:
                ps_norm = np.linalg.norm(ps)
                # Approximate E[||z||] where z ~ N(0, C)
                # For identity C, E[||z||] = sqrt(dim) * sqrt(2/dim) * Gamma((dim+1)/2) / Gamma(dim/2)
                # Simplified: sqrt(dim) is a good approximation for large dim, but dim=5 is small.
                # Let's use the exact expectation for standard normal:
                # E[||z||] = sqrt(2) * Gamma((dim+1)/2) / Gamma(dim/2)
                # For dim=5: sqrt(2) * Gamma(3) / Gamma(2.5) = 1.414 * 2 / 1.329 = 2.12
                # Let's precompute this
                # Actually, simpler: just use the standard CMA-ES update which relies on the ratio.
                # If we don't track the exact expected norm, we can use a heuristic.
                # Given the short budget, let's stick to a robust simplified sigma update:
                # sigma *= exp(c_sigma / d_sigma * (ps_norm / np.sqrt(dim) - 1))
                # This is an approximation but works well enough for 500 calls.
                
                # Better: Calculate expected norm of N(0, I)
                # For dim=5, E[||z||] ~ 2.12
                # Let's hardcode or calculate
                # import math
                # exp_norm = math.sqrt(2) * math.gamma((dim+1)/2.0) / math.gamma(dim/2.0)
                
                # To avoid importing math again, let's use a lookup or approximation.
                # Or just use sqrt(dim) which is 2.23 for dim=5. Close enough.
                
                sigma *= np.exp(c_sigma / d_sigma * (ps_norm / np.sqrt(dim) - 1.0))
                sigma = np.clip(sigma, 1e-10, 10.0) # Prevent explosion or vanishing
            else:
                # If hsig is false, typically sigma is not updated or updated less aggressively
                # Standard CMA-ES: if hsig is false, ps is still updated but sigma update is skipped?
                # No, hsig controls if ps is used for pc update.
                # Sigma update is always done in standard CMA-ES, but the condition hsig is for pc.
                # Wait, standard CMA-ES:
                # ps is always updated.
                # sigma is always updated based on ps.
                # pc is updated only if hsig is true.
                # So the code above should be:
                # ps = ... (always)
                # sigma *= ... (always)
                # if hsig: pc = ...
                
                # Let's correct the logic:
                pass # Sigma update was done above inside 'if hsig'. 
                # Actually, standard CMA-ES updates sigma regardless of hsig.
                # hsig only gates the pc update.
                # Let's re-structure the update section.

            # Re-doing the update logic correctly:
            
            # 1. Mean update (done above)
            
            # 2. Sigma update
            # ps was updated in the block above? No, I put it inside. Let's move it out.
            
            # Let's restart the update block cleanly.
            
            # --- Correct Update Sequence ---
            
            # Mean
            mean = np.sum([weights[i] * xs[i] for i in range(mu)], axis=0)
            
            # ps
            y_mean = np.sum([weights[i] * ys[i] for i in range(mu)], axis=0)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (sqrtC @ y_mean)
            
            # hsig
            # We need the expected norm of a standard normal vector of dim 'dim'
            # E[||z||] for z ~ N(0, I_dim)
            # Using approximation or precomputed value for dim=5
            # For dim=5, E[||z||] ≈ 2.12
            # Let's use a generic approximation: sqrt(dim) * (1 - 1/(4*dim) + ...) ?
            # Or just calculate it once.
            # Since we can't import math easily in the loop without overhead, 
            # and dim is fixed, let's use a constant.
            # E[||z||] = sqrt(2) * Gamma((dim+1)/2) / Gamma(dim/2)
            # dim=5: sqrt(2) * Gamma(3) / Gamma(2.5) = 1.4142 * 2 / 1.3293 = 2.123
            exp_norm = 2.123 if dim == 5 else np.sqrt(dim) # Fallback
            
            # Correction: sqrtC @ y_mean is NOT the step in standard normal space.
            # The term in ps update is: sqrt(c_sigma(2-c_sigma)mu_eff) * (sqrt(C)^{-1} * (mean_new - mean_old)) / sigma
            # But we have ys which are standard normals.
            # The update for ps is: ps = (1-c_sigma)ps + sqrt(c_sigma(2-c_sigma)mu_eff) * (sqrt(C)^{-1} (x_mean - x_old_mean) / sigma)
            # Note: x_mean - x_old_mean = sigma * sqrt(C) @ y_mean
            # So (x_mean - x_old_mean) / sigma = sqrt(C) @ y_mean
            # Then sqrt(C)^{-1} (sqrt(C) @ y_mean) = y_mean
            # So ps = (1-c_sigma)ps + sqrt(c_sigma(2-c_sigma)mu_eff) * y_mean
            # My previous line: ps = ... + ... * (sqrtC @ y_mean) was WRONG.
            # It should be just y_mean.
            
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_mean
            
            # hsig
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_made // lam + 1)))) < (1.4 + 2 / (dim + 1)) * exp_norm / np.sqrt(dim) * np.sqrt(np.trace(C)/dim)
            # Simplified hsig check often used:
            # hsig = np.linalg.norm(ps) < (1.4 + 2/(dim+1)) * exp_norm * np.sqrt(np.trace(C)/dim) / np.sqrt(dim) ?
            # Let's use the standard formula:
            # hsig = ||ps|| / sqrt(1-(1-c_sigma)^(2t)) < (1.4 + 2/(d+1)) * E||z|| * sqrt(tr(C)/d) / sqrt(d) ?
            # Actually, the condition is ||ps|| < (1.4 + 2/(d+1)) * E||N(0,C)||
            # E||N(0,C)|| = E||z|| * sqrt(tr(C)/d) * ... no.
            # E||N(0,C)||^2 = tr(C).
            # So E||N(0,C)|| is not simply sqrt(tr(C)).
            # However, for isotropic C, E||N(0,C)|| = sqrt(tr(C)) * (E||z|| / sqrt(d)).
            # So:
            # hsig = np.linalg.norm(ps) < (1.4 + 2/(dim+1)) * exp_norm * np.sqrt(np.trace(C)/dim)
            
            hsig = np.linalg.norm(ps) < (1.4 + 2 / (dim + 1)) * exp_norm * np.sqrt(np.trace(C) / dim)
            
            # Sigma update
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / exp_norm * np.sqrt(dim) / np.sqrt(np.trace(C)) - 1.0))
            # Simplified: sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / (exp_norm * np.sqrt(np.trace(C)/dim)) - 1.0))
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / (exp_norm * np.sqrt(np.trace(C) / dim)) - 1.0))
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # pc update
            if hsig:
                pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (sqrtC @ y_mean)
            else:
                pc = (1 - c_c) * pc
            
            # Covariance update
            # Rank-1
            v = mean - mean_prev # We need mean_prev
            # I didn't store mean_prev. Let's assume mean is the new one.
            # We need the difference between new mean and old mean.
            # Let's store mean_prev before updating mean.
            
            # This requires restructuring the loop slightly to store prev_mean.
            
            # Let's restart the loop body with proper state storage.
            pass

        return best_x, best_f

# REWRITE WITH CORRECT STATE MANAGEMENT
class Optimizer:
    """CMA-ES optimizer for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320
        
        dim = self.dim
        lam = 4 + int(3 * np.log(dim))
        mu = lam // 2
        
        weights = np.log(lam + 0.5) - np.log(np.arange(1, lam + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights[:mu]**2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        rng = np.random.default_rng(0)
        mean = rng.uniform(-5, 5, dim)
        C = np.eye(dim)
        D, B = np.linalg.eigh(C)
        sqrtC = B @ np.diag(np.sqrt(np.maximum(D, 1e-10)))
        
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        sigma = 1.0
        
        best_x = mean.copy()
        best_f = func(best_x)
        calls_made = 1
        
        # Precompute expected norm
        # For dim=5, E[||z||] ~ 2.12
        exp_norm = 2.123 if dim == 5 else np.sqrt(dim)
        
        while calls_made + lam <= self.budget:
            if time.time() > deadline:
                break
            
            # Generate
            ys = []
            xs = []
            fs = []
            
            for i in range(lam):
                z = rng.standard_normal(dim)
                y = sqrtC @ z
                x = mean + sigma * y
                x = np.clip(x, -5, 5)
                f = func(x)
                
                ys.append(y)
                xs.append(x)
                fs.append(f)
                
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                
                calls_made += 1
                
                if calls_made + (lam - i - 1) > self.budget:
                    break
            
            if calls_made + (lam - len(ys)) > self.budget:
                break
            
            # Sort
            sorted_indices = np.argsort(fs)
            ys = [ys[i] for i in sorted_indices]
            xs = [xs[i] for i in sorted_indices]
            
            # Update
            mean_old = mean.copy()
            mean = np.sum([weights[i] * xs[i] for i in range(mu)], axis=0)
            
            y_mean = np.sum([weights[i] * ys[i] for i in range(mu)], axis=0)
            
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_mean
            
            hsig = np.linalg.norm(ps) < (1.4 + 2 / (dim + 1)) * exp_norm * np.sqrt(np.trace(C) / dim)
            
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / (exp_norm * np.sqrt(np.trace(C) / dim)) - 1.0))
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            if hsig:
                pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (sqrtC @ y_mean)
            else:
                pc = (1 - c_c) * pc
            
            # Covariance update
            # Rank-1
            v = (mean - mean_old) / sigma
            C = (1 + c_1 * (hsig * (1 - c_c) - c_c) - c_1 - c_mu) * C + (c_1 + c_mu) * np.outer(v, v)
            
            # Rank-mu
            # v_i = (x_i - mean_old) / sigma
            # C += c_mu * sum(weights[i] * outer(v_i, v_i))
            for i in range(mu):
                v_i = (xs[i] - mean_old) / sigma
                C += c_mu * weights[i] * np.outer(v_i, v_i)
            
            # Ensure symmetry and positive semi-definite
            C = (C + C.T) / 2
            # Regularize if needed
            try:
                D, B = np.linalg.eigh(C)
                if np.any(D < 1e-10):
                    D = np.maximum(D, 1e-10)
                sqrtC = B @ np.diag(np.sqrt(D))
            except np.linalg.LinAlgError:
                # Reset if singular
                C = np.eye(dim)
                D, B = np.linalg.eigh(C)
                sqrtC = B @ np.diag(np.sqrt(D))
                
        return best_x, best_f

# EVOLVE-BLOCK-END
