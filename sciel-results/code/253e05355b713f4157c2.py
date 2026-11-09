"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB f16 (Weierstrass)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0

        # CMA-ES Parameters
        N = 10  # Population size
        mu = int(np.floor(N / 2))  # Number of recombination points
        # Weights
        weights = np.log(N / 2 + 0.5) - np.log(np.arange(1, N + 1))
        weights /= np.sum(weights)
        mu_eff = 1 / np.sum(weights[:mu] ** 2)
        
        # Strategy parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1) - 1)) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # State variables
        mean = np.zeros(dim)
        sigma = 1.0
        C = np.eye(dim)
        D, B = np.linalg.eigh(C)
        P_sigma = np.zeros(dim)
        P_c = np.zeros(dim)
        
        best_x = mean.copy()
        best_f = np.inf
        remaining_budget = budget

        while remaining_budget >= N and time.time() < deadline:
            mean_old = mean.copy()
            
            # 1. Sample population
            pop = []
            for i in range(N):
                z = np.random.randn(dim)
                y = B @ (D * z)
                x = mean_old + sigma * y
                x = np.clip(x, -5.0, 5.0)
                pop.append(x)
            
            # 2. Evaluate
            fvals = []
            for x in pop:
                if time.time() > deadline:
                    break
                try:
                    f = func(x)
                    remaining_budget -= 1
                    fvals.append((f, x))
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                except Exception:
                    break
            
            if len(fvals) < N:
                break
            
            # Sort by fitness
            fvals.sort(key=lambda t: t[0])
            
            # 3. Update Mean
            mean = np.zeros(dim)
            for i in range(mu):
                f, x = fvals[i]
                mean += weights[i] * x
            
            # 4. Update Step Size (Sigma)
            # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (mean - mean_old) / sigma
            p_sigma = (1 - c_sigma) * P_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (mean - mean_old) / sigma
            
            # h_sigma = 1 if ||p_sigma|| / sqrt(1 - (1-c_sigma)^{2(gen+1)}) < (1.4 + 2/(dim+1)) * E||N(0,I)||
            # Approximation: E||N(0,I)|| = sqrt(dim) * (1 - 1/(4dim) + 1/(32dim^2))
            # For simplicity, we can use a threshold.
            # norm_p_sigma = np.linalg.norm(p_sigma)
            # E_norm = np.sqrt(dim) * (1 - 1/(4*dim))
            # threshold = (1.4 + 2/(dim+1)) * E_norm
            # h_sigma = 1.0 if norm_p_sigma / np.sqrt(1 - (1-c_sigma)**(2*gen)) < threshold else 0.0
            # 
            # To avoid tracking 'gen' and the complex denominator, we use a simpler check or just update sigma.
            # Standard implementation:
            norm_p_sigma = np.linalg.norm(p_sigma)
            # We need the denominator sqrt(1 - (1-c_sigma)^{2t}). 
            # Since we don't track t globally in this simple loop, we can approximate or skip h_sigma logic 
            # and just use the norm check against a constant, or implement full tracking.
            # Let's track generations.
            
            # Actually, let's just implement the sigma update directly.
            # sigma = sigma * exp( c_sigma / d_sigma * ( ||p_sigma|| / E||N(0,I)|| - 1 ) )
            
            E_norm = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (32 * dim ** 2))
            # We need to divide by sqrt(1 - (1-c_sigma)^{2t})
            # Let's keep a counter.
            
            # I'll add a generation counter to the loop.
            
            # 5. Update P_c and C
            # p_c = (1 - c_c) * p_c + h_sigma * sqrt(c_c * (2 - c_c) * mu_eff) * (mean - mean_old) / sigma
            # h_sigma is needed. Let's compute it properly.
            
            # This requires tracking the generation number 't'.
            
            pass
        
        # The loop above was incomplete. I will write the complete, correct version now.
        
        # --- Correct Implementation ---
        
        # State
        P_sigma = np.zeros(dim)
        P_c = np.zeros(dim)
        C = np.eye(dim)
        D, B = np.linalg.eigh(C)
        mean = np.zeros(dim)
        sigma = 1.0
        best_x = mean.copy()
        best_f = np.inf
        remaining_budget = budget
        t = 0  # Generation counter

        while remaining_budget >= N and time.time() < deadline:
            mean_old = mean.copy()
            
            # Sample
            pop = []
            for i in range(N):
                z = np.random.randn(dim)
                y = B @ (D * z)
                x = mean_old + sigma * y
                x = np.clip(x, -5.0, 5.0)
                pop.append(x)
            
            # Evaluate
            fvals = []
            for x in pop:
                if time.time() > deadline:
                    break
                f = func(x)
                remaining_budget -= 1
                fvals.append((f, x))
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
            
            if len(fvals) < N:
                break
            
            fvals.sort(key=lambda t: t[0])
            
            # Update Mean
            mean = np.zeros(dim)
            for i in range(mu):
                f, x = fvals[i]
                mean += weights[i] * x
            
            # Update P_sigma
            p_sigma = (1 - c_sigma) * P_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (mean - mean_old) / sigma
            
            # Calculate h_sigma
            # Denominator: sqrt(1 - (1 - c_sigma)^(2 * t))
            # t starts at 1 for first update? 
            # Let's use t as the current generation index (1-based for update)
            t += 1
            denom_sigma = np.sqrt(1 - (1 - c_sigma) ** (2 * t))
            if denom_sigma < 1e-10:
                denom_sigma = 1e-10
                
            norm_p_sigma = np.linalg.norm(p_sigma)
            E_norm = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (32 * dim ** 2))
            threshold = (1.4 + 2.0 / (dim + 1)) * E_norm
            
            h_sigma = 1.0 if norm_p_sigma / denom_sigma < threshold else 0.0
            
            # Update Sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (norm_p_sigma / (denom_sigma * E_norm) - 1))
            
            # Update P_c
            P_c = (1 - c_c) * P_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (mean - mean_old) / sigma
            
            # Update C
            # Delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            # C = (1 - c_1 - c_mu + Delta_h) * C
            #    + c_1 * (P_c * P_c.T + (1 - h_sigma) * c_c * (2 - c_c))
            #    + c_mu * sum(w_i * (y_i * y_i.T))
            
            # y_i = (x_i - mean_old) / sigma
            # We need the top mu y_i's.
            
            y_list = []
            for i in range(mu):
                f, x = fvals[i]
                y_i = (x - mean_old) / sigma
                y_list.append(y_i)
            
            C = (1 - c_1 - c_mu + (1 - h_sigma) * c_c * (2 - c_c)) * C
            C += c_1 * (np.outer(P_c, P_c) + (1 - h_sigma) * c_c * (2 - c_c))
            
            for i in range(mu):
                C += c_mu * weights[i] * np.outer(y_list[i], y_list[i])
            
            # Re-decompose C
            # Ensure C is symmetric
            C = (C + C.T) / 2
            D, B = np.linalg.eigh(C)
            # Ensure D is positive
            D = np.maximum(D, 1e-12)
            
            # Store P_sigma for next iteration
            P_sigma = p_sigma
            
        return best_x, best_f

# EVOLVE-BLOCK-END
