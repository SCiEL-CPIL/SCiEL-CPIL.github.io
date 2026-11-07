"""Optimized program for WFG9 using MO-CMA-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """MO-CMA-ES based Multi-Objective Optimizer."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        center = (lo + hi) / 2.0
        range_ = hi - lo
        
        # CMA-ES parameters
        # Population size lambda
        # With budget 200, let's use lambda=20. 
        # Generations = 200 / 20 = 10.
        # This is low for CMA-ES to learn covariance, but let's try.
        # Maybe lambda=10? Generations=20.
        # Let's use lambda=12. Generations ~ 16.
        lambda_ = 12
        mu = lambda_ // 2  # Number of parents used for mean update
        
        # Initialize
        # Start with a uniform distribution to explore, then switch to CMA?
        # Or start with CMA centered at center.
        # If we start with CMA, we might miss parts of the front.
        # WFG9 front is curved.
        
        # Strategy: 
        # 1. Initial population: Uniform random to cover the space.
        # 2. Subsequent populations: CMA-ES sampling.
        # 3. Selection: Non-dominated sorting to pick parents for CMA-ES update.
        
        # However, standard CMA-ES updates mean and covariance based on the selected parents.
        # If we start with uniform, the initial CMA parameters should reflect the spread.
        
        # Let's initialize CMA parameters:
        # mu_vec (mean) = center
        # sigma = 0.5 * min(range_) ? Or range_/4?
        # C = I
        
        # Actually, to ensure good coverage initially, we can run a few generations of random search or use a large sigma.
        # Let's set initial sigma to 0.5 * range_ to cover the space.
        
        mean = center.copy()
        sigma = 0.5 * np.min(range_) # Start broad
        C = np.eye(self.dim)
        
        # Weights for CMA-ES
        # Standard weights: w_i = ln(mu/2 + 1) - ln(i) for i=1..mu, normalized
        weights = np.zeros(mu)
        for i in range(mu):
            weights[i] = np.log(mu/2.0 + 1.0) - np.log(i + 1.0)
        weights /= np.sum(weights)
        
        # Effective population size
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (self.dim + mu_eff + 5)
        c_c = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
        c_1 = 2 / ((self.dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2.0 / mu_eff) / ((self.dim + 2)**2 + mu_eff))
        
        # Damping
        d_sigma = 1.0 + 2 * max(0, np.sqrt((mu_eff - 1) / (self.dim + 1)) - 1) + c_sigma
        
        # State variables
        ps = np.zeros(self.dim)
        cs = 1.0
        
        # Population storage
        # We need to keep track of all evaluated points for the final return?
        # The task says "Every returned objective vector is recorded".
        # So we should return a list of all non-dominated points found?
        # Or just the final population?
        # "The return value of __call__ is ignored."
        # Wait, "Every returned objective vector is recorded... The return value of __call__ is ignored."
        # This implies the evaluator records the calls to `problem(x)`.
        # So I just need to make sure I don't exceed the budget and handle exceptions.
        # I don't need to return the front explicitly, but I should return something (e.g., None or the last pop).
        
        # However, to be safe, I will maintain an archive of non-dominated solutions found so far.
        # And I will return the archive (or the last population) just in case, though it's ignored.
        
        # Let's run generations
        gen = 0
        all_evaluated_F = []
        all_evaluated_X = []
        
        # Initial population: Uniform random
        X = np.zeros((lambda_, self.dim))
        F = np.zeros((lambda_, self.n_obj))
        
        for i in range(lambda_):
            X[i] = lo + range_ * np.random.rand(self.dim)
            F[i] = problem(X[i])
            all_evaluated_F.append(F[i])
            all_evaluated_X.append(X[i])
            
        evaluations = lambda_
        
        # Main loop
        while evaluations + lambda_ <= self.budget:
            if time.time() > deadline:
                break
            
            # 1. Sample offspring from CMA-ES
            # Z ~ N(0, C)
            # We need to sample from N(mean, sigma^2 C)
            
            # Cholesky decomposition of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback if C is not positive definite
                L = np.eye(self.dim)
                
            Z = np.random.randn(lambda_, self.dim)
            # Transform Z: Z @ L.T gives covariance C
            # x = mean + sigma * (Z @ L.T)
            X_offspring = mean[None, :] + sigma * (Z @ L.T)
            
            # Clip to bounds
            X_offspring = np.clip(X_offspring, lo, hi)
            
            # 2. Evaluate offspring
            F_offspring = np.zeros((lambda_, self.n_obj))
            for i in range(lambda_):
                F_offspring[i] = problem(X_offspring[i])
                all_evaluated_F.append(F_offspring[i])
                all_evaluated_X.append(X_offspring[i])
                evaluations += 1
                
            # 3. Selection
            # Combine current population? No, CMA-ES is generational.
            # But for MOO, we might want to keep the best from previous gen if the new one is bad?
            # Standard MO-CMA-ES usually replaces the population.
            # However, if the new population is dominated, we might stagnate.
            # Let's use a "elitist" strategy: Combine X and X_offspring, select best lambda.
            # This is more like NSGA-II but with CMA-ES sampling for the new candidates.
            
            # Actually, let's stick to pure CMA-ES generational replacement to allow covariance to adapt.
            # But we need to select parents from the offspring to update the mean.
            
            # Perform non-dominated sorting on X_offspring
            # Since lambda is small (12), we can do this efficiently.
            
            # Identify non-dominated sets
            # Front 0: Non-dominated in X_offspring
            # Front 1: Non-dominated in X_offspring \ Front 0
            # ...
            
            # We need to select 'mu' parents from X_offspring.
            # Usually, we take the first few fronts.
            
            # Helper: Non-dominated sort
            def non_dominated_sort(F_pop):
                n = len(F_pop)
                # S_i: set of indices dominated by i
                # n_i: number of solutions that dominate i
                S = [[] for _ in range(n)]
                n_dom = [0] * n
                fronts = []
                
                for i in range(n):
                    for j in range(n):
                        if i == j: continue
                        # i dominates j if i <= j and i != j
                        if (F_pop[i] <= F_pop[j]).all() and (F_pop[i] < F_pop[j]).any():
                            S[i].append(j)
                        # j dominates i if j <= i and j != i
                        if (F_pop[j] <= F_pop[i]).all() and (F_pop[j] < F_pop[i]).any():
                            n_dom[i] += 1
                            
                # Front 0
                front = [i for i in range(n) if n_dom[i] == 0]
                while front:
                    fronts.append(front)
                    next_front = []
                    for i in front:
                        for j in S[i]:
                            n_dom[j] -= 1
                            if n_dom[j] == 0:
                                next_front.append(j)
                    front = next_front
                return fronts
            
            fronts = non_dominated_sort(F_offspring)
            
            # Select parents
            parents_idx = []
            for f in fronts:
                if len(parents_idx) + len(f) <= mu:
                    parents_idx.extend(f)
                else:
                    # Need to pick from this front
                    needed = mu - len(parents_idx)
                    # Pick randomly or by crowding? Random is fine for small mu.
                    parents_idx.extend(np.random.choice(f, needed, replace=False))
                    break
            
            # If we still don't have enough (shouldn't happen), fill with random
            while len(parents_idx) < mu:
                parents_idx.append(np.random.randint(lambda_))
            
            # 4. Update CMA-ES parameters
            # Mean update
            # x_parents = X_offspring[parents_idx]
            x_parents = X_offspring[parents_idx]
            mean_new = np.sum(weights[:, None] * x_parents, axis=0)
            
            # Covariance update
            # y = (x_parents - mean) / sigma
            y = (x_parents - mean[None, :]) / sigma
            y = y @ L.T # Transform to standard normal space?
            # Actually, the update formula is:
            # C <- (1 - c_1 - c_mu) C + c_1 (ps ps^T + (1 - delta_sigma) C) + c_mu sum w_i y_i y_i^T
            # where y_i = (x_i - mean) / (sigma * L) ?
            # Let's use the standard formulation.
            
            # Calculate y_i in the "standard normal" space
            # If x = mean + sigma * L * z, then z = L^-1 * (x - mean) / sigma
            # L is Cholesky of C, so C = L L^T.
            # We want z such that Cov(z) = I.
            # x - mean = sigma L z
            # z = (1/sigma) L^-1 (x - mean)
            
            # We have L from Cholesky(C).
            # We need L_inv.
            try:
                L_inv = np.linalg.inv(L)
            except:
                L_inv = np.eye(self.dim)
                
            z_parents = (x_parents - mean[None, :]) / sigma
            z_parents = z_parents @ L_inv.T # (x-mean)/sigma @ L^-1.T = L^-1 (x-mean)/sigma ?
            # (x-mean) is row vector. (x-mean) @ L_inv.T = (L_inv @ (x-mean).T).T
            # So z_parents[i] = L_inv @ ((x_parents[i] - mean) / sigma)
            
            # Update ps
            # ps = (1 - c_sigma) ps + sqrt(c_sigma (2 - c_sigma) mu_eff) * (z_parents[0] - ...?)
            # The formula uses the weighted sum of z's?
            # Standard CMA-ES:
            # ps = (1-c_sigma)ps + sqrt(c_sigma(2-c_sigma)mu_eff) * (1/sigma) * L^-1 (x_best - mean)
            # Wait, the "x_best" is the weighted mean of the parents?
            # In CMA-ES, the step is determined by the weighted mean of the selected parents.
            # Let's call the weighted mean of parents in original space: m_parents
            # m_parents = sum w_i x_i
            # The step vector in standard normal space is:
            # z_step = (1/sigma) L^-1 (m_parents - mean)
            
            m_parents = np.sum(weights[:, None] * x_parents, axis=0)
            z_step = (m_parents - mean) / sigma
            z_step = z_step @ L_inv.T
            
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z_step
            
            # Update sigma
            # sigma = sigma * exp( c_sigma / d_sigma * ( ||ps|| / chi_n - 1 ) )
            chi_n = np.sqrt(self.dim) * (1 - 1/(4*self.dim) + 1/(21*self.dim**2))
            norm_ps = np.linalg.norm(ps)
            sigma *= np.exp((c_sigma / d_sigma) * (norm_ps / chi_n - 1.0))
            
            # Update C
            # Delta term
            delta_sigma = (1 - c_sigma) ** 2 - 1
            
            # C update
            C_new = (1 - c_1 - c_mu) * C
            C_new += c_1 * (np.outer(ps, ps) + delta_sigma * C)
            
            # Rank-mu update
            # Sum w_i z_i z_i^T
            # z_parents are the vectors for each parent
            # C_new += c_mu * sum( w_i * np.outer(z_parents[i], z_parents[i]) )
            
            # Efficient computation:
            # Z_w = z_parents * np.sqrt(weights[:, None])
            # C_new += c_mu * (Z_w.T @ Z_w)
            
            Z_w = z_parents * np.sqrt(weights[:, None])
            C_new += c_mu * (Z_w.T @ Z_w)
            
            # Symmetrize
            C_new = (C_new + C_new.T) / 2.0
            
            # Update state
            mean = mean_new
            C = C_new
            # Ensure C is positive definite? 
            # If it becomes singular, we might need to regularize.
            # For now, assume it stays PD.
            
            gen += 1
            # Optional: print progress? No.
            
        # Return the last population or archive. 
        # The return value is ignored, but let's return the last evaluated offspring.
        return X_offspring

    def _generate_ref_points(self, n_points, n_obj):
        # Not used in MO-CMA-ES, but kept for interface compatibility if called externally?
        # The task says "Keep the function names and signatures of the seed program."
        # The seed program had _generate_ref_points.
        # I should keep it, even if unused, or implement it.
        pts = np.random.rand(n_points, n_obj)
        sums = pts.sum(axis=1, keepdims=True)
        pts = pts / sums
        return pts

# EVOLVE-BLOCK-END
