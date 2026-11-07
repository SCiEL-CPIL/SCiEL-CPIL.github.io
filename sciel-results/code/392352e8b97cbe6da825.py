# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-start CMA-ES based black-box continuous optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        n = self.dim
        total_budget = self.budget
        
        best_x = np.zeros(n)
        best_f = float('inf')
        
        # Split budget into 3 phases
        # Phase 1: Explore broadly
        # Phase 2: Explore moderately
        # Phase 3: Polish
        phases = [100, 100, 100]
        
        for phase_idx, phase_budget in enumerate(phases):
            if total_budget <= 0:
                break
            
            # Initialize starting point
            if phase_idx == 0:
                x = np.zeros(n)
                sigma = 2.0
            else:
                # Restart from best found so far, but with some noise to escape local optima
                # Or pick a random point?
                # For multimodal, random restarts are better.
                # But we have limited budget.
                # Let's mix: 50% chance to start from best_x + noise, 50% random.
                if rng.random() < 0.5:
                    x = best_x + rng.normal(0, 0.5, n)
                else:
                    x = rng.uniform(-5, 5, n)
                sigma = 1.0
            
            x = np.clip(x, -5, 5)
            
            # CMA-ES parameters for this phase
            lam = 6
            mu = 3
            C = np.eye(n)
            evals, evec = np.linalg.eigh(C)
            
            chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
            cs = (mu + 2) / (n + mu + 5)
            ds = 1 + 2 * max(0, np.sqrt((mu + 1) / (n + 1) - 1)) + cs
            cc = (4 + n/mu) / (n + 4 + 2*mu/n)
            c1 = (2 + 2*min(mu/n - 1, 1/(n+1))) / ((n+1.3)**2 + mu)
            cmu = min(1 - c1, 2 * (mu - 2 + 1/mu) / ((n+2)**2 + mu))
            
            ps = np.zeros(n)
            pc = np.zeros(n)
            
            weights = np.zeros(mu)
            for i in range(mu):
                weights[i] = np.log(mu + 0.5) - np.log(i + 1)
            weights /= weights.sum()
            mueff = 1 / np.sum(weights**2)
            
            # Run CMA-ES for this phase
            while total_budget > 0:
                # Sample population
                D = np.sqrt(np.maximum(evals, 1e-20))
                B = evec
                N = rng.standard_normal((lam, n))
                Y = (B @ (D * N.T)).T
                pop = x + sigma * Y
                pop = np.clip(pop, -5, 5)
                
                # Evaluate
                fitnesses = np.zeros(lam)
                for i in range(lam):
                    if total_budget == 0: break
                    fitnesses[i] = func(pop[i])
                    total_budget -= 1
                    if fitnesses[i] < best_f:
                        best_f = fitnesses[i]
                        best_x = pop[i].copy()
                
                if total_budget == 0:
                    break
                
                # Sort by fitness
                order = np.argsort(fitnesses)
                pop_sorted = pop[order]
                
                # Mean update
                m_old = x.copy()
                x_new = np.sum(pop_sorted[:mu] * weights[:, None], axis=0)
                x = x_new
                
                # Step-size control
                z = (x - m_old) / (sigma * chi_n)
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * z
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**2)) < (1.4 + 2/(n+1)) * chi_n
                sigma *= np.exp(cs/ds * (np.linalg.norm(ps)/chi_n - 1))
                
                # Covariance matrix update
                y = (x - m_old) / sigma
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * y
                C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + cc*(1-cc)*C) + cmu * np.sum(
                    [np.outer(pop_sorted[i] - m_old, pop_sorted[i] - m_old) / (sigma**2) * weights[i] for i in range(mu)], axis=0
                )
                
                # Symmetrize
                C = 0.5 * (C + C.T)
                
                # Eigen decomposition
                evals, evec = np.linalg.eigh(C)
                evals = np.maximum(evals, 1e-20)
                
                # Clip x to box
                x = np.clip(x, -5, 5)
                
                # Check if we should stop early in this phase?
                # If sigma is very small, we are converged.
                if sigma < 1e-6:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
