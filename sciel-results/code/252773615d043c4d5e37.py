"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Hybrid Differential Evolution and (1+1)-ES optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        
        # Phase 1: Differential Evolution for global exploration
        # We allocate 60% of the budget to DE.
        de_budget = int(self.budget * 0.7)
        pop_size = 20
        # Ensure pop_size is not larger than de_budget
        if pop_size > de_budget:
            pop_size = max(2, de_budget)
            
        # Initialize population
        population = rng.uniform(-5, 5, (pop_size, self.dim))
        fitnesses = np.array([func(ind) for ind in population])
        
        # DE Parameters
        F = 0.8
        CR = 0.9
        
        # Run DE for remaining DE budget
        calls_made = pop_size
        while calls_made < de_budget:
            # Determine how many individuals we can process in this generation
            # We process one individual at a time to strictly control budget
            # But DE usually processes the whole population. 
            # To strictly adhere to budget, we will iterate through individuals 
            # and stop if we exceed budget.
            
            # For simplicity and strict budget control, we do a "semi-online" DE
            # or just run full generations if they fit.
            # Since pop_size=20, and we have ~300 calls, we can do ~15 generations.
            # Let's just do full generations if they fit.
            
            if calls_made + pop_size > de_budget:
                # If a full generation doesn't fit, we stop DE phase
                break
                
            # Update population
            new_fitnesses = fitnesses.copy()
            for i in range(pop_size):
                # Select two distinct random indices
                r1, r2 = rng.choice(pop_size, 2, replace=False)
                while r1 == i or r2 == i:
                    r1, r2 = rng.choice(pop_size, 2, replace=False)
                
                # Create trial vector
                v = population[i] + F * (population[r1] - population[r2])
                v = np.clip(v, -5, 5)
                
                fv = func(v)
                calls_made += 1
                
                # Selection
                if fv < fitnesses[i]:
                    population[i] = v
                    new_fitnesses[i] = fv
            
            fitnesses = new_fitnesses

        # Find the best individual from DE phase
        best_idx = np.argmin(fitnesses)
        x_best = population[best_idx].copy()
        f_best = fitnesses[best_idx]
        
        # Phase 2: (5+10)-ES for local refinement
        # Population-based ES is more robust to the fractal landscape of Weierstrass
        # than (1+1)-ES, as it maintains diversity in the local neighborhood.
        remaining_budget = self.budget - calls_made
        if remaining_budget > 0:
            mu = 5
            lam = 10
            # Ensure we can at least run one generation
            if remaining_budget < lam:
                mu = max(1, remaining_budget // 2)
                lam = remaining_budget // mu if mu > 0 else 1
                if mu * lam > remaining_budget:
                    mu = 1
                    lam = remaining_budget
            
            # Initialize population around best DE solution
            # Start with a small sigma to refine, but large enough to explore local basin
            sigma = 0.5
            # Create initial population
            # 1 is the best, others are perturbations
            pop = np.empty((mu, self.dim))
            pop[0] = x_best
            f_pop = np.empty(mu)
            f_pop[0] = f_best
            
            for i in range(1, mu):
                pop[i] = x_best + sigma * rng.standard_normal(self.dim)
                pop[i] = np.clip(pop[i], -5, 5)
                f_pop[i] = func(pop[i])
                calls_made += 1
            
            # Sort initial population
            idx = np.argsort(f_pop)
            pop = pop[idx]
            f_pop = f_pop[idx]
            
            # Run ES generations
            while calls_made + lam <= remaining_budget:
                # Generate offspring
                off = np.empty((lam, self.dim))
                f_off = np.empty(lam)
                
                for j in range(lam):
                    # Select parent (rank-based, simple random from top mu)
                    p_idx = rng.integers(0, mu)
                    # Mutate
                    off[j] = pop[p_idx] + sigma * rng.standard_normal(self.dim)
                    off[j] = np.clip(off[j], -5, 5)
                    f_off[j] = func(off[j])
                    calls_made += 1
                
                # Combine parent and offspring
                all_pop = np.vstack([pop, off])
                all_fit = np.concatenate([f_pop, f_off])
                
                # Select best mu
                idx = np.argsort(all_fit)[:mu]
                pop = all_pop[idx]
                f_pop = all_fit[idx]
                
                # Adapt sigma
                # Simple adaptation: if best improved, increase slightly, else decrease
                # Or use success rate. For simplicity, let's use a basic rule:
                # If the best offspring is better than the previous best parent, increase sigma.
                prev_best = f_pop[0]
                # We need to track previous best before update? 
                # Actually, f_pop is already updated to the new best.
                # Let's keep a variable for the best fitness from the previous generation's parents
                # We didn't store it explicitly, but we can infer or just use a fixed schedule?
                # Better: track success.
                # Let's stick to a simple heuristic:
                # If the new best is significantly better, keep sigma or increase.
                # If not, decrease.
                # Since we don't have the "previous" best easily accessible without storing, 
                # let's store it.
                
                # Re-implementing loop to track prev_best
                pass # This block is getting complex, let's simplify the adaptation.
                
                # Simplified Sigma Adaptation:
                # If the best of the offspring is better than the best of the parents (from before selection),
                # it suggests the step size might be good or too small.
                # Let's just use a fixed decay or a simple success-based rule.
                # Success: number of offspring better than the worst parent?
                # Let's use: if best_off < worst_parent, success.
                
                # Let's restart the loop logic with proper tracking.
                break # Break to use a cleaner implementation below

            # The above loop was messy. Let's replace the whole Phase 2 with a cleaner (mu+lambda) ES implementation.
            
            # --- Clean (mu+lambda) ES Implementation ---
            mu = 5
            lam = 10
            if remaining_budget < lam:
                 # Fallback to (1+1) if budget is too small for population
                 sigma = 0.5
                 x = x_best
                 fx = f_best
                 for _ in range(remaining_budget):
                     y = x + sigma * rng.standard_normal(self.dim)
                     y = np.clip(y, -5, 5)
                     fy = func(y)
                     if fy < fx:
                         x, fx = y, fy
                         sigma *= 1.2
                     else:
                         sigma *= 0.9
                 x_best, f_best = x, fx
                 return x_best, f_best

            # Initialize population
            pop = np.empty((mu, self.dim))
            f_pop = np.empty(mu)
            pop[0] = x_best
            f_pop[0] = f_best
            for i in range(1, mu):
                pop[i] = x_best + 0.5 * rng.standard_normal(self.dim)
                pop[i] = np.clip(pop[i], -5, 5)
                f_pop[i] = func(pop[i])
                calls_made += 1
                
            idx = np.argsort(f_pop)
            pop = pop[idx]
            f_pop = f_pop[idx]
            
            sigma = 0.5
            
            while calls_made + lam <= self.budget:
                # Generate offspring
                off = np.empty((lam, self.dim))
                f_off = np.empty(lam)
                
                for j in range(lam):
                    p_idx = rng.integers(0, mu)
                    off[j] = pop[p_idx] + sigma * rng.standard_normal(self.dim)
                    off[j] = np.clip(off[j], -5, 5)
                    f_off[j] = func(off[j])
                    calls_made += 1
                
                # Combine and Select
                all_pop = np.vstack([pop, off])
                all_fit = np.concatenate([f_pop, f_off])
                
                # Track success for sigma adaptation
                # Success if best offspring is better than best parent
                best_parent_fit = f_pop[0]
                best_off_fit = np.min(f_off)
                
                idx = np.argsort(all_fit)[:mu]
                pop = all_pop[idx]
                f_pop = all_fit[idx]
                
                # Adapt sigma
                if best_off_fit < best_parent_fit:
                    sigma *= 1.1
                else:
                    sigma *= 0.9
                # Clamp sigma
                sigma = np.clip(sigma, 1e-4, 5.0)
                
            # Final best
            best_idx = np.argmin(f_pop)
            x_best = pop[best_idx]
            f_best = f_pop[best_idx]

        return x_best, f_best
# EVOLVE-BLOCK-END
