"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import math


class Optimizer:
    """Hybrid Global-Local Optimizer: Differential Evolution + Nelder-Mead."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _nelder_mead(self, func, x0, budget, rng):
        """Run Nelder-Mead local search."""
        dim = self.dim
        # Initialize simplex
        x0 = np.array(x0, dtype=float)
        simplex = [x0.copy() for _ in range(dim + 1)]
        f_simplex = [func(x0) for _ in range(dim + 1)]
        calls = dim + 1
        
        # Perturb other points
        for i in range(1, dim + 1):
            step = 0.1 * (1 if x0[i-1] < 4.5 else -1)
            simplex[i] = x0.copy()
            simplex[i][i-1] += step
            simplex[i] = np.clip(simplex[i], -5, 5)
            f_simplex[i] = func(simplex[i])
            calls += 1
            
        # Sort initial simplex
        order = np.argsort(f_simplex)
        simplex = [simplex[i] for i in order]
        f_simplex = [f_simplex[i] for i in order]
        
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        best_x = simplex[0].copy()
        best_f = f_simplex[0]
        
        while calls < budget:
            # Calculate centroid of all points except the worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, -5, 5)
            f_r = func(x_r)
            calls += 1
            
            if f_r < f_simplex[0]:
                # Expansion
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, -5, 5)
                f_e = func(x_e)
                calls += 1
                if f_e < f_r:
                    simplex[-1] = x_e
                    f_simplex[-1] = f_e
                else:
                    simplex[-1] = x_r
                    f_simplex[-1] = f_r
            elif f_r < f_simplex[-2]:
                # Accept reflection
                simplex[-1] = x_r
                f_simplex[-1] = f_r
            else:
                # Contraction
                if f_r < f_simplex[-1]:
                    # Outside contraction
                    x_c = centroid + rho * (x_r - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    calls += 1
                    if f_c <= f_r:
                        simplex[-1] = x_c
                        f_simplex[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_simplex[i] = func(simplex[i])
                            calls += 1
                        if calls >= budget:
                            break
                else:
                    # Inside contraction
                    x_c = centroid - rho * (centroid - simplex[-1])
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    calls += 1
                    if f_c < f_simplex[-1]:
                        simplex[-1] = x_c
                        f_simplex[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_simplex[i] = func(simplex[i])
                            calls += 1
                        if calls >= budget:
                            break
            
            # Re-sort
            order = np.argsort(f_simplex)
            simplex = [simplex[i] for i in order]
            f_simplex = [f_simplex[i] for i in order]
            
            # Update best
            if f_simplex[0] < best_f:
                best_f = f_simplex[0]
                best_x = simplex[0].copy()
                
            # Convergence check
            if max(f_simplex) - min(f_simplex) < 1e-12:
                break
                
        return best_x, best_f, calls

    def _differential_evolution(self, func, x0, budget, rng):
        """Run Differential Evolution global search."""
        dim = self.dim
        pop_size = 10
        F = 0.8
        CR = 0.9
        
        # Initialize population
        population = []
        fitness = []
        
        # Include x0
        population.append(x0.copy())
        fitness.append(func(x0))
        
        # Generate rest randomly
        for _ in range(pop_size - 1):
            x = rng.uniform(-5, 5, dim)
            population.append(x)
            f = func(x)
            fitness.append(f)
            
        calls = pop_size
        
        while calls < budget:
            # Check if we can run a full generation
            if calls + pop_size > budget:
                # Run partial generation
                for i in range(pop_size):
                    if calls >= budget:
                        break
                    
                    # Select r1, r2, r3 distinct from i
                    indices = [j for j in range(pop_size) if j != i]
                    r1, r2, r3 = rng.choice(indices, size=3, replace=False)
                    
                    # Mutation
                    v = population[r1] + F * (population[r2] - population[r3])
                    
                    # Crossover
                    trial = population[i].copy()
                    j_rand = rng.integers(dim)
                    for j in range(dim):
                        if rng.random() < CR or j == j_rand:
                            trial[j] = v[j]
                    trial = np.clip(trial, -5, 5)
                    
                    # Evaluation
                    f_trial = func(trial)
                    calls += 1
                    
                    # Selection
                    if f_trial <= fitness[i]:
                        population[i] = trial
                        fitness[i] = f_trial
                        
            else:
                # Full generation
                for i in range(pop_size):
                    indices = [j for j in range(pop_size) if j != i]
                    r1, r2, r3 = rng.choice(indices, size=3, replace=False)
                    
                    # Mutation
                    v = population[r1] + F * (population[r2] - population[r3])
                    
                    # Crossover
                    trial = population[i].copy()
                    j_rand = rng.integers(dim)
                    for j in range(dim):
                        if rng.random() < CR or j == j_rand:
                            trial[j] = v[j]
                    trial = np.clip(trial, -5, 5)
                    
                    # Evaluation
                    f_trial = func(trial)
                    calls += 1
                    
                    # Selection
                    if f_trial <= fitness[i]:
                        population[i] = trial
                        fitness[i] = f_trial
                        
        # Find best
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx], calls

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # Split budget: 60% Global, 40% Local
        global_budget = int(budget * 0.6)
        local_budget = budget - global_budget
        
        # Phase 1: Global Search (DE)
        # Start with a random point
        x_start = rng.uniform(-5, 5, dim)
        best_x, best_f, _ = self._differential_evolution(func, x_start, global_budget, rng)
        
        # Phase 2: Local Polish (Nelder-Mead)
        final_x, final_f, _ = self._nelder_mead(func, best_x, local_budget, rng)
        
        return final_x, final_f
# EVOLVE-BLOCK-END
