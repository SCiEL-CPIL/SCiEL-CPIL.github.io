"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Hybrid DE + Nelder-Mead optimizer for black-box continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        total_budget = self.budget
        lower = -5.0
        upper = 5.0
        
        # Phase 1: Differential Evolution
        # Allocate 70% of budget to DE
        de_budget = int(total_budget * 0.7)
        np_pop = 10  # Population size
        
        # Initialize population
        # Spread out the initial population
        pop = rng.uniform(lower, upper, size=(np_pop, n))
        fits = np.zeros(np_pop)
        
        eval_count = 0
        for i in range(np_pop):
            fits[i] = func(pop[i])
            eval_count += 1
            
        best_idx = np.argmin(fits)
        best_x = pop[best_idx].copy()
        best_f = fits[best_idx]
        
        F = 0.8  # Scale factor
        CR = 0.5  # Crossover rate
        
        while eval_count < de_budget:
            for i in range(np_pop):
                if eval_count >= de_budget:
                    break
                
                # Select three distinct random indices
                r1, r2, r3 = rng.choice(np_pop, size=3, replace=False)
                while r1 == r2 or r1 == r3 or r2 == r3:
                    r1, r2, r3 = rng.choice(np_pop, size=3, replace=False)
                
                # Create trial vector
                # DE/rand/1/bin
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Crossover
                jrand = rng.integers(n)
                trial = pop[i].copy()
                for j in range(n):
                    if rng.random() < CR or j == jrand:
                        trial[j] = v[j]
                
                # Clip to bounds
                trial = np.clip(trial, lower, upper)
                
                # Evaluate trial
                f_trial = func(trial)
                eval_count += 1
                
                # Selection
                if f_trial < fits[i]:
                    fits[i] = f_trial
                    pop[i] = trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
        
        # Phase 2: Nelder-Mead Local Search
        # Use remaining budget
        nm_budget = total_budget - eval_count
        if nm_budget > 0:
            # Initialize simplex around best_x
            # Simplex has n+1 vertices
            vertices = np.zeros((n + 1, n))
            vertices[0] = best_x
            for i in range(1, n + 1):
                vertices[i] = best_x.copy()
                # Perturb along axis i-1
                step = 0.5  # Initial step size
                if vertices[i][i-1] + step > upper:
                    vertices[i][i-1] = upper
                elif vertices[i][i-1] - step < lower:
                    vertices[i][i-1] = lower
                else:
                    vertices[i][i-1] += step
            
            f_vertices = np.zeros(n + 1)
            for i in range(n + 1):
                f_vertices[i] = func(vertices[i])
                eval_count += 1
                if f_vertices[i] < best_f:
                    best_f = f_vertices[i]
                    best_x = vertices[i].copy()
            
            alpha = 1.0  # Reflection
            gamma = 2.0  # Expansion
            rho = 0.5    # Contraction
            sigma = 0.5  # Shrink
            
            while eval_count < total_budget:
                # Sort vertices by function value
                idx = np.argsort(f_vertices)
                vertices = vertices[idx]
                f_vertices = f_vertices[idx]
                
                # Check convergence (optional, but we are budget limited)
                # If range is very small, break
                if f_vertices[-1] - f_vertices[0] < 1e-10:
                    break
                    
                # Calculate centroid of best n vertices
                centroid = np.mean(vertices[:-1], axis=0)
                
                # Reflection
                xr = centroid + alpha * (centroid - vertices[-1])
                xr = np.clip(xr, lower, upper)
                fr = func(xr)
                eval_count += 1
                if fr < best_f:
                    best_f = fr
                    best_x = xr.copy()
                
                if fr < f_vertices[0]:
                    # Expansion
                    xe = centroid + gamma * (xr - centroid)
                    xe = np.clip(xe, lower, upper)
                    fe = func(xe)
                    eval_count += 1
                    if fe < best_f:
                        best_f = fe
                        best_x = xe.copy()
                    if fe < fr:
                        vertices[-1] = xe
                        f_vertices[-1] = fe
                    else:
                        vertices[-1] = xr
                        f_vertices[-1] = fr
                elif fr < f_vertices[-2]:
                    # Accept reflection
                    vertices[-1] = xr
                    f_vertices[-1] = fr
                else:
                    # Contraction
                    if fr < f_vertices[-1]:
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - rho * (vertices[-1] - centroid)
                    
                    xc = np.clip(xc, lower, upper)
                    fc = func(xc)
                    eval_count += 1
                    if fc < best_f:
                        best_f = fc
                        best_x = xc.copy()
                    
                    if fc < min(fr, f_vertices[-1]):
                        vertices[-1] = xc
                        f_vertices[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            vertices[i] = vertices[0] + sigma * (vertices[i] - vertices[0])
                            vertices[i] = np.clip(vertices[i], lower, upper)
                            f_vertices[i] = func(vertices[i])
                            eval_count += 1
                            if f_vertices[i] < best_f:
                                best_f = f_vertices[i]
                                best_x = vertices[i].copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END
