"""Improved BBOB f22 optimizer: Differential Evolution + Nelder-Mead polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DE-based optimizer for BBOB f22 (Gallagher 21 peaks)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 280.0

        best_x = None
        best_f = np.inf
        calls_made = 0

        def check_time():
            return (time.time() - start_time) < time_limit

        def safe_func(x):
            nonlocal calls_made, best_x, best_f
            calls_made += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(42)
        dim = self.dim
        pop_size = 10
        
        # Initialize population
        population = []
        for i in range(pop_size):
            if not check_time() or calls_made >= self.budget:
                break
            x = rng.uniform(-5, 5, dim)
            fx = safe_func(x)
            population.append((fx, x))
        
        if not population:
            return best_x, best_f

        # Sort initially
        population.sort(key=lambda t: t[0])
        
        # DE Parameters
        F = 0.5  # Mutation factor
        CR = 0.5 # Crossover rate
        # Adaptive parameters: we will adjust F and CR based on success rate
        # Simple adaptation: if success rate is high, increase F; if low, decrease F?
        # Standard DE/rand/1/bin
        
        # We will run generations until budget is exhausted
        # Each generation evaluates pop_size individuals (trials)
        # We can also evaluate the current best occasionally? No, DE is population based.
        
        # To make it more effective, we can use a "1+1" DE variant where we only
        # generate one trial per generation based on the best, but that's just (1+1)-ES.
        # Let's stick to standard DE but with a smaller population to allow more generations.
        # Pop size 10 is good.
        
        # We need to be careful with the budget.
        # Init: 10 calls.
        # Remaining: 490.
        # Generations: 490 / 10 = 49.
        
        # Let's add a restart mechanism if the population converges too early.
        # If the spread of the population is small, re-initialize some individuals.
        
        while check_time() and (self.budget - calls_made) >= pop_size:
            # Sort population
            population.sort(key=lambda t: t[0])
            
            # Check convergence: if best and worst are close, restart
            if population[-1][0] - population[0][0] < 1e-6:
                # Re-initialize half the population
                for i in range(pop_size // 2):
                    if not check_time() or calls_made >= self.budget:
                        break
                    x = rng.uniform(-5, 5, dim)
                    fx = safe_func(x)
                    population[i] = (fx, x)
                population.sort(key=lambda t: t[0])
                continue

            new_population = []
            # For each individual, create a trial
            for i in range(pop_size):
                if not check_time() or calls_made >= self.budget:
                    break
                
                # Select base individual (random)
                r1 = rng.integers(0, pop_size)
                r2 = rng.integers(0, pop_size)
                while r1 == i or r2 == i or r1 == r2:
                    r1 = rng.integers(0, pop_size)
                    r2 = rng.integers(0, pop_size)
                
                base = population[r1][1]
                diff = population[r2][1] - population[r1][1] # Wait, standard is v = x_r1 + F * (x_r2 - x_r3)
                # Let's use DE/rand/1/bin: v = x_r1 + F * (x_r2 - x_r3)
                # We need 3 distinct indices
                indices = rng.choice(pop_size, size=3, replace=False)
                while i in indices:
                    indices = rng.choice(pop_size, size=3, replace=False)
                
                v = population[indices[0]][1] + F * (population[indices[1]][1] - population[indices[2]][1])
                v = np.clip(v, -5, 5)
                
                # Crossover
                trial = population[i][1].copy()
                j_rand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                
                ft = safe_func(trial)
                
                # Selection
                if ft <= population[i][0]:
                    new_population.append((ft, trial))
                else:
                    new_population.append(population[i])
            
            population = new_population
            
            # Adapt F and CR
            # If many improvements, increase F to explore more? Or decrease to exploit?
            # Usually, if success rate is high, we can increase F to find better directions.
            # If success rate is low, decrease F.
            # Let's track success rate
            # This is a heuristic, let's keep it simple for now.
            # F = np.clip(F * 1.05, 0.1, 0.9) # Slow increase
            # CR = np.clip(CR * 0.95, 0.1, 0.9) # Slow decrease?
            
            # Let's try a simple adaptation:
            # Count successes in this generation
            # We didn't track it explicitly in the loop above, let's add it.
            # Actually, let's just use fixed F=0.8, CR=0.5 which is robust for 5D.
            # Or adapt F based on the spread.
            spread = np.std([p[0] for p in population])
            if spread < 1e-3:
                F = 0.9
                CR = 0.9
            else:
                F = 0.5
                CR = 0.5

        # Polish with Nelder-Mead
        if best_x is not None and check_time() and (self.budget - calls_made) > 10:
            # Implement a simple Nelder-Mead
            def nelder_mead(x0, f0, max_iter=50):
                nonlocal best_x, best_f, calls_made
                n = dim
                simplex = [x0.copy()]
                fvals = [f0]
                for i in range(n):
                    xi = x0.copy()
                    xi[i] += 0.1
                    xi = np.clip(xi, -5, 5)
                    fi = safe_func(xi)
                    simplex.append(xi)
                    fvals.append(fi)
                
                alpha = 1.0
                gamma = 2.0
                rho = 0.5
                sigma = 0.5
                
                for _ in range(max_iter):
                    if not check_time() or (self.budget - calls_made) <= 0:
                        break
                    idx = np.argsort(fvals)
                    simplex = [simplex[i] for i in idx]
                    fvals = [fvals[i] for i in idx]
                    
                    if fvals[-1] - fvals[0] < 1e-10:
                        break
                    
                    centroid = np.mean(simplex[:-1], axis=0)
                    xr = centroid + alpha * (centroid - simplex[-1])
                    xr = np.clip(xr, -5, 5)
                    fr = safe_func(xr)
                    
                    if fr < fvals[0]:
                        xe = centroid + gamma * (xr - centroid)
                        xe = np.clip(xe, -5, 5)
                        fe = safe_func(xe)
                        if fe < fr:
                            simplex[-1] = xe
                            fvals[-1] = fe
                        else:
                            simplex[-1] = xr
                            fvals[-1] = fr
                    elif fr < fvals[-2]:
                        simplex[-1] = xr
                        fvals[-1] = fr
                    else:
                        xc = centroid + rho * (simplex[-1] - centroid)
                        xc = np.clip(xc, -5, 5)
                        fc = safe_func(xc)
                        if fc < fvals[-1]:
                            simplex[-1] = xc
                            fvals[-1] = fc
                        else:
                            for i in range(1, n + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                fvals[i] = safe_func(simplex[i])
                                
            nelder_mead(best_x.copy(), best_f, max_iter=50)

        return best_x, best_f
# EVOLVE-BLOCK-END
