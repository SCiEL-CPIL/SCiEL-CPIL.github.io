import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget
        
        # Smaller population allows more generations for adaptation
        pop_size = 10 
        
        # Initialize
        population = rng.uniform(-5, 5, (pop_size, dim))
        fitnesses = np.array([func(x) for x in population])
        calls_made = pop_size
        
        best_idx = np.argmin(fitnesses)
        global_best_f = fitnesses[best_idx]
        global_best_x = population[best_idx].copy()
        
        # Adaptive parameters
        F = 0.8
        CR = 0.9
        
        # Threshold for switching to exploitation
        switch_threshold = int(budget * 0.7)
        
        while calls_made < budget:
            # Determine mode
            if calls_made < switch_threshold:
                # Exploration: DE/rand/1/bin
                strategy = 'rand'
            else:
                # Exploitation: DE/current-to-pbest/1/bin (simplified: current-to-best)
                strategy = 'current'
            
            successes = 0
            new_population = population.copy()
            new_fitnesses = fitnesses.copy()
            
            for i in range(pop_size):
                if calls_made >= budget:
                    break
                
                if strategy == 'rand':
                    # DE/rand/1/bin
                    while True:
                        a = rng.integers(0, pop_size)
                        b = rng.integers(0, pop_size)
                        c = rng.integers(0, pop_size)
                        if a != i and b != i and c != i and a != b and a != c:
                            break
                    v = population[a] + F * (population[b] - population[c])
                else:
                    # DE/current-to-pbest/1/bin
                    # Use best individual (or pbest, but best is simpler)
                    # v = x_i + F*(x_best - x_i) + F*(x_r1 - x_r2)
                    while True:
                        r1 = rng.integers(0, pop_size)
                        r2 = rng.integers(0, pop_size)
                        if r1 != i and r2 != i and r1 != r2:
                            break
                    # Use a smaller F for exploitation
                    F_exploit = 0.5 
                    v = population[i] + F_exploit * (global_best_x - population[i]) + F_exploit * (population[r1] - population[r2])
                    # Note: We might want to adapt F_exploit too, but 0.5 is a good start
                    
                # Crossover
                j_rand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        new_population[i][j] = v[j]
                    else:
                        new_population[i][j] = population[i][j]
                
                new_population[i] = np.clip(new_population[i], -5, 5)
                
                # Evaluate
                f_new = func(new_population[i])
                calls_made += 1
                new_fitnesses[i] = f_new
                
                # Selection
                if f_new < fitnesses[i]:
                    population[i] = new_population[i]
                    fitnesses[i] = f_new
                    successes += 1
                    if f_new < global_best_f:
                        global_best_f = f_new
                        global_best_x = new_population[i].copy()
            
            # Adapt F and CR for next generation (if in exploration phase)
            if calls_made < switch_threshold:
                success_rate = successes / pop_size
                # If success rate is low, increase F (more exploration)
                # If success rate is high, decrease F (more exploitation)
                F = np.clip(F + 0.1 * (0.5 - success_rate), 0.4, 1.0)
                CR = np.clip(CR + 0.1 * (0.5 - success_rate), 0.5, 1.0)
                
        return global_best_x, global_best_f
