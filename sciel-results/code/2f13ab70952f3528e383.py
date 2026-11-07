import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320  # Safety margin

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Phase 1: Global search with adaptive DE
        pop_size = 10
        max_iter = (budget - pop_size) // pop_size
        
        # Initialize population with stratified sampling
        pop = np.zeros((pop_size, dim))
        for d in range(dim):
            edges = np.linspace(-5, 5, pop_size + 1)
            for i in range(pop_size):
                pop[i, d] = rng.uniform(edges[i], edges[i+1])
        
        fpop = np.array([func(x) for x in pop])
        best_idx = np.argmin(fpop)
        best_x = pop[best_idx].copy()
        best_f = fpop[best_idx]
        
        calls_made = pop_size
        
        # Adaptive DE parameters
        F_start = 0.8
        F_end = 0.3
        CR = 0.9
        
        no_improve = 0
        patience = 15
        
        for gen in range(max_iter):
            if time.time() > deadline:
                break
            
            # Adaptive F: decrease over generations
            F = F_start + (F_end - F_start) * (gen / max_iter)
            
            for i in range(pop_size):
                if time.time() > deadline:
                    break
                
                # Select 3 distinct random indices different from i
                while True:
                    a, b, c = rng.choice(pop_size, 3, replace=False)
                    if a != i and b != i and c != i:
                        break
                
                v = pop[a] + F * (pop[b] - pop[c])
                
                # Crossover
                u = v.copy()
                j_rand = rng.integers(dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        u[j] = v[j]
                    else:
                        u[j] = pop[i][j]
                
                u = np.clip(u, -5, 5)
                fu = func(u)
                calls_made += 1
                
                if fu < fpop[i]:
                    pop[i] = u
                    fpop[i] = fu
                    if fu < best_f:
                        best_f = fu
                        best_x = u.copy()
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
            
            # Restart if stuck: perturb around best
            if no_improve > patience:
                no_improve = 0
                for i in range(pop_size):
                    if i == np.argmin(fpop):
                        continue
                    pop[i] = best_x + rng.normal(0, 0.5, dim)
                    pop[i] = np.clip(pop[i], -5, 5)
                    fpop[i] = func(pop[i])
                    calls_made += 1
                    if fpop[i] < best_f:
                        best_f = fpop[i]
                        best_x = pop[i].copy()
                
                if calls_made >= budget:
                    break

        # Phase 2: Local refinement with simple coordinate descent
        remaining = budget - calls_made
        if remaining > 5:
            step = 0.5
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                improved = False
                for d in range(dim):
                    for direction in [-1, 1]:
                        trial = best_x.copy()
                        trial[d] += direction * step
                        trial = np.clip(trial, -5, 5)
                        ft = func(trial)
                        calls_made += 1
                        if ft < best_f:
                            best_f = ft
                            best_x = trial.copy()
                            improved = True
                if not improved:
                    step *= 0.5
                    if step < 1e-6:
                        break

        return best_x, best_f
