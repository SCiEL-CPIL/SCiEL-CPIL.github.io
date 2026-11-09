<<<<<<< SEARCH
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 320.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Use np.random to respect the seed set by the evaluator
        # pop_size = 10 allows 20 generations with budget 200
        pop_size = 10
        
        # Initialize
        X = lo + (hi - lo) * np.random.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        eta_c = 20.0
        pc = 0.9
        sigma_scale = 0.05 # Reduced from 0.1
        
        while used < self.budget:
            if time.time() > deadline:
                break
            
            Q = np.empty_like(X)
            for i in range(pop_size):
                # Tournament Selection
                p1 = self._tournament_select(X, F)
                p2 = self._tournament_select(X, F)
                
                # Crossover (SBX)
                if np.random.random() < pc:
                    c1, c2 = self._sbx_crossover(p1, p2, lo, hi, eta_c)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                
                # Gaussian Mutation
                c1 = c1 + sigma_scale * (hi - lo) * np.random.standard_normal(self.dim)
                c2 = c2 + sigma_scale * (hi - lo) * np.random.standard_normal(self.dim)
                
                c1 = np.clip(c1, lo, hi)
                c2 = np.clip(c2, lo, hi)
                
                Q[i] = c1
                
            F_q = np.array([problem(x) for x in Q])
            used += pop_size
            
            P = np.vstack([X, Q])
            F_p = np.vstack([F, F_q])
            
            X, F = self._nsga2_select(P, F_p, pop_size)
            
        return X
=======
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 320.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Use np.random to respect the seed set by the evaluator
        # Larger population for better coverage
        pop_size = 25
        
        # Initialize
        X = lo + (hi - lo) * np.random.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        eta_c = 10.0
        pc = 0.9
        pm = 0.5  # mutation probability
        eta_m = 10.0  # mutation distribution index
        
        while used < self.budget:
            if time.time() > deadline:
                break
            
            Q = np.empty_like(X)
            for i in range(pop_size):
                # Tournament Selection
                p1 = self._tournament_select(X, F)
                p2 = self._tournament_select(X, F)
                
                # Crossover (SBX)
                if np.random.random() < pc:
                    c1, c2 = self._sbx_crossover(p1, p2, lo, hi, eta_c)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                
                # Polynomial Mutation
                c1 = self._poly_mutate(c1, lo, hi, eta_m)
                c2 = self._poly_mutate(c2, lo, hi, eta_m)
                
                Q[i] = c1
                
            F_q = np.array([problem(x) for x in Q])
            used += pop_size
            
            P = np.vstack([X, Q])
            F_p = np.vstack([F, F_q])
            
            X, F = self._nsga2_select(P, F_p, pop_size)
            
        return X
>>>>>>> REPLACE

