"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Strategy: Adaptive Differential Evolution (JADE-like).
    
    The Bent Cigar function (BBOB f12) is highly ill-conditioned. 
    Local methods like Nelder-Mead or Powell often struggle or get stuck 
    in flat regions or fail to converge efficiently along the narrow axis.
    
    Differential Evolution (DE) is a population-based stochastic optimizer 
    that is robust to ill-conditioning because it uses differences between 
    random pairs of individuals to generate trial vectors. This allows it to 
    explore the landscape broadly and adaptively converge without requiring 
    gradients or assuming convexity.
    
    Key features:
    - Population-based search avoids getting trapped in local minima as easily.
    - Adaptive step size (F) and crossover rate (CR) based on successful 
      candidates (JADE algorithm logic).
    - Efficient use of the budget: 500 evaluations in 5D is sufficient for 
      DE to converge to high precision if parameters are tuned correctly.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.start_time = None
        self.deadline = None
        self.best_x = None
        self.best_f = float('inf')

    def __call__(self, func):
        self.start_time = time.time()
        # Safety margin: 300s limit, stop at 290s
        self.deadline = self.start_time + 290.0
        self.calls = 0
        self.best_x = None
        self.best_f = float('inf')

        def wrapped_func(x):
            if time.time() > self.deadline:
                raise TimeoutError("Time limit exceeded")
            self.calls += 1
            if self.calls > self.budget:
                raise ValueError("Budget exceeded")
            # Clip to box [-5, 5]
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x_clipped.copy()
            return fx

        rng = np.random.default_rng(42)
        
        # DE Parameters
        # Population size: Standard recommendation is 5-10 * dim.
        # dim=5, so N=25 is a good balance.
        # Budget=500. 500 / 25 = 20 generations.
        # This is a bit low for DE, but with good initialization and adaptive params,
        # it should make significant progress.
        # Alternatively, smaller N allows more generations.
        # Let's try N=10 (2*dim) to allow 50 generations.
        N = 10
        dim = self.dim
        
        # Initialize population
        # Use a mix of random points and origin to ensure coverage
        pop = np.zeros((N, dim))
        # First few points: structured starts
        pop[0] = np.zeros(dim)
        for i in range(1, min(4, N)):
            pop[i] = rng.uniform(-4, 4, dim)
        # Rest random
        pop[4:] = rng.uniform(-5, 5, dim)
        
        # Evaluate initial population
        for i in range(N):
            if self.calls >= self.budget or time.time() > self.deadline:
                break
            pop_f = wrapped_func(pop[i])
            # Store fitness in a separate array for sorting
            # We need to keep track of which x corresponds to which f
            # Since wrapped_func updates self.best, we just need local tracking for DE selection
            # Let's use a list of (f, x)
            pass
            
        # We need to manage the population state explicitly
        # Let's re-initialize with a proper loop that captures values
        pop = np.zeros((N, dim))
        pop_f = np.zeros(N)
        
        pop[0] = np.zeros(dim)
        for i in range(1, min(4, N)):
            pop[i] = rng.uniform(-4, 4, dim)
        pop[4:] = rng.uniform(-5, 5, dim)

        for i in range(N):
            if self.calls >= self.budget or time.time() > self.deadline:
                break
            pop_f[i] = wrapped_func(pop[i])

        # If we didn't fill the population, adjust N
        valid_N = self.calls
        if valid_N < N:
            N = valid_N
            pop = pop[:N]
            pop_f = pop_f[:N]

        if N == 0:
             # Fallback
             self.best_x = np.zeros(dim)
             self.best_f = func(self.best_x)
             self.calls = 1
             return self.best_x, self.best_f

        # Sort initial population
        idx = np.argsort(pop_f)
        pop = pop[idx]
        pop_f = pop_f[idx]
        
        # DE Parameters (JADE)
        F_mean = 0.5
        CR_mean = 0.5
        F_sigma = 0.1
        CR_sigma = 0.1
        
        # Main DE Loop
        # Each generation takes N evaluations (for trial vectors)
        # We have remaining budget: budget - N
        # Generations available: (budget - N) // N
        max_gen = (self.budget - N) // N
        
        for gen in range(max_gen):
            if time.time() > self.deadline:
                break
            if self.calls >= self.budget:
                break

            # Adaptive parameters for this generation
            # JADE uses Cauchy distribution for F and CR, but simple normal is often fine
            F = rng.normal(F_mean, F_sigma)
            F = np.clip(F, 0.1, 0.9)
            CR = rng.normal(CR_mean, CR_sigma)
            CR = np.clip(CR, 0.1, 0.9)

            # Generate trial vectors
            trial_pop = np.zeros((N, dim))
            trial_f = np.zeros(N)
            
            # Track successful F and CR for adaptation
            F_succ = []
            CR_succ = []
            
            for i in range(N):
                if self.calls >= self.budget or time.time() > self.deadline:
                    break
                
                # Selection of base vector: Current-to-Best/2 or Best/1
                # Let's use "Best/1/bin" or "Rand/1/bin"
                # "Best/1/bin" is fast but can converge prematurely.
                # "Rand/1/bin" is more robust.
                # Given the short budget, "Best/1/bin" might be better for exploitation.
                # Let's try a mix: 50% Best/1, 50% Rand/1
                
                if rng.random() < 0.5:
                    # Best/1
                    base = 0 # index of best is 0
                    # Pick two distinct random indices a, b != i, != base
                    candidates = [j for j in range(N) if j != i and j != base]
                    a, b = rng.choice(candidates, size=2, replace=False)
                else:
                    # Rand/1
                    # Pick three distinct random indices a, b, c != i
                    candidates = [j for j in range(N) if j != i]
                    a, b, c = rng.choice(candidates, size=3, replace=False)
                    base = a

                # Mutation: x_trial = x_base + F * (x_a - x_b)
                trial = pop[base] + F * (pop[a] - pop[b])
                
                # Crossover (Binomial)
                # JADE uses binomial crossover
                r = rng.random()
                j_rand = rng.integers(0, dim)
                
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = trial[j]
                    else:
                        trial[j] = pop[i][j]
                
                # Clip trial to bounds
                trial = np.clip(trial, -5, 5)
                
                # Evaluate trial
                trial_f[i] = wrapped_func(trial)
                trial_pop[i] = trial
                
                # Selection
                if trial_f[i] <= pop_f[i]:
                    pop[i] = trial
                    pop_f[i] = trial_f[i]
                    F_succ.append(F)
                    CR_succ.append(CR)

            # Update population fitness order for next iteration
            # We only evaluated N new points. We need to re-sort the whole population
            # Note: pop and pop_f are updated in-place for the winners
            idx = np.argsort(pop_f)
            pop = pop[idx]
            pop_f = pop_f[idx]
            
            # Adapt parameters if we have successes
            if len(F_succ) > 0:
                F_mean = 0.9 * F_mean + 0.1 * np.mean(F_succ)
                CR_mean = 0.9 * CR_mean + 0.1 * np.mean(CR_succ)
            
            # Convergence check? If best hasn't improved in last K generations, 
            # maybe perturb? With only 20-50 gens, simple DE is usually fine.

        return self.best_x, self.best_f

# EVOLVE-BLOCK-END
