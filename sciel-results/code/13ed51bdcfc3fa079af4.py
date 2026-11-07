"""
Advanced DE (JADE variant) with LHS initialization and L-BFGS-B polish
for BBOB-style 5-D box minimization.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """
    Optimizer using JADE (adaptive DE) with Latin Hypercube Sampling initialization
    and a final L-BFGS-B polish.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        DEADLINE = t0 + 280.0  # Safety margin

        def alive():
            return time.time() < DEADLINE

        def clip(v):
            return np.clip(v, -5.0, 5.0)

        # --- Phase 0: Latin Hypercube Sampling (LHS) Initialization ---
        # Larger population for better gradient estimation in DE
        pop_size = min(50, budget // 5)  # At least 5 generations if possible, but 50 is good for d=5
        if pop_size < 5:
            pop_size = 5
            
        rng = np.random.default_rng(42)
        
        # Generate LHS points
        # For each dimension, split [0,1] into pop_size bins, pick one uniform point in each
        lhs_points = np.zeros((pop_size, dim))
        for d in range(dim):
            bins = np.linspace(0, 1, pop_size + 1)
            points_in_bin = [rng.uniform(bins[i], bins[i+1]) for i in range(pop_size)]
            rng.shuffle(points_in_bin)
            lhs_points[:, d] = points_in_bin
            
        # Map [0,1] to [-5,5]
        X = lhs_points * 10.0 - 5.0
        
        F = np.empty(pop_size)
        for i in range(pop_size):
            F[i] = func(X[i])
            
        best_idx = np.argmin(F)
        bestx = X[best_idx].copy()
        bestf = F[best_idx]
        
        calls = pop_size
        if not alive() or budget <= calls:
            return bestx, bestf

        # --- Phase 1: JADE DE Evolution ---
        # JADE parameters
        F_mean = 0.5
        CR_mean = 0.1
        K = 0.1  # Learning rate for F and CR updates
        
        # Track history for adaptation
        F_hist = []
        CR_hist = []
        
        # Stagnation counter
        stagnation_count = 0
        max_stagnation = 10 # Generations without improvement before restart
        
        while calls < budget and alive():
            # Generate trial vectors
            # Sample F and CR from Cauchy distributions
            F_trial = np.clip(rng.cauchy(F_mean, 0.1), 0.1, 0.9)
            CR_trial = np.clip(rng.cauchy(CR_mean, 0.1), 0.0, 1.0)
            
            # In JADE, we usually sample F and CR for each individual or use a single one?
            # Standard JADE samples F and CR for each trial vector.
            
            improved = False
            new_X = X.copy()
            new_F = F.copy()
            
            for i in range(pop_size):
                if not alive() or calls >= budget:
                    break
                    
                # Select p, q, r distinct from i
                # Use random selection without replacement
                candidates = list(range(pop_size))
                candidates.remove(i)
                p, q, r = rng.choice(candidates, size=3, replace=False)
                
                # Mutation: DE/rand/1
                # v = X[i] + F * (X[p] - X[q]) + (1-F) * (X[i] - X[r]) ? 
                # JADE uses: v = X[i] + F * (X[p] - X[q])
                # Actually JADE uses: v = X[i] + F * (X[p] - X[q]) where p,q,r are distinct
                # Wait, standard JADE mutation: x_r1 - x_r2 + F * (x_r3 - x_i) ?? No.
                # JADE: v_i = x_i + F * (x_r1 - x_r2)
                
                v = X[i] + F_trial * (X[p] - X[q])
                v = clip(v)
                
                fv = func(v)
                calls += 1
                
                # Binomial Crossover
                j_rand = rng.integers(0, dim)
                trial = np.zeros(dim)
                for j in range(dim):
                    if rng.uniform() <= CR_trial or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = X[i][j]
                        
                # Selection
                if fv <= F[i]:
                    new_X[i] = trial
                    new_F[i] = fv
                    if fv < bestf:
                        bestf = fv
                        bestx = trial.copy()
                        improved = True
                        stagnation_count = 0
                    else:
                        stagnation_count += 1 # Individual didn't improve best, but might improve its own slot
                else:
                    stagnation_count += 1 # No improvement in this trial

            X, F = new_X, new_F
            
            # Update F_mean and CR_mean based on successful trials
            # In JADE, we update based on the F and CR values that produced accepted trials.
            # For simplicity and robustness, we just keep the global adaptive means if we track them per trial.
            # Here, we used a single F_trial and CR_trial for the whole generation.
            # This is a simplified "JADE-like" approach.
            
            # Check for stagnation of the global best
            if not improved:
                stagnation_count += 1
            else:
                stagnation_count = 0

            # Restart if stagnant
            if stagnation_count > max_stagnation:
                # Restart population around the best point
                # Keep best point, resample others with smaller spread
                spread = 1.0
                X[0] = bestx
                F[0] = bestf
                for i in range(1, pop_size):
                    if not alive() or calls >= budget:
                        break
                    X[i] = clip(bestx + rng.normal(0, spread, dim))
                    F[i] = func(X[i])
                    calls += 1
                stagnation_count = 0
                # Reduce spread for next restarts
                spread *= 0.5

        # --- Phase 2: Polish with L-BFGS-B ---
        if alive() and calls < budget:
            try:
                # L-BFGS-B handles bounds well and is efficient for smooth functions
                res = minimize(
                    func, 
                    bestx, 
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': max(10, (budget - calls) // 5), 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res.x is not None:
                    rx = clip(np.asarray(res.x, dtype=float))
                    rf = float(func(rx))
                    calls += 1
                    if rf < bestf:
                        bestx, bestf = rx, rf
            except Exception:
                pass

        return bestx, bestf
# EVOLVE-BLOCK-END
