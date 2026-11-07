"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Multi-start optimizer combining L-BFGS-B, Differential Evolution, and Nelder-Mead.
    Strategy:
    1. Generate diverse starting points.
    2. Run a short L-BFGS-B from each start to find local minima.
    3. Run a brief Differential Evolution (DE/rand/1) phase to explore the landscape
       and potentially escape local minima, updating the global best.
    4. Polish the global best with Nelder-Mead.
    5. Use remaining budget for (1+1)-ES refinement with adaptive step size.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin
        dim = self.dim
        budget = self.budget
        calls = [0]

        def wrapped(x):
            calls[0] += 1
            if calls[0] > budget:
                raise RuntimeError("budget exceeded")
            return float(func(x))

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        # Phase 1: Rapid diverse sampling to get a good initial best
        # We need a good starting point for the local optimizers.
        # Sample 20 points: 1 center, 19 random.
        for _ in range(20):
            if calls[0] >= budget or time.time() > deadline:
                break
            if _ == 0:
                x = np.zeros(dim)
            else:
                x = rng.uniform(-5, 5, dim)
            try:
                f_val = wrapped(x)
                update_best(x, f_val)
            except Exception:
                pass

        # Phase 2: Multi-start L-BFGS-B
        # Use the best point and some random points around it as starts.
        # L-BFGS-B is very efficient for smooth functions like Sphere, Rosenbrock.
        # For Rastrigin, it might get stuck in local minima, but DE will help later.
        
        starts = [best_x.copy()]
        # Add some random starts to escape local minima
        for _ in range(10):
            if rng.random() < 0.5:
                # Perturb best
                s = best_x + rng.standard_normal(dim) * 1.0
            else:
                # Random
                s = rng.uniform(-5, 5, dim)
            s = np.clip(s, -5, 5)
            starts.append(s)

        # Shuffle starts but keep best first? No, best is likely to be found quickly.
        # Actually, let's just iterate.
        for s in starts:
            if calls[0] >= budget or time.time() > deadline:
                break
            
            s = np.clip(s, -5, 5)
            
            try:
                # L-BFGS-B is fast. Let it run until convergence or maxiter.
                # maxiter 100 is usually enough for 5D.
                res = minimize(
                    wrapped,
                    s,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 100, 'maxls': 20, 'gtol': 1e-5, 'ftol': 1e-10}
                )
                update_best(res.x, res.fun)
            except Exception:
                pass

        # Phase 3: Differential Evolution (DE/rand/1)
        # This phase is crucial for Rastrigin and Gallagher.
        # It explores the space more broadly.
        if best_x is not None and calls[0] < budget:
            pop_size = 15
            # Use remaining budget to determine generations
            # Each generation costs pop_size * 2 evaluations (mutant + crossover candidate)
            # Roughly.
            remaining_calls = budget - calls[0]
            n_gen = max(1, min(10, remaining_calls // (pop_size * 2)))
            
            # Initialize population
            pop = [best_x.copy()]
            for _ in range(pop_size - 1):
                if rng.random() < 0.3:
                    p = best_x + rng.standard_normal(dim) * 2.0
                else:
                    p = rng.uniform(-5, 5, dim)
                pop.append(np.clip(p, -5, 5))
            
            # Evaluate initial population
            pop_f = []
            for p in pop:
                if calls[0] >= budget or time.time() > deadline:
                    break
                try:
                    f_val = wrapped(p)
                    pop_f.append(f_val)
                    update_best(p, f_val)
                except Exception:
                    break
            
            if len(pop_f) < pop_size:
                pop = pop[:len(pop_f)]
                pop_size = len(pop)
                if pop_size < 3:
                    return best_x, best_f # Not enough for DE

            # DE Loop
            for gen in range(n_gen):
                if calls[0] >= budget or time.time() > deadline:
                    break
                
                new_pop = []
                new_pop_f = []
                
                for i in range(pop_size):
                    if calls[0] >= budget or time.time() > deadline:
                        break
                    
                    # Select three distinct random individuals
                    indices = [j for j in range(pop_size) if j != i]
                    if len(indices) < 2:
                        break
                    r1, r2 = rng.choice(indices, 2, replace=False)
                    r3 = rng.choice([j for j in range(pop_size) if j != i and j != r1 and j != r2], 1, replace=False)[0]
                    
                    # Mutation: DE/rand/1
                    F = 0.8 # Higher F for more exploration
                    v = pop[r1] + F * (pop[r2] - pop[r3])
                    v = np.clip(v, -5, 5)
                    
                    # Evaluate mutant
                    try:
                        f_v = wrapped(v)
                    except Exception:
                        continue
                    
                    # Crossover
                    CR = 0.5
                    u = np.zeros(dim)
                    j_rand = rng.integers(dim)
                    for j in range(dim):
                        if rng.random() < CR or j == j_rand:
                            u[j] = v[j]
                        else:
                            u[j] = pop[i][j]
                    
                    # Evaluate candidate
                    try:
                        f_u = wrapped(u)
                    except Exception:
                        continue
                    
                    # Selection
                    if f_u <= pop_f[i]:
                        new_pop.append(u)
                        new_pop_f.append(f_u)
                        update_best(u, f_u)
                    else:
                        new_pop.append(pop[i])
                        new_pop_f.append(pop_f[i])
                
                # If we broke out early due to budget, break the generation loop
                if len(new_pop) < pop_size:
                    break
                    
                pop = new_pop
                pop_f = new_pop_f

        # Phase 4: Polish the best point with Nelder-Mead
        # NM is robust for final polishing.
        if best_x is not None and calls[0] < budget:
            try:
                remaining = budget - calls[0]
                nm_maxiter = min(200, max(10, remaining // 2))
                
                res_nm = minimize(
                    wrapped,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': nm_maxiter, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
                )
                update_best(res_nm.x, res_nm.fun)
            except Exception:
                pass

        # Phase 5: (1+1)-ES refinement from the best point
        # Very fine-tuning.
        if best_x is not None and calls[0] < budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.05  # Start with small step
            no_improve = 0
            max_no_improve = 30
            
            while calls[0] < budget and time.time() < deadline:
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                try:
                    fy = wrapped(y)
                except Exception:
                    break
                
                if fy < fx:
                    x, fx = y, fy
                    update_best(x, fx)
                    sigma *= 1.05
                    no_improve = 0
                else:
                    sigma *= 0.9
                    no_improve += 1
                
                if no_improve > max_no_improve:
                    break
                if sigma < 1e-7:
                    break

        # Phase 6: Final tight polish if budget allows
        # Use L-BFGS-B again with tight tolerances on the final best.
        if best_x is not None and calls[0] < budget - 5:
            try:
                res_final = minimize(
                    wrapped,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 50, 'maxls': 10, 'gtol': 1e-8, 'ftol': 1e-12}
                )
                update_best(res_final.x, res_final.fun)
            except Exception:
                pass

        if best_x is None:
            # Fallback
            best_x = rng.uniform(-5, 5, dim)
            best_f = wrapped(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
