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

        # Generate a set of diverse starting points
        starts = []
        starts.append(np.zeros(dim))
        for _ in range(15):
            starts.append(rng.uniform(-5, 5, dim))
        for i in range(dim):
            for val in [-4, -2, 0, 2, 4]:
                pt = np.zeros(dim)
                pt[i] = val
                starts.append(pt)

        # Deduplicate approximately
        unique_starts = []
        for s in starts:
            if not any(np.allclose(s, u, atol=0.1) for u in unique_starts):
                unique_starts.append(s)
        starts = unique_starts

        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        # Sort starts: center first, then random, then structured
        rng.shuffle(starts)
        for i, s in enumerate(starts):
            if np.allclose(s, 0, atol=0.1):
                starts[0], starts[i] = starts[i], starts[0]
                break

        # Phase 1: Multi-start L-BFGS-B
        # Limit evaluations per run to ensure we visit many starts.
        for s in starts:
            if calls[0] >= budget:
                break
            if time.time() > deadline:
                break

            s = np.clip(s, -5, 5)

            try:
                res = minimize(
                    wrapped,
                    s,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 50, 'maxls': 20, 'gtol': 1e-4, 'ftol': 1e-10}
                )
                update_best(res.x, res.fun)
            except Exception:
                pass

            if calls[0] >= budget:
                break
            if time.time() > deadline:
                break

        # Phase 2: Differential Evolution (DE/rand/1)
        # Use a small population and limited generations to explore.
        if best_x is not None and calls[0] < budget:
            pop_size = 10
            n_gen = 5
            # Initialize population around the best point found so far
            # and some random points to maintain diversity
            pop = [best_x.copy()]
            for _ in range(pop_size - 1):
                # Mix of points near best and random
                if rng.random() < 0.5:
                    p = best_x + 0.5 * rng.standard_normal(dim)
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
                # If we couldn't evaluate the whole pop, resize it
                pop = pop[:len(pop_f)]
                pop_size = len(pop)
            
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
                    r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
                    
                    # Mutation: DE/rand/1
                    # F (mutation factor) adaptive? Let's keep it simple 0.5-0.8
                    F = 0.7 
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

        # Phase 3: Polish the best point with Nelder-Mead
        if best_x is not None and calls[0] < budget:
            try:
                remaining = budget - calls[0]
                nm_maxiter = min(100, max(10, remaining // 2))
                
                res_nm = minimize(
                    wrapped,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': nm_maxiter, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                update_best(res_nm.x, res_nm.fun)
            except Exception:
                pass

        # Phase 4: (1+1)-ES refinement from the best point
        if best_x is not None and calls[0] < budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1  # Start with small step since we are polishing
            no_improve = 0
            max_no_improve = 50
            
            while calls[0] < budget and time.time() < deadline:
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                try:
                    fy = wrapped(y)
                except Exception:
                    break
                
                if fy < fx:
                    x, fx = y, fy
                    update_best(x, fx)
                    sigma *= 1.1
                    no_improve = 0
                else:
                    sigma *= 0.9
                    no_improve += 1
                
                if no_improve > max_no_improve:
                    break
                if sigma < 1e-6:
                    break

        # Phase 5: Final tight polish if budget allows
        if best_x is not None and calls[0] < budget - 10:
            try:
                res_final = minimize(
                    wrapped,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 50, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
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
