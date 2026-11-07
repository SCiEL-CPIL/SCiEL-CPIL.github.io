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
        # Use stratified sampling for better coverage: 1 center, 8 axis-aligned, 11 random
        n_init = min(20, budget // 10)  # Use ~10% of budget for initialization
        for i in range(n_init):
            if calls[0] >= budget or time.time() > deadline:
                break
            if i == 0:
                x = np.zeros(dim)
            elif i <= dim:
                # Axis-aligned points at +-3
                axis = i - 1
                sign = 1 if i <= dim // 2 else -1
                x = np.zeros(dim)
                x[axis] = sign * 3.0
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
        # Add perturbed and random starts to escape local minima
        n_starts = 8
        for i in range(n_starts):
            if rng.random() < 0.6:
                # Perturb best with varying scale
                scale = 0.5 + rng.random() * 2.0
                s = best_x + rng.standard_normal(dim) * scale
            else:
                # Random
                s = rng.uniform(-5, 5, dim)
            s = np.clip(s, -5, 5)
            starts.append(s)

        for s in starts:
            if calls[0] >= budget or time.time() > deadline:
                break
            
            s = np.clip(s, -5, 5)
            
            try:
                # L-BFGS-B: increase maxiter for harder functions
                res = minimize(
                    wrapped,
                    s,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 150, 'maxls': 30, 'gtol': 1e-6, 'ftol': 1e-12}
                )
                update_best(res.x, res.fun)
            except Exception:
                pass

        # Phase 3: Differential Evolution (DE/rand/1)
        # This phase is crucial for Rastrigin and Gallagher.
        # It explores the space more broadly.
        if best_x is not None and calls[0] < budget:
            pop_size = 20  # Larger population for better exploration
            remaining_calls = budget - calls[0]
            # Each generation costs ~pop_size * 2 evaluations
            n_gen = max(1, min(15, remaining_calls // (pop_size * 2)))
            
            # Initialize population: mix of best, perturbations, and random
            pop = [best_x.copy()]
            for _ in range(pop_size - 1):
                r = rng.random()
                if r < 0.25:
                    # Close to best
                    p = best_x + rng.standard_normal(dim) * 1.0
                elif r < 0.5:
                    # Medium distance
                    p = best_x + rng.standard_normal(dim) * 3.0
                else:
                    # Random
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

            # DE Loop with adaptive F
            for gen in range(n_gen):
                if calls[0] >= budget or time.time() > deadline:
                    break
                
                new_pop = []
                new_pop_f = []
                
                # Adaptive F: start high, decrease as we converge
                F = max(0.5, 0.9 - 0.03 * gen)
                
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
                    v = pop[r1] + F * (pop[r2] - pop[r3])
                    v = np.clip(v, -5, 5)
                    
                    # Crossover with CR=0.7 for more exploration
                    CR = 0.7
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

        # Phase 4: CMA-ES-like adaptive search
        # This helps with rotated/shifted functions by adapting the search distribution
        if best_x is not None and calls[0] < budget:
            x = best_x.copy()
            fx = best_f
            sigma = 1.0  # Initial step size
            C = np.eye(dim)  # Covariance matrix
            C_inv_sqrt = np.linalg.inv(np.linalg.cholesky(C))
            alpha_mu = 1.0 / (dim * dim)
            c_sigma = (dim + 2) / (dim + 12)
            c_c = (4 + dim / dim) / (dim + 4)
            mu = dim // 2 + 1  # Number of best individuals used for update
            w = np.zeros(mu)
            w_sum = 0.0
            for i in range(mu):
                w[i] = np.log(dim / 2 + 0.5) - np.log(i + 1)
            w_sum = np.sum(w)
            w = w / w_sum
            mu_eff = 1.0 / np.sum(w ** 2)
            c_c = (4 + dim / dim) / (dim + 4)
            c_sigma = (dim + 2) / (dim + 12)
            c_c = (4 + dim / dim) / (dim + 4)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            chi_N = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            
            max_iter_cma = min(50, (budget - calls[0]) // (mu + 1))
            
            for gen in range(max_iter_cma):
                if calls[0] >= budget or time.time() > deadline:
                    break
                
                # Generate offspring
                offspring = []
                offspring_f = []
                
                for k in range(mu):
                    if calls[0] >= budget or time.time() > deadline:
                        break
                    y = C_inv_sqrt @ rng.standard_normal(dim)
                    z = sigma * y
                    x_new = x + z
                    x_new = np.clip(x_new, -5, 5)
                    try:
                        f_new = wrapped(x_new)
                    except Exception:
                        break
                    offspring.append(x_new)
                    offspring_f.append(f_new)
                
                if len(offspring) < mu:
                    break
                
                # Sort by fitness
                order = np.argsort(offspring_f)
                y_w = np.array([offspring[i] for i in order[:mu]])
                y_w = (y_w - x) / sigma
                
                # Update mean
                x = x + sigma * np.sum(w[:, None] * y_w, axis=0)
                x = np.clip(x, -5, 5)
                
                # Update best
                for i in range(mu):
                    if offspring_f[order[i]] < fx:
                        fx = offspring_f[order[i]]
                        x = offspring[order[i]].copy()
                        update_best(x, fx)
                
                # Update covariance matrix (simplified)
                y_w_centered = y_w - np.sum(w[:, None] * y_w, axis=0)
                C = (1 - c_c * (2 - c_c)) * C + c_c * (2 - c_c) * (
                    (1 / (1 - np.sqrt(mu_eff / 2))) * np.outer(y_w_centered.T @ w, y_w_centered.T @ w)
                    + (1 - c_c) * np.eye(dim)
                )
                C_inv_sqrt = np.linalg.inv(np.linalg.cholesky(C + 1e-10 * np.eye(dim)))
                
                # Update sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / chi_N * y_w.T @ w
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_N - 1))
                sigma = np.clip(sigma, 1e-6, 10.0)
                
                if sigma < 1e-5:
                    break

        # Phase 5: (1+1)-ES refinement from the best point
        # Very fine-tuning with better sigma adaptation.
        if best_x is not None and calls[0] < budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1  # Start with moderate step
            no_improve = 0
            max_no_improve = 20
            success_count = 0
            total_trials = 0
            
            while calls[0] < budget and time.time() < deadline:
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                try:
                    fy = wrapped(y)
                except Exception:
                    break
                
                total_trials += 1
                if fy < fx:
                    x, fx = y, fy
                    update_best(x, fx)
                    sigma *= 1.1
                    no_improve = 0
                    success_count += 1
                else:
                    sigma *= 0.95
                    no_improve += 1
                
                # Adapt sigma based on success rate over last 10 trials
                if total_trials % 10 == 0:
                    success_rate = success_count / min(total_trials, 10)
                    if success_rate > 0.5:
                        sigma *= 1.2
                    elif success_rate < 0.3:
                        sigma *= 0.8
                    success_count = 0
                
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
                    options={'maxiter': 100, 'maxls': 20, 'gtol': 1e-10, 'ftol': 1e-14}
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
