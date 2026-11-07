"""Combined DE + Nelder-Mead optimizer for BBOB f4 (Büche-Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid Differential Evolution + Nelder-Mead optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _nelder_mead(self, func, x0, max_iter, calls_ref, deadline):
        """Run Nelder-Mead from x0, updating calls_ref[0] with count used."""
        dim = self.dim
        calls = 0

        # Initial simplex
        simplex = []
        f0 = func(x0)
        calls += 1
        simplex.append((x0.copy(), f0))

        # Create simplex points
        base = x0.copy()
        for i in range(dim):
            point = base.copy()
            # Use a small initial step size for local refinement
            step = 0.1 if point[i] == 0 else 0.1 * np.sign(point[i])
            point[i] += step
            point = np.clip(point, -5, 5)
            f = func(point)
            calls += 1
            simplex.append((point, f))

        # Sort by function value
        simplex.sort(key=lambda p: p[1])

        best_x = simplex[0][0].copy()
        best_f = simplex[0][1]

        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma = 0.5   # shrink

        for _ in range(max_iter):
            if calls_ref[0] + calls >= self.budget:
                break
            if time.time() > deadline - 2.0:
                break

            # Sort simplex
            simplex.sort(key=lambda p: p[1])

            # Check convergence
            fvals = np.array([p[1] for p in simplex])
            if np.max(fvals) - np.min(fvals) < 1e-12:
                break

            x_worst = simplex[-1][0]
            x_best = simplex[0][0]

            # Centroid of all but worst
            centroid = np.mean([p[0] for p in simplex[:-1]], axis=0)

            # Reflection
            x_r = centroid + alpha * (centroid - x_worst)
            x_r = np.clip(x_r, -5, 5)
            f_r = func(x_r)
            calls += 1

            if f_r < simplex[0][1]:
                # Expansion
                x_e = centroid + gamma * (centroid - x_worst)
                x_e = np.clip(x_e, -5, 5)
                f_e = func(x_e)
                calls += 1
                if f_e < f_r:
                    simplex[-1] = (x_e, f_e)
                else:
                    simplex[-1] = (x_r, f_r)
            elif f_r < simplex[-2][1]:
                simplex[-1] = (x_r, f_r)
            else:
                # Contraction
                if f_r < simplex[-1][1]:
                    # Outside contraction
                    x_c = centroid + rho * (x_r - centroid)
                else:
                    # Inside contraction
                    x_c = centroid - rho * (x_r - centroid)
                x_c = np.clip(x_c, -5, 5)
                f_c = func(x_c)
                calls += 1

                if f_c <= f_r:
                    simplex[-1] = (x_c, f_c)
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i][0] = x_best + sigma * (simplex[i][0] - x_best)
                        simplex[i][0] = np.clip(simplex[i][0], -5, 5)
                        simplex[i][1] = func(simplex[i][0])
                        calls += 1
                    if calls_ref[0] + calls >= self.budget:
                        break

            # Update best
            simplex.sort(key=lambda p: p[1])
            if simplex[0][1] < best_f:
                best_f = simplex[0][1]
                best_x = simplex[0][0].copy()

        calls_ref[0] += calls
        return best_x, best_f

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        calls_ref = [0]

        # --- Phase 1: Differential Evolution for Global Exploration ---
        # DE parameters
        pop_size = 20
        F = 0.8  # Differential weight
        CR = 0.9  # Crossover rate

        # Allocate 60% of budget to DE
        de_budget = int(budget * 0.6)
        
        # Initialize population
        population = [rng.uniform(-5, 5, size=dim) for _ in range(pop_size)]
        fitness = [np.inf] * pop_size
        
        # Evaluate initial population
        for i, ind in enumerate(population):
            if calls_ref[0] >= de_budget:
                break
            fitness[i] = func(ind)
            calls_ref[0] += 1

        best_x = population[np.argmin(fitness)].copy()
        best_f = min(fitness)

        # DE Loop
        while calls_ref[0] < de_budget:
            if time.time() > deadline - 5.0:
                break
                
            for i in range(pop_size):
                if calls_ref[0] >= de_budget:
                    break
                if time.time() > deadline - 5.0:
                    break

                # Select 3 distinct random individuals
                r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
                
                # Mutation: v = x[r1] + F * (x[r2] - x[r3])
                v = population[r1] + F * (population[r2] - population[r3])
                v = np.clip(v, -5, 5)
                
                # Crossover: binomial
                trial = population[i].copy()
                j_rand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                
                trial = np.clip(trial, -5, 5)
                
                # Evaluation
                trial_f = func(trial)
                calls_ref[0] += 1
                
                # Selection
                if trial_f <= fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_f
                    if trial_f < best_f:
                        best_f = trial_f
                        best_x = trial.copy()

        # --- Phase 2: Local Refinement with Nelder-Mead ---
        # Use remaining budget for local search
        remaining_budget = budget - calls_ref[0]
        
        # Refine the best candidate found
        if remaining_budget > 10:
            # Allocate most of remaining budget to the best candidate
            nm_budget = int(remaining_budget * 0.8)
            max_iter = max(10, nm_budget // 2) # Rough estimate of calls per iteration
            
            bx, bf = self._nelder_mead(func, best_x, max_iter, calls_ref, deadline)
            
            if bf < best_f:
                best_f = bf
                best_x = bx.copy()

            # If there is still budget, try refining the second best candidate from DE
            if calls_ref[0] < budget:
                # Find second best from DE population
                sorted_indices = np.argsort(fitness)
                second_best_idx = sorted_indices[1]
                second_best_x = population[second_best_idx].copy()
                
                # Small budget for second refinement
                remaining2 = budget - calls_ref[0]
                if remaining2 > 10:
                    max_iter2 = max(5, remaining2 // 3)
                    bx2, bf2 = self._nelder_mead(func, second_best_x, max_iter2, calls_ref, deadline)
                    if bf2 < best_f:
                        best_f = bf2
                        best_x = bx2.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
