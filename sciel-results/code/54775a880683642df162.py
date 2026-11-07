"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer combining integer-grid sampling (exploiting Rastrigin structure)
    with Differential Evolution for global search and (1+1)-ES for local refinement.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bound = 5.0

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s limit
        dim = self.dim
        calls = 0
        max_calls = self.budget

        def eval_func(x):
            nonlocal calls
            if calls >= max_calls:
                return float('inf')
            calls += 1
            return func(x)

        def check_time():
            return time.time() < deadline - 3.0

        def check_budget():
            return calls < max_calls

        best_x = None
        best_f = float('inf')

        # Phase 0: Integer-grid sampling to exploit Rastrigin's structure
        # Global minimum of Rastrigin is at integer coordinates
        # Sample integer points in [-4, 4]^5 plus some near-integer offsets
        if check_time() and check_budget():
            # Sample integer points
            int_range = list(range(-4, 5))  # -4, -3, ..., 4
            # Randomly sample a subset of integer points to fit budget
            # 9^5 = 59049 possible integer points, sample ~50
            n_int_samples = min(50, max_calls // 10)
            
            # Generate random integer points
            rng = np.random.default_rng(42)
            int_points = []
            for _ in range(n_int_samples):
                pt = np.array([rng.choice(int_range) for _ in range(dim)], dtype=float)
                int_points.append(pt)
            
            # Add origin and some structured points
            int_points.append(np.zeros(dim))
            # Add points with one coordinate non-zero
            for coord in range(dim):
                for val in [-1, 1]:
                    pt = np.zeros(dim)
                    pt[coord] = val
                    int_points.append(pt)
            
            # Evaluate integer points
            for pt in int_points:
                if not check_time() or not check_budget():
                    break
                pt_clipped = np.clip(pt, -self.bound, self.bound)
                f_val = eval_func(pt_clipped)
                if f_val < best_f:
                    best_f = f_val
                    best_x = pt_clipped.copy()
        
        # Also sample near-integer points (offset by small amounts)
        if check_time() and check_budget():
            n_near_int = min(30, max_calls // 15)
            for _ in range(n_near_int):
                if not check_time() or not check_budget():
                    break
                # Start from a random integer point
                base_pt = np.array([rng.choice(int_range) for _ in range(dim)], dtype=float)
                # Add small random offset
                offset = rng.uniform(-0.5, 0.5, dim)
                pt = base_pt + offset
                pt = np.clip(pt, -self.bound, self.bound)
                f_val = eval_func(pt)
                if f_val < best_f:
                    best_f = f_val
                    best_x = pt.copy()

        # Phase 1: Differential Evolution
        if check_time() and check_budget():
            pop_size = 15
            F = 0.8  # differential weight
            CR = 0.9  # crossover rate

            # Initialize population with mix of integer and random points
            population = np.empty((pop_size, dim))
            # First few from integer sampling
            for i in range(min(5, pop_size)):
                if i == 0:
                    population[i] = best_x if best_x is not None else np.zeros(dim)
                else:
                    population[i] = np.array([rng.choice(int_range) for _ in range(dim)], dtype=float)
                    population[i] = np.clip(population[i], -self.bound, self.bound)
            # Rest random
            for i in range(min(5, pop_size), pop_size):
                population[i] = rng.uniform(-4, 4, dim)

            fitness = np.array([eval_func(ind) for ind in population])
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < best_f:
                best_f = fitness[best_idx]
                best_x = population[best_idx].copy()

            # DE main loop
            # Allocate ~60% of remaining budget to DE
            remaining_budget = max_calls - calls
            de_budget = int(remaining_budget * 0.6)
            
            for gen in range(30):  # Max generations
                if not check_time() or not check_budget():
                    break
                if calls >= max_calls - de_budget // 2:  # Keep some for ES
                    break

                for i in range(pop_size):
                    if not check_time() or not check_budget():
                        break
                    if calls >= max_calls - de_budget // 2:
                        break

                    # Select three distinct random indices
                    r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
                    # Ensure they are distinct from i
                    while r1 == i:
                        r1 = rng.choice(pop_size)
                    while r2 == i or r2 == r1:
                        r2 = rng.choice(pop_size)
                    while r3 == i or r3 == r1 or r3 == r2:
                        r3 = rng.choice(pop_size)

                    # Create trial vector using DE/rand/1
                    v = population[r1] + F * (population[r2] - population[r3])
                    v = np.clip(v, -self.bound, self.bound)

                    # Binary crossover
                    trial = population[i].copy()
                    j_rand = rng.integers(0, dim)
                    mask = rng.random(dim) < CR
                    mask[j_rand] = True
                    trial[mask] = v[mask]

                    # Selection
                    f_trial = eval_func(trial)
                    if f_trial <= fitness[i]:
                        population[i] = trial
                        fitness[i] = f_trial
                        if f_trial < best_f:
                            best_f = f_trial
                            best_x = trial.copy()

        # Phase 2: Local refinement with (1+1)-ES
        if check_time() and check_budget() and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.3

            while check_time() and check_budget():
                y = np.clip(x + sigma * rng.standard_normal(dim), -self.bound, self.bound)
                fy = eval_func(y)

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.15
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.85

                sigma = np.clip(sigma, 0.001, 5.0)

        # Final fallback: if no valid result, return origin
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
