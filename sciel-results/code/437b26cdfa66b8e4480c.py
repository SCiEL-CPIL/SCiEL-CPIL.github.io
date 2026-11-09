import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.perf_counter()
        deadline = 290  # Safety margin for 300s limit

        calls = 0
        best_x = None
        best_f = np.inf

        def safe_func(x):
            nonlocal calls, best_x, best_f
            if time.perf_counter() - t0 > deadline:
                raise TimeoutError("Deadline exceeded")
            if calls >= self.budget:
                raise StopIteration("Budget exceeded")

            x = np.asarray(x, dtype=np.float64)
            f = func(x)
            calls += 1

            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)
        n = self.dim

        # Phase 0: Multi-point initialization
        # Sample several points to get a good starting region for Schwefel
        n_init = 5
        candidates = []
        for i in range(n_init):
            if calls >= self.budget or time.perf_counter() - t0 > deadline:
                break
            if i == 0:
                x_c = rng.uniform(-0.5, 0.5, n)
            else:
                scale = 1.5 + 0.5 * i
                x_c = rng.uniform(-scale, scale, n)
            x_c = np.clip(x_c, -5.0, 5.0)
            try:
                f_c = safe_func(x_c)
                candidates.append((f_c, x_c))
            except (TimeoutError, StopIteration):
                break

        if not candidates:
            return best_x, best_f

        candidates.sort(key=lambda t: t[0])
        fx, x = candidates[0]
        best_local = fx
        best_local_x = x.copy()

        # Phase 1: (1+1)-ES with window-based success-rate sigma adaptation
        # Use ~65% of budget for global exploration
        global_budget = int(self.budget * 0.65)

        sigma = 2.0
        sigma_min = 1e-4
        sigma_max = 4.0
        window_size = 20
        success_window = []
        stall = 0
        STALL_LIMIT = 40
        restart_count = 0
        MAX_RESTARTS = 8

        while calls < global_budget and time.perf_counter() - t0 < deadline:
            y = np.clip(x + sigma * rng.standard_normal(n), -5.0, 5.0)
            try:
                fy = safe_func(y)
            except (TimeoutError, StopIteration):
                break

            if fy < fx:
                x, fx = y, fy
                success_window.append(1)
                stall = 0
                if fx < best_local:
                    best_local = fx
                    best_local_x = x.copy()
            else:
                success_window.append(0)
                stall += 1

            # Trim window
            if len(success_window) > window_size:
                success_window.pop(0)

            # Adapt sigma based on success rate in window
            if len(success_window) >= window_size // 2:
                sr = sum(success_window) / len(success_window)
                if sr > 0.3:
                    sigma *= 1.15
                elif sr < 0.15:
                    sigma *= 0.85

            sigma = max(sigma_min, min(sigma, sigma_max))

            if stall >= STALL_LIMIT:
                # Aggressive restart: jump significantly from best
                if calls < global_budget and time.perf_counter() - t0 < deadline and restart_count < MAX_RESTARTS:
                    jump_scale = max(sigma * 3.0, 1.5)
                    x_new = np.clip(x + jump_scale * rng.standard_normal(n), -5.0, 5.0)
                    try:
                        fx_new = safe_func(x_new)
                        if fx_new < fx:
                            x, fx = x_new, fx_new
                            if fx < best_local:
                                best_local = fx
                                best_local_x = x.copy()
                        else:
                            # If the jump didn't help, try a smaller one
                            if calls < global_budget and time.perf_counter() - t0 < deadline:
                                x_new2 = np.clip(x + sigma * 1.5 * rng.standard_normal(n), -5.0, 5.0)
                                fx_new2 = safe_func(x_new2)
                                if fx_new2 < fx:
                                    x, fx = x_new2, fx_new2
                                    if fx < best_local:
                                        best_local = fx
                                        best_local_x = x.copy()
                    except (TimeoutError, StopIteration):
                        break
                    sigma = max(sigma * 0.7, sigma_min)
                    stall = 0
                    restart_count += 1

        # Phase 2: Population-based RCGA with SBX and polynomial mutation
        remaining = self.budget - calls
        if remaining > 50 and best_x is not None and time.perf_counter() - t0 < deadline:
            pop_size = min(40, max(10, remaining // 15))
            
            # Initialize population with diversity
            population = [best_x.copy()]
            for i in range(pop_size - 1):
                if i < pop_size // 2:
                    # Small perturbations around best
                    pert = rng.standard_normal(n) * (0.1 + 0.05 * i)
                elif i < 3 * pop_size // 4:
                    # Medium perturbations
                    pert = rng.standard_normal(n) * (0.3 + 0.1 * i)
                else:
                    # Larger perturbations to explore other basins
                    pert = rng.uniform(-1.0, 1.0, n)
                new_point = best_x + pert
                new_point = np.clip(new_point, -5.0, 5.0)
                population.append(new_point)

            # Evaluate initial population
            pop_values = []
            for ind in population:
                try:
                    val = safe_func(ind)
                    pop_values.append(val)
                except (TimeoutError, StopIteration):
                    pop_values.append(np.inf)

            # RCGA loop
            eta_m = 20.0  # polynomial mutation distribution index
            eta_c = 20.0  # SBX distribution index
            p_c = 0.9     # crossover probability
            p_m = 1.0 / n  # mutation probability per gene

            while calls < self.budget and time.perf_counter() - t0 < deadline:
                # Selection: tournament selection
                new_population = []
                while len(new_population) < pop_size:
                    # Tournament of 3
                    idx1 = rng.integers(0, pop_size)
                    idx2 = rng.integers(0, pop_size)
                    idx3 = rng.integers(0, pop_size)
                    best_idx = min(idx1, idx2, idx3, key=lambda i: pop_values[i])
                    new_population.append(population[best_idx].copy())

                # Crossover and Mutation
                offspring = []
                for i in range(0, pop_size - 1, 2):
                    if calls >= self.budget or time.perf_counter() - t0 > deadline:
                        break

                    parent1 = new_population[i]
                    parent2 = new_population[i + 1]

                    # SBX Crossover
                    if rng.random() < p_c:
                        child1 = np.zeros(n)
                        child2 = np.zeros(n)
                        for j in range(n):
                            x1 = parent1[j]
                            x2 = parent2[j]
                            if rng.random() < 0.5:
                                if abs(x1 - x2) < 1e-14:
                                    child1[j] = x1
                                    child2[j] = x2
                                else:
                                    u = rng.random()
                                    if u <= 0.5:
                                        beta = (2 * u) ** (1.0 / (eta_c + 1))
                                    else:
                                        beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta_c + 1))
                                    child1[j] = 0.5 * ((1 + beta) * x1 + (1 - beta) * x2)
                                    child2[j] = 0.5 * ((1 - beta) * x1 + (1 + beta) * x2)
                            else:
                                child1[j] = x2
                                child2[j] = x1
                    else:
                        child1 = parent1.copy()
                        child2 = parent2.copy()

                    # Polynomial Mutation
                    for j in range(n):
                        if rng.random() < p_m:
                            u = rng.random()
                            if u < 0.5:
                                delta1 = (2 * u) ** (1.0 / (eta_m + 1)) - 1
                            else:
                                delta1 = 1 - (2 * (1 - u)) ** (1.0 / (eta_m + 1))
                            child1[j] += delta1 * (5.0 - (-5.0))
                        if rng.random() < p_m:
                            u = rng.random()
                            if u < 0.5:
                                delta1 = (2 * u) ** (1.0 / (eta_m + 1)) - 1
                            else:
                                delta1 = 1 - (2 * (1 - u)) ** (1.0 / (eta_m + 1))
                            child2[j] += delta1 * (5.0 - (-5.0))

                    child1 = np.clip(child1, -5.0, 5.0)
                    child2 = np.clip(child2, -5.0, 5.0)
                    offspring.append(child1)
                    offspring.append(child2)

                if not offspring:
                    break

                # Evaluate offspring
                offspring_values = []
                for off in offspring:
                    try:
                        val = safe_func(off)
                        offspring_values.append(val)
                    except (TimeoutError, StopIteration):
                        offspring_values.append(np.inf)
                        break

                # Elitism: keep the best individual
                all_inds = population + offspring
                all_vals = pop_values + offspring_values
                best_idx = np.argmin(all_vals)
                
                # Combine and select next generation
                combined = list(zip(all_inds, all_vals))
                combined.sort(key=lambda t: t[1])
                population = [ind for ind, val in combined[:pop_size]]
                pop_values = [val for ind, val in combined[:pop_size]]

        # Phase 3: Coordinate descent polish
        remaining = self.budget - calls
        if remaining > 0 and time.perf_counter() - t0 < deadline:
            step = max(sigma, 0.02)
            max_cd_rounds = min(5, remaining // (2 * n))

            for _round in range(max_cd_rounds):
                if time.perf_counter() - t0 > deadline or calls >= self.budget:
                    break
                moved_any = False
                for d in range(n):
                    if calls >= self.budget or time.perf_counter() - t0 > deadline:
                        break
                    best_trial = None
                    best_trial_f = fx
                    for sign in (1.0, -1.0):
                        if calls >= self.budget:
                            break
                        trial = x.copy()
                        trial[d] = np.clip(trial[d] + sign * step, -5.0, 5.0)
                        try:
                            ftrial = safe_func(trial)
                            if ftrial < best_trial_f:
                                best_trial_f = ftrial
                                best_trial = trial.copy()
                        except (TimeoutError, StopIteration):
                            break
                    if best_trial is not None:
                        x = best_trial
                        fx = best_trial_f
                        moved_any = True
                        if fx < best_local:
                            best_local = fx
                            best_local_x = x.copy()

                if not moved_any:
                    step *= 0.5
                    if step < 1e-5:
                        break

        return best_x, best_f
