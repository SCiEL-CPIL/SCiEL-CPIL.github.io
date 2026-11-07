import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285

        call_count = 0
        best_x = None
        best_f = float('inf')

        def do_call(x):
            nonlocal call_count, best_f, best_x
            if call_count >= budget:
                raise ValueError("Budget exceeded")
            call_count += 1
            x_c = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f = float(func(x_c))
            if f < best_f:
                best_f = f
                best_x = x_c.copy()
            return f

        rng = np.random.default_rng(42)

        # --- Phase 0: Initial scatter ---
        n_init = 20
        for i in range(n_init):
            if call_count >= budget or time.time() > deadline:
                break
            x = rng.uniform(-4.5, 4.5, dim)
            try:
                do_call(x)
            except Exception:
                break

        if best_x is None:
            return (None, float('inf'))

        # --- Phase 1: Differential Evolution ---
        # DE is population-based and good for multimodal functions like Rastrigin
        pop_size = 30
        F = 0.8  # Differential weight
        CR = 0.9  # Crossover rate

        # Initialize population around best point and random
        pop = [best_x.copy()]
        for i in range(1, pop_size):
            if i < 5:
                # Some points near best
                x = best_x + rng.normal(0, 1.0, dim)
            else:
                x = rng.uniform(-4.8, 4.8, dim)
            x = np.clip(x, -4.9, 4.9)
            try:
                do_call(x)
            except Exception:
                break
            pop.append(x)

        # Evaluate initial population (we already have f values from do_call)
        # We need to track f values for the population
        pop_f = []
        for x in pop:
            # We need to find the f value; since do_call updates best_f, we need a different approach
            # Let's re-evaluate or track separately
            pass

        # Actually, let's track population f values properly
        pop = []
        pop_f = []

        # Re-initialize population with tracked f values
        # We already used some calls, so let's be careful
        # Start fresh with tracked population
        pop = []
        pop_f = []

        # Add best point
        pop.append(best_x.copy())
        pop_f.append(best_f)

        # Add initial scatter points
        for i in range(pop_size - 1):
            if call_count >= budget or time.time() > deadline:
                break
            if i < 5:
                x = best_x + rng.normal(0, 1.0, dim)
            else:
                x = rng.uniform(-4.8, 4.8, dim)
            x = np.clip(x, -4.9, 4.9)
            try:
                f = do_call(x)
            except Exception:
                break
            pop.append(x)
            pop_f.append(f)

        pop = np.array(pop)
        pop_f = np.array(pop_f)

        # DE main loop
        max_de_iters = 50
        for it in range(max_de_iters):
            if call_count >= budget - 10 or time.time() > deadline:
                break

            n_pop = len(pop)
            if n_pop < 4:
                break

            # Shuffle population indices for DE
            idx = rng.permutation(n_pop)

            new_pop = []
            new_pop_f = []

            for i in range(n_pop):
                if call_count >= budget - 5 or time.time() > deadline:
                    break

                # Select 3 distinct individuals different from i
                a, b, c = idx[i], idx[(i+1) % n_pop], idx[(i+2) % n_pop]
                while a == b or a == c or b == c:
                    a, b, c = rng.integers(0, n_pop), rng.integers(0, n_pop), rng.integers(0, n_pop)

                # Mutation: v = x[a] + F * (x[b] - x[c])
                v = pop[a] + F * (pop[b] - pop[c])
                v = np.clip(v, -4.9, 4.9)

                # Crossover: trial vector
                trial = pop[i].copy()
                j_rand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                trial = np.clip(trial, -4.9, 4.9)

                try:
                    f_trial = do_call(trial)
                except Exception:
                    break

                # Selection
                if f_trial <= pop_f[i]:
                    new_pop.append(trial)
                    new_pop_f.append(f_trial)
                else:
                    new_pop.append(pop[i])
                    new_pop_f.append(pop_f[i])

            if new_pop:
                pop = np.array(new_pop)
                pop_f = np.array(new_pop_f)

                # Update best
                best_idx = np.argmin(pop_f)
                if pop_f[best_idx] < best_f:
                    best_f = pop_f[best_idx]
                    best_x = pop[best_idx].copy()

                # Adaptive F and CR
                F = max(0.5, F * 0.98)
                CR = min(0.95, CR * 1.01)

        # --- Phase 2: Local refinement with Nelder-Mead ---
        if best_x is not None and call_count < budget - 10 and time.time() < deadline:
            from scipy.optimize import minimize

            n_nm = 3
            for i in range(n_nm):
                if call_count >= budget - 5 or time.time() > deadline:
                    break

                if i == 0:
                    x0 = best_x.copy()
                else:
                    x0 = best_x + rng.normal(0, 0.3 * (i + 1), dim)
                    x0 = np.clip(x0, -4.9, 4.9)

                nm_calls = [0]
                def nm_func(x):
                    nm_calls[0] += 1
                    if nm_calls[0] > 30 or call_count >= budget - 2 or time.time() > deadline:
                        return 1e10
                    return do_call(x)

                try:
                    result = minimize(
                        nm_func, x0, method='Nelder-Mead',
                        options={'maxiter': 25, 'xatol': 1e-4, 'fatol': 1e-6}
                    )
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = np.clip(result.x, -4.9, 4.9)
                except Exception:
                    pass

        # --- Phase 3: Final coordinate descent polish ---
        if best_x is not None and call_count < budget - 5 and time.time() < deadline:
            x = best_x.copy()
            f = best_f
            step = 0.1
            for _ in range(15):
                if call_count >= budget or time.time() > deadline:
                    break
                improved = False
                for i in range(dim):
                    if call_count >= budget or time.time() > deadline:
                        break
                    for d in [1, -1]:
                        x_try = x.copy()
                        x_try[i] += d * step
                        try:
                            f_try = do_call(x_try)
                        except Exception:
                            break
                        if f_try < f:
                            x[i] = x_try[i]
                            f = f_try
                            improved = True
                if not improved:
                    step *= 0.5
                    if step < 1e-6:
                        break
            best_x = x
            best_f = f

        return best_x, best_f
