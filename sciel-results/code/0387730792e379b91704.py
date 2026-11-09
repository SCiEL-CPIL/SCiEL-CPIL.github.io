"""Combined Lattice-Aware DE + Nelder-Mead optimizer for BBOB f4 (Büche-Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Lattice-aware DE for global search + Nelder-Mead for local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit
        dim = self.dim
        budget = self.budget

        x_best = None
        f_best = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal x_best, f_best, calls_made
            if calls_made >= budget:
                return np.inf
            x = np.clip(np.asarray(x, dtype=np.float64), -5, 5)
            fx = float(func(x))
            calls_made += 1
            if fx < f_best:
                f_best = fx
                x_best = x.copy()
            return fx

        def local_refine(x_center, max_calls):
            """Local refinement around a point using Nelder-Mead."""
            nonlocal calls_made, x_best, f_best
            if x_center is None or max_calls < 5 or time.time() > deadline:
                return

            local_x = np.array(x_center, dtype=np.float64)
            local_f = f_best

            def wrapped(x):
                nonlocal calls_made
                if calls_made >= budget or time.time() > deadline:
                    return local_f
                x = np.clip(np.asarray(x, dtype=np.float64), -5, 5)
                fx = float(func(x))
                calls_made += 1
                if fx < local_f:
                    local_f = fx
                    x_best = x.copy()
                return fx

            try:
                res = minimize(wrapped, local_x, method='Nelder-Mead',
                               options={'maxiter': max_calls, 'maxfev': max_calls,
                                        'xatol': 1e-8, 'fatol': 1e-10})
                if res.fun < f_best:
                    f_best = float(res.fun)
                    x_best = np.array(res.x, dtype=np.float64)
            except Exception:
                pass

        def lattice_init(max_calls):
            """
            Initialize search by evaluating points near the integer lattice.
            BBOB f4 (Büche-Rastrigin) has local minima at shifted integer coordinates.
            We sample a grid of integers and their neighbors to find a good starting basin.
            """
            nonlocal calls_made, x_best, f_best
            if calls_made >= max_calls or time.time() > deadline:
                return

            rng = np.random.default_rng(42)
            
            # Generate candidate centers: integers from -4 to 4 (since shift is in [-4,4], min is in [-4,4] approx)
            # Actually, the minimizer is in [-4,4]^5. The function is periodic-ish around integers.
            # Let's check integers from -4 to 4.
            # Total points: 9^5 = 59049. Too many.
            # We will sample a subset.
            
            # Strategy: 
            # 1. Check pure integers in a smaller range first? No, shift is up to 4.
            # 2. Randomly sample from the set of integer points in [-4, 4]^5.
            
            # Let's create a list of candidate integer points
            # To save time, we don't want to iterate 59k points.
            # We will randomly pick integer points and evaluate them.
            # If we find a very low value, we stop early.
            
            # Heuristic: The global minimum is at z + shift, where z is an integer vector.
            # So we just need to find the correct z.
            
            # Let's evaluate random integer points.
            # We allocate 10% of budget for this phase.
            lattice_budget = min(max_calls, int(budget * 0.1))
            
            # Pre-generate a pool of random integer points to avoid overhead in loop
            # But 59049 is small enough to just sample from.
            
            # To be efficient, we can just pick random integers in [-4, 4]
            # and evaluate.
            
            for _ in range(lattice_budget):
                if calls_made >= budget or time.time() > deadline:
                    break
                # Pick random integer coordinates
                x_int = rng.integers(-4, 5, size=dim).astype(np.float64)
                # The function is shifted, so the minimum is at x_int + shift.
                # Evaluating x_int is not necessarily the minimum, but it's in the basin of the minimum at x_int+shift?
                # Actually, f(x) = f0(x - z*) + ...
                # If we evaluate x_int, and z* is close to x_int, we are in the basin.
                # However, the "shift" is hidden.
                # The inspiration suggests "search around integer lattice".
                # Let's evaluate x_int directly.
                
                fx = evaluate(x_int)
                
                # If we find a significantly better point, we might want to refine immediately?
                # Or just collect best.
                # For now, let's just update global best.
                
                # Optional: If this point is very good, do a tiny local refine to see if we can drop quickly?
                # No, let's stick to the phase structure.
                
            # After lattice sampling, if we found a good point, we can refine it slightly before DE?
            # This might help DE start in a better basin.
            if x_best is not None and calls_made < budget - 20:
                # Do a very short refine to ensure we are at the bottom of the basin found by lattice
                local_refine(x_best.copy(), min(10, budget - calls_made))

        def de_search(max_calls, pop_size=15, F_mut=0.8, CR=0.5):
            """Differential evolution for global exploration."""
            nonlocal calls_made, x_best, f_best
            rng = np.random.default_rng(int(time.time() * 1000) % (2**31))
            pop_size = min(pop_size, max(8, max_calls // 15))

            # Initialize population
            pop = []
            
            # Seed the population with the best lattice point if available
            if x_best is not None:
                pop.append((x_best.copy(), f_best))
                
            # Fill the rest with random points, biased towards integer lattice?
            # Or just random. DE is good at exploring.
            # Let's mix: 50% random, 50% integer neighbors of best?
            # Actually, standard random is fine if we have a good seed.
            
            while len(pop) < pop_size:
                if calls_made >= max_calls or time.time() > deadline:
                    break
                x = rng.uniform(-5, 5, dim)
                f = evaluate(x)
                pop.append((x, f))
            
            if not pop:
                return

            pop.sort(key=lambda item: item[1])

            while calls_made < max_calls and time.time() < deadline:
                x_best_pop = pop[0][0]
                new_pop = []

                for i in range(pop_size):
                    if calls_made >= max_calls or time.time() > deadline:
                        new_pop.append(pop[i])
                        continue

                    indices = [j for j in range(pop_size) if j != i]
                    if len(indices) < 3:
                        new_pop.append(pop[i])
                        continue
                    r1, r2, r3 = rng.choice(indices, 3, replace=False)

                    # Mutation: best/1/bin
                    v = x_best_pop + F_mut * (pop[r1][0] - pop[r2][0])

                    # Crossover
                    j_rand = rng.integers(0, dim)
                    trial = np.zeros(dim)
                    for j in range(dim):
                        if rng.random() < CR or j == j_rand:
                            trial[j] = v[j]
                        else:
                            trial[j] = pop[i][0][j]

                    trial = np.clip(trial, -5, 5)
                    f_trial = evaluate(trial)

                    if f_trial <= pop[i][1]:
                        new_pop.append((trial, f_trial))
                    else:
                        new_pop.append(pop[i])

                pop = new_pop
                pop.sort(key=lambda item: item[1])

            if pop and pop[0][1] < f_best:
                f_best = pop[0][1]
                x_best = pop[0][0].copy()

        # Strategy:
        # 1. Lattice Initialization: Quickly probe integer points to find the correct basin.
        # 2. DE: Global search to ensure we don't miss a better basin (though lattice should be good).
        # 3. Local Refinement: Nelder-Mead to converge to the exact minimum.
        # 4. Final Polish: Another short Nelder-Mead run.

        # Phase 1: Lattice Search (10% budget)
        lattice_calls = int(budget * 0.1)
        if time.time() < deadline:
            lattice_init(lattice_calls)

        # Phase 2: DE Global Search (60% budget)
        de_calls = int(budget * 0.6)
        if time.time() < deadline:
            de_search(de_calls, pop_size=15)

        # Phase 3: Local Refinement (20% budget)
        remaining = budget - calls_made
        if remaining >= 10 and x_best is not None and time.time() < deadline:
            refine_calls = min(remaining, int(budget * 0.2))
            local_refine(x_best.copy(), refine_calls)

        # Phase 4: Final Polish (10% budget)
        remaining = budget - calls_made
        if remaining >= 5 and x_best is not None and time.time() < deadline:
            polish_calls = min(remaining, int(budget * 0.1))
            local_refine(x_best.copy(), polish_calls)

        # Final safety: ensure we have a result
        if x_best is None:
            x_best = np.zeros(dim)
            f_best = evaluate(x_best)

        return x_best, f_best
# EVOLVE-BLOCK-END
