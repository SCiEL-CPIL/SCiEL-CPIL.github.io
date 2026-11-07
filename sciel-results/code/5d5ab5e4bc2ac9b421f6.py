# EVOLVE-BLOCK-START
import numpy as np
import time
import math

class Optimizer:
    """Multi-start Differential Evolution optimizer for BBOB f23 (Katsuura)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        evals_used = 0

        def clip(x):
            return np.clip(x, -5.0, 5.0)

        def evaluate(x):
            nonlocal best_f, best_x, evals_used
            x = clip(x)
            f = func(x)
            evals_used += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def nelder_mead(x0, max_evals, step=0.3):
            """Simple Nelder-Mead for local refinement."""
            nonlocal best_f, best_x, evals_used
            n = dim
            simplex = np.zeros((n + 1, n))
            simplex[0] = x0.copy()
            for i in range(n):
                simplex[i + 1] = x0.copy()
                simplex[i + 1][i] += step

            fvals = np.zeros(n + 1)
            for i in range(n + 1):
                if time.time() > deadline or evals_used >= budget:
                    return
                fvals[i] = evaluate(simplex[i])

            alpha, gamma, rho, sigma_sh = 1.0, 2.0, 0.5, 0.5

            for _ in range(max_evals):
                if time.time() > deadline or evals_used >= budget:
                    return

                idx = np.argsort(fvals)
                simplex = simplex[idx]
                fvals = fvals[idx]

                centroid = np.mean(simplex[:-1], axis=0)
                x_r = centroid + alpha * (centroid - simplex[-1])
                f_r = evaluate(x_r)

                if f_r < fvals[0]:
                    x_e = centroid + gamma * (x_r - centroid)
                    f_e = evaluate(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    if f_r < fvals[-1]:
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        x_c = centroid - rho * (simplex[-1] - centroid)
                    f_c = evaluate(x_c)
                    if f_c < min(f_r, fvals[-1]):
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma_sh * (simplex[i] - simplex[0])
                            if time.time() > deadline or evals_used >= budget:
                                return
                            fvals[i] = evaluate(simplex[i])
                        break

        def differential_evolution(pop, pop_f, max_evals, F=0.8, CR=0.5):
            """Differential Evolution with adaptive parameters."""
            nonlocal best_f, best_x, evals_used
            n = dim
            N = len(pop)
            no_improve = 0
            best_local = best_f
            F_cur = F
            CR_cur = CR

            while evals_used < max_evals and time.time() < deadline:
                # Sort population by fitness
                order = np.argsort(pop_f)
                pop = [pop[i] for i in order]
                pop_f = [pop_f[i] for i in order]

                # Check for improvement
                if pop_f[0] < best_local:
                    best_local = pop_f[0]
                    no_improve = 0
                    # Success: keep parameters
                else:
                    no_improve += 1
                    # Adapt parameters: increase exploration when stagnating
                    if no_improve > 10:
                        F_cur = min(1.0, F_cur * 1.1)
                        CR_cur = min(1.0, CR_cur * 1.05)
                        if no_improve > 30:
                            # Reset some population members around best
                            for i in range(N // 2):
                                idx = rng.integers(1, N)
                                pop[idx] = pop[0] + rng.normal(0, 0.5, n)
                                pop[idx] = clip(pop[idx])
                                pop_f[idx] = evaluate(pop[idx])

                # Generate offspring
                for i in range(N):
                    if time.time() > deadline or evals_used >= budget:
                        return

                    # Select three distinct random individuals
                    r1, r2, r3 = rng.choice(N, 3, replace=False)
                    while r1 == i or r2 == i or r3 == i:
                        r1, r2, r3 = rng.choice(N, 3, replace=False)

                    # Differential mutation
                    v = pop[r1] + F_cur * (pop[r2] - pop[r3])
                    v = clip(v)

                    # Binomial crossover
                    j_rand = rng.integers(n)
                    trial = pop[i].copy()
                    for j in range(n):
                        if rng.random() < CR_cur or j == j_rand:
                            trial[j] = v[j]
                    trial = clip(trial)

                    # Evaluate trial
                    f_trial = evaluate(trial)

                    # Selection
                    if f_trial <= pop_f[i]:
                        pop[i] = trial
                        pop_f[i] = f_trial

                # Periodic restart: inject diversity if stagnating
                if no_improve > 20:
                    # Keep top half, reinitialize bottom half
                    half = N // 2
                    for i in range(half, N):
                        pop[i] = rng.uniform(-4, 4, n)
                        pop_f[i] = evaluate(pop[i])
                    no_improve = 0
                    F_cur = 0.8
                    CR_cur = 0.5

                # Stagnation check
                if no_improve > 50 and F_cur > 0.95:
                    break

        # === Phase 1: Diverse initialization ===
        # Center
        x_center = np.zeros(dim)
        evaluate(x_center)

        # Edge midpoints
        for i in range(dim):
            for sign in [-1, 1]:
                if time.time() > deadline or evals_used >= budget:
                    break
                xm = np.zeros(dim)
                xm[i] = sign * 3.0
                evaluate(xm)

        # Build initial population
        pop = []
        pop_f = []

        # Add center and edge points
        pop.append(x_center.copy())
        pop_f.append(best_f)
        for i in range(dim):
            for sign in [-1, 1]:
                xm = np.zeros(dim)
                xm[i] = sign * 3.0
                if xm not in [p.tolist() for p in pop]:
                    pop.append(xm)
                    pop_f.append(evaluate(xm))

        # Add random points to fill population
        target_pop_size = 20
        while len(pop) < target_pop_size:
            if time.time() > deadline or evals_used >= budget:
                break
            x = rng.uniform(-4, 4, dim)
            pop.append(x)
            pop_f.append(evaluate(x))

        if time.time() > deadline or evals_used >= budget:
            return best_x, best_f

        # === Phase 2: Main DE search ===
        remaining = budget - evals_used
        de_budget = int(remaining * 0.7)
        differential_evolution(pop, pop_f, de_budget, F=0.8, CR=0.5)

        if time.time() > deadline or evals_used >= budget:
            return best_x, best_f

        # === Phase 3: Nelder-Mead local refinement ===
        remaining = budget - evals_used
        nm_budget = min(80, remaining // 2)
        nelder_mead(best_x.copy(), nm_budget, step=0.3)

        if time.time() > deadline or evals_used >= budget:
            return best_x, best_f

        # === Phase 4: Second DE run with smaller step from best ===
        remaining = budget - evals_used
        if remaining > 50:
            # Build new population centered around best
            pop2 = [best_x.copy()]
            pop_f2 = [best_f]
            for i in range(19):
                if time.time() > deadline or evals_used >= budget:
                    break
                x = best_x + rng.normal(0, 0.5, dim)
                x = clip(x)
                pop2.append(x)
                pop_f2.append(evaluate(x))
            de_budget2 = remaining - 10
            differential_evolution(pop2, pop_f2, de_budget2, F=0.6, CR=0.7)

        if time.time() > deadline or evals_used >= budget:
            return best_x, best_f

        # === Phase 5: Final Nelder-Mead polish ===
        remaining = budget - evals_used
        if remaining > 20:
            nelder_mead(best_x.copy(), min(30, remaining), step=0.1)

        return best_x, best_f
# EVOLVE-BLOCK-END
