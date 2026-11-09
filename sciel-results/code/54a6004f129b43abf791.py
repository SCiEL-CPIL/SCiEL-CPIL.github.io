"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer: Sobol init, NSGA-II style with SBX crossover,
    adaptive mutation, and final local polish. Includes deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s kill
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Sobol quasi-random initialization
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=0)
        n_init = min(32, self.budget)
        sobol_samples = sampler.random(n_init)
        X = lo + span * sobol_samples
        F = np.array([problem(x) for x in X])
        used = n_init

        # Helper: check deadline
        def time_ok():
            return time.time() < deadline

        # Helper: fast non-dominated sort (returns ranks)
        def fast_non_dominated_sort(F):
            n = len(F)
            if n == 0:
                return np.array([])
            ranks = np.full(n, -1, dtype=int)
            S = [[] for _ in range(n)]
            n_dom = np.zeros(n, dtype=int)
            front = []
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dom[i] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dom[j] += 1
                if n_dom[i] == 0:
                    front.append(i)
                    ranks[i] = 0
            i = 0
            while front:
                next_front = []
                for p in front:
                    for q_idx in S[p]:
                        n_dom[q_idx] -= 1
                        if n_dom[q_idx] == 0:
                            next_front.append(q_idx)
                            ranks[q_idx] = i + 1
                front = next_front
                i += 1
            return ranks

        # Helper: crowding distance
        def crowding_distance(F, indices):
            if len(indices) <= 2:
                return np.full(len(indices), np.inf)
            m = len(indices)
            cd = np.zeros(m)
            F_sub = F[indices]
            for k in range(self.n_obj):
            # Sort by k-th objective
                order = np.argsort(F_sub[:, k])
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                f_range = F_sub[order[-1], k] - F_sub[order[0], k]
                if f_range == 0:
                    continue
                for i in range(1, m - 1):
                    cd[order[i]] += (F_sub[order[i + 1], k] - F_sub[order[i - 1], k]) / f_range
            return cd

        # Main loop: NSGA-II style with SBX crossover
        pop_size = n_init
        # We'll keep a population and evolve
        # Use remaining budget for generations
        # Each generation: select parents, create children, evaluate, select new pop
        # Let's use a simpler approach: keep population, each step create one child
        # via SBX from two parents, then replace worst

        # Actually, let's do proper NSGA-II with population management
        # But budget is 200, so let's be careful
        # Use population of size ~20, each generation produces ~20 children
        # That's ~20*10 = 200 evaluations

        # Let's do: initial 32, then 168 remaining
        # Population size 24, each gen 24 children = 24 evals
        # 168 / 24 = 7 generations

        pop_size = 24
        # If initial was 32, trim to pop_size
        if n_init > pop_size:
            # Keep non-dominated points, then fill with diverse
            ranks = fast_non_dominated_sort(F)
            # Get non-dominated
            nd_idx = np.where(ranks == 0)[0]
            if len(nd_idx) >= pop_size:
                # Use crowding distance to select
                cd = crowding_distance(F, nd_idx)
                keep = nd_idx[np.argsort(-cd)][:pop_size]
            else:
                # Take all non-dominated, then add most diverse dominated
                rest = np.setdiff1d(np.arange(n_init), nd_idx)
                # Sort rest by rank, then by sum
                rest_sorted = rest[np.lexsort((np.sum(F[rest], axis=1), ranks[rest]))]
                need = pop_size - len(nd_idx)
                keep = np.concatenate([nd_idx, rest_sorted[:need]])
            X = X[keep]
            F = F[keep]
            used = pop_size  # We conceptually re-use these, but they were already evaluated
            # Actually used stays at n_init, we just trimmed the pop
            # But we need to track that we have pop_size individuals
            # The used count is still n_init evaluations done

        # Now evolve
        while used < self.budget and time_ok():
            # Create children via SBX crossover + mutation
            children_X = []
            children_F = []
            n_children = min(pop_size, self.budget - used)
            for _ in range(n_children):
                # Select two parents using tournament selection
                p1_idx = self._tournament_select(F, pop_size, rng)
                p2_idx = self._tournament_select(F, pop_size, rng)
                # SBX crossover
                child = self._sbx_crossover(X[p1_idx], X[p2_idx], lo, hi, rng)
                # Adaptive mutation
                child = self._mutate(child, lo, hi, rng, used, self.budget)
                child = np.clip(child, lo, hi)
                fc = problem(child)
                children_X.append(child)
                children_F.append(fc)
                used += 1
                if used >= self.budget or not time_ok():
                    break

            if not children_X:
                break

            children_X = np.array(children_X)
            children_F = np.array(children_F)

            # Combine parents and children
            all_X = np.concatenate([X, children_X], axis=0)
            all_F = np.concatenate([F, children_F], axis=0)
            all_ranks = fast_non_dominated_sort(all_F)

            # Select next population of size pop_size
            next_X, next_F = self._select_population(all_X, all_F, all_ranks, pop_size)
            X, F = next_X, next_F

        # Final local polish: use remaining budget for local refinement
        if used < self.budget and time_ok():
            # Get non-dominated points
            ranks = fast_non_dominated_sort(F)
            nd_idx = np.where(ranks == 0)[0]
            if len(nd_idx) == 0:
                nd_idx = [int(np.argmin(np.sum(F, axis=1)))]

            # Local search around each non-dominated point
            for idx in nd_idx:
                if used >= self.budget or not time_ok():
                    break
                # Try small perturbations
                x_base = X[idx].copy()
                for _ in range(min(5, self.budget - used)):
                    if not time_ok():
                        break
                    # Random direction, small step
                    direction = rng.standard_normal(self.dim)
                    direction /= (np.linalg.norm(direction) + 1e-10)
                    step_size = 0.02 * np.linalg.norm(span)
                    x_new = x_base + step_size * direction
                    x_new = np.clip(x_new, lo, hi)
                    f_new = problem(x_new)
                    used += 1
                    # Accept if non-dominated or improves
                    if self._is_better(f_new, F, X, x_new, problem):
                        x_base = x_new
                    if used >= self.budget or not time_ok():
                        break

        return X

    def _tournament_select(self, F, pop_size, rng, k=3):
        """Tournament selection: pick k random, return index of best (non-dominated preferred)."""
        candidates = rng.integers(0, pop_size, size=k)
        # Score: prefer non-dominated, then lower sum
        # Simple: use rank-like proxy
        scores = []
        for idx in candidates:
            # Check if dominated by any other in population
            dominated = False
            for j in range(pop_size):
                if j != idx and np.all(F[j] <= F[idx]) and np.any(F[j] < F[idx]):
                    dominated = True
                    break
            score = (1 if dominated else 0, np.sum(F[idx]))
            scores.append(score)
        best = int(np.argmin(scores))
        return candidates[best]

    def _sbx_crossover(self, x1, x2, lo, hi, rng, eta=20.0):
        """Simulated Binary Crossover."""
        child = np.zeros_like(x1)
        for i in range(len(x1)):
            if rng.random() < 0.5:
                if abs(x1[i] - x2[i]) > 1e-14:
                    if rng.random() < 0.5:
                        beta = 1.0 + 2.0 * min(x1[i], x2[i]) / abs(x1[i] - x2[i])
                    else:
                        beta = 1.0 + 2.0 * (1.0 - max(x1[i], x2[i])) / abs(x1[i] - x2[i])
                    alpha = 2.0 - beta ** (-(eta + 1.0))
                    rand = rng.random()
                    if rand <= 1.0 / alpha:
                        beta_q = (rand * alpha) ** (1.0 / (eta + 1.0))
                    else:
                        beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
                    child[i] = 0.5 * ((x1[i] + x2[i]) - beta_q * (x1[i] - x2[i]))
                else:
                    child[i] = x1[i]
            else:
                if abs(x1[i] - x2[i]) > 1e-14:
                    if rng.random() < 0.5:
                        beta = 1.0 + 2.0 * min(x1[i], x2[i]) / abs(x1[i] - x2[i])
                    else:
                        beta = 1.0 + 2.0 * (1.0 - max(x1[i], x2[i])) / abs(x1[i] - x2[i])
                    alpha = 2.0 - beta ** (-(eta + 1.0))
                    rand = rng.random()
                    if rand <= 1.0 / alpha:
                        beta_q = (rand * alpha) ** (1.0 / (eta + 1.0))
                    else:
                        beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
                    child[i] = 0.5 * ((x1[i] + x2[i]) + beta_q * (x1[i] - x2[i]))
                else:
                    child[i] = x1[i]
        return child

    def _mutate(self, x, lo, hi, rng, used, budget):
        """Adaptive polynomial-like mutation with decreasing step size."""
        # Step size decreases as we use more budget
        progress = used / max(budget, 1)
        base_step = 0.3 * (1.0 - 0.7 * progress)
        for i in range(len(x)):
            if rng.random() < 0.5:  # mutate each dimension with 50% probability
                step = base_step * (hi[i] - lo[i]) * rng.standard_normal()
                x[i] += step
        return x

    def _select_population(self, all_X, all_F, all_ranks, pop_size):
        """Select best pop_size individuals using rank and crowding distance."""
        n = len(all_X)
        if n <= pop_size:
            return all_X, all_F

        # Group by rank
        next_X = []
        next_F = []
        selected = set()
        for rank in range(max(all_ranks) + 1):
            idx_in_rank = np.where(all_ranks == rank)[0]
            if len(next_X) + len(idx_in_rank) <= pop_size:
                for idx in idx_in_rank:
                    next_X.append(all_X[idx])
                    next_F.append(all_F[idx])
                    selected.add(idx)
            else:
                # Need crowding distance to select
                cd = crowding_distance(all_F, idx_in_rank)
                order = np.argsort(-cd)
                need = pop_size - len(next_X)
                for i in range(need):
                    idx = idx_in_rank[order[i]]
                    next_X.append(all_X[idx])
                    next_F.append(all_F[idx])
                    selected.add(idx)
                break

        return np.array(next_X), np.array(next_F)

    def _is_better(self, f_new, F, X, x_new, problem):
        """Check if new point is better: non-dominated by current pop or dominates some."""
        # Check if dominated by any current point
        for i in range(len(F)):
            if np.all(F[i] <= f_new) and np.any(F[i] < f_new):
                return False
        # Check if it dominates any current point
        for i in range(len(F)):
            if np.all(f_new <= F[i]) and np.any(f_new < F[i]):
                return True
        # Non-dominated and doesn't dominate: still accept to maintain diversity
        return True
# EVOLVE-BLOCK-END
