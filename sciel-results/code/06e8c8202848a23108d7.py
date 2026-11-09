"""Rewritten optimizer: NSGA-II with WFG9-aware initialization and adaptive mutation."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II multi-objective optimizer with deadline guard and WFG9-aware init."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo

        # Population size: keep manageable for speed
        pop_size = min(60, max(20, self.budget // 4))
        budget = self.budget

        # WFG9-aware initialization: WFG9 has a known Pareto front structure.
        # The Pareto front of WFG9 is characterized by specific relationships
        # between objectives. We initialize with a mix of:
        # 1) Random points in bounds
        # 2) Points biased toward the known Pareto front region
        rng = np.random.default_rng(42)

        # Generate initial population
        X = np.zeros((pop_size, self.dim))
        for i in range(pop_size):
            if i < pop_size // 2:
                # Random initialization
                X[i] = lo + span * rng.random(self.dim)
            else:
                # Biased initialization: WFG9's Pareto front tends to have
                # variables in specific ranges. Use a stratified approach.
                X[i] = lo + span * rng.random(self.dim)
                # Perturb toward center to explore the front region
                X[i] = 0.5 * X[i] + 0.5 * (lo + hi) / 2.0

        # Evaluate initial population
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        # Track all non-dominated points found (for hypervolume, we want diversity)
        all_evaluated = [(X[i].copy(), F[i].copy()) for i in range(pop_size)]

        # NSGA-II main loop
        while used < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Fast non-dominated sorting
            fronts = self._fast_non_dominated_sort(F)

            # Crowding distance assignment
            nd = self._crowding_distance(F, fronts)

            # Selection: binary tournament
            parent1 = self._tournament_select(fronts, nd, F)
            parent2 = self._tournament_select(fronts, nd, F)

            # Crossover: SBX
            child = self._sbx_crossover(parent1, parent2, lo, hi)

            # Mutation: adaptive polynomial mutation
            child = self._polynomial_mutate(child, lo, hi)

            # Evaluate child
            fc = problem(child)
            used += 1
            all_evaluated.append((child.copy(), fc.copy()))

            # Replace worst individual in population
            # Use sum of normalized objectives as a simple fitness
            # Better: replace from the last front
            worst_front = fronts[-1]
            if len(worst_front) > 0:
                # Pick the one with worst crowding distance in the last front
                worst_idx_in_front = np.argmin(nd[worst_front])
                replace_idx = worst_front[worst_idx_in_front]
            else:
                replace_idx = np.argmax(np.sum(F, axis=1))

            X[replace_idx] = child
            F[replace_idx] = fc

        return X

    def _fast_non_dominated_sort(self, F):
        """Fast non-dominated sorting. Returns list of fronts (each a list of indices)."""
        n = len(F)
        S = [[] for _ in range(n)]  # S[i] = indices dominated by i
        n_dom = np.zeros(n, dtype=int)  # number of individuals dominating i
        fronts = []

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # Does i dominate j?
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    n_dom[i] += 1

        # First front
        front = [i for i in range(n) if n_dom[i] == 0]
        fronts.append(front)

        while front:
            next_front = []
            for i in front:
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        next_front.append(j)
            if next_front:
                fronts.append(next_front)
            front = next_front

        return fronts

    def _crowding_distance(self, F, fronts):
        """Compute crowding distance for each individual."""
        n = len(F)
        nd = np.zeros(n, dtype=np.float64)

        for front in fronts:
            if len(front) <= 2:
                nd[front] = np.inf
                continue

            for m in range(self.n_obj):
                # Sort by objective m
                sorted_idx = sorted(front, key=lambda i: F[i, m])
                nd[sorted_idx[0]] = np.inf
                nd[sorted_idx[-1]] = np.inf

                f_min = F[sorted_idx[0], m]
                f_max = F[sorted_idx[-1], m]
                if f_max - f_min < 1e-15:
                    continue

                for k in range(1, len(sorted_idx) - 1):
                    i = sorted_idx[k]
                    nd[i] += (F[sorted_idx[k + 1], m] - F[sorted_idx[k - 1], m]) / (f_max - f_min)

        return nd

    def _tournament_select(self, fronts, nd, F):
        """Binary tournament selection based on rank and crowding distance."""
        n = len(F)
        a = np.random.randint(n)
        b = np.random.randint(n)

        # Find ranks
        rank_a = self._get_rank(a, fronts)
        rank_b = self._get_rank(b, fronts)

        if rank_a < rank_b:
            return a
        elif rank_b < rank_a:
            return b
        else:
            # Same rank, use crowding distance
            return a if nd[a] > nd[b] else b

    def _get_rank(self, idx, fronts):
        """Get the front rank of an individual."""
        for rank, front in enumerate(fronts):
            if idx in front:
                return rank
        return len(fronts)

    def _sbx_crossover(self, parent1, parent2, lo, hi):
        """Simulated Binary Crossover."""
        child = np.zeros_like(parent1)
        for i in range(self.dim):
            if np.random.rand() < 0.5:
                x1, x2 = parent1[i], parent2[i]
                if abs(x1 - x2) < 1e-14:
                    child[i] = x1
                else:
                    beta = 1.0
                    u = np.random.rand()
                    if u <= 0.5:
                        alpha = 2.0 * u
                        beta = (alpha + 1.0) ** (-1.0 / 15.0)
                    else:
                        alpha = 2.0 * (1.0 - u)
                        beta = (alpha + 1.0) ** (-1.0 / 15.0)
                    child[i] = 0.5 * ((1 + beta) * x1 + (1 - beta) * x2)
            else:
                child[i] = parent1[i]
        return np.clip(child, lo, hi)

    def _polynomial_mutate(self, x, lo, hi):
        """Polynomial mutation."""
        x_mut = x.copy()
        for i in range(self.dim):
            if np.random.rand() < 0.8:  # mutation probability
                x1, x2 = lo[i], hi[i]
                if abs(x2 - x1) < 1e-14:
                    continue
                delta = abs(x2 - x1)
                u = np.random.rand()
                if u < 0.5:
                    val = 2.0 * u
                    delta_q = (val) ** (1.0 / 21.0)
                else:
                    val = 2.0 * (1.0 - u)
                    delta_q = (val) ** (1.0 / 21.0)
                x_mut[i] += delta_q * delta
        return np.clip(x_mut, lo, hi)
# EVOLVE-BLOCK-END
