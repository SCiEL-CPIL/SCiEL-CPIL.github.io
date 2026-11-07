"""Multi-objective optimizer for RE37 with Halton + NSGA-II + local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.eval_count = 0

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 315.0
        self.eval_count = 0

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj

        rng = np.random.default_rng(42)

        def eval_real(x):
            if self.eval_count >= self.budget:
                raise RuntimeError("budget exceeded")
            x = np.asarray(x, dtype=float).reshape(-1)
            if x.size != dim:
                x = np.resize(x, dim)
            x = np.clip(x, lo, hi)
            f = np.asarray(problem(x), dtype=float).reshape(-1)
            self.eval_count += 1
            return f

        # --- Halton sequence generator ---
        def halton(dim, start, count, primes=None):
            if primes is None:
                primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29][:dim]
            while len(primes) < dim:
                p = primes[-1] + 2
                while any(p % q == 0 for q in primes[:len(primes)//2 + 1]):
                    p += 2
                primes.append(p)
            points = np.zeros((count, dim))
            for j in range(dim):
                frac = np.zeros(count)
                prime = primes[j]
                for i in range(count):
                    f = 1.0
                    n = start + i
                    while n > 0:
                        n, rem = divmod(n, prime)
                        f /= prime
                        frac[i] += rem * f
                points[:, j] = frac
            return points

        # --- Nondominated sort ---
        def nondominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_count = [0] * n
            fronts = []
            for p in range(n):
                for q in range(n):
                    if p == q:
                        continue
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].append(q)
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        n_count[p] += 1
            front0 = [p for p in range(n) if n_count[p] == 0]
            if front0:
                fronts.append(front0)
            i = 0
            while fronts[i]:
                next_front = []
                for p in fronts[i]:
                    for q in S[p]:
                        n_count[q] -= 1
                        if n_count[q] == 0:
                            next_front.append(q)
                if next_front:
                    fronts.append(next_front)
                    i += 1
            return fronts

        # --- Crowding distance ---
        def crowding_distance(F, front_indices):
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n)
            m = len(F)
            for k in range(n_obj):
                order = np.argsort([F[front_indices[i], k] for i in range(n)])
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                fmin = F[front_indices[order[0]], k]
                fmax = F[front_indices[order[-1]], k]
                if fmax - fmin < 1e-15:
                    continue
                for i in range(1, n - 1):
                    cd[order[i]] += (F[front_indices[order[i + 1]], k] - F[front_indices[order[i - 1]], k]) / (fmax - fmin)
            return cd

        # --- NSGA-II crossover ---
        def crossover(x1, x2):
            alpha = rng.uniform(0.3, 0.7)
            c1 = alpha * x1 + (1 - alpha) * x2
            c2 = (1 - alpha) * x1 + alpha * x2
            return np.clip(c1, lo, hi), np.clip(c2, lo, hi)

        # --- Local refinement via coordinate descent ---
        def local_refine(x0, f0, n_iter=8):
            x = np.asarray(x0, dtype=float).copy()
            f = np.asarray(f0, dtype=float).copy()
            for _ in range(n_iter):
                if self.eval_count >= self.budget or time.time() > deadline:
                    break
                improved = False
                for d in range(dim):
                    for delta in [0.05, 0.02, 0.01, 0.005, -0.05, -0.02, -0.01, -0.005]:
                        if self.eval_count >= self.budget or time.time() > deadline:
                            return x, f
                        cand = x.copy()
                        cand[d] += delta
                        cand = np.clip(cand, lo, hi)
                        fc = eval_real(cand)
                        if np.all(fc <= f) and np.any(fc < f):
                            x = cand
                            f = fc
                            improved = True
                            break
                    if improved:
                        break
                if not improved:
                    break
            return x, f

        # --- Main optimization loop ---
        pop = []
        F = []

        def add_to_pop(x, f):
            pop.append(np.asarray(x, dtype=float))
            F.append(np.asarray(f, dtype=float))

        # Initial design: Halton + special points
        n_halton = 60
        halton_pts = halton(dim, 1, n_halton)
        specials = [
            lo, hi, (lo + hi) / 2.0,
            np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]),
            np.array([0, 1, 1, 0]), np.array([1, 0, 0, 1]),
            np.array([1, 0, 1, 0]), np.array([1, 1, 0, 0]),
        ]
        specials = np.array([np.clip(s, lo, hi) for s in specials], dtype=float)

        init_pts = np.vstack([halton_pts, specials])
        rng.shuffle(init_pts)

        for x in init_pts:
            if self.eval_count >= self.budget or time.time() > deadline:
                break
            try:
                f = eval_real(x)
                add_to_pop(x, f)
            except Exception:
                break

        if len(pop) == 0:
            return None

        # NSGA-II-style evolutionary loop
        pop_size = min(len(pop), 80)
        while self.eval_count < self.budget and time.time() < deadline:
            if len(pop) < 4:
                x = lo + (hi - lo) * rng.random(dim)
                f = eval_real(x)
                add_to_pop(x, f)
                continue

            # Sort by nondomination
            fronts = nondominated_sort(np.array(F))
            if not fronts:
                break

            # Select parents from first two fronts
            parents_pool = fronts[0].copy()
            if len(parents_pool) < max(4, len(pop) // 2) and len(fronts) > 1:
                parents_pool = parents_pool + fronts[1][:max(2, len(pop) // 4)]

            # Crossover to generate children
            children = []
            n_children = min(len(parents_pool), max(6, len(pop) // 4))
            for _ in range(n_children):
                if self.eval_count >= self.budget or time.time() > deadline:
                    break
                i1 = rng.integers(0, len(parents_pool))
                i2 = rng.integers(0, len(parents_pool))
                x1 = pop[parents_pool[i1]]
                x2 = pop[parents_pool[i2]]
                c1, c2 = crossover(x1, x2)
                # Mutate
                c1 = np.clip(c1 + rng.normal(0, 0.05, dim), lo, hi)
                c2 = np.clip(c2 + rng.normal(0, 0.05, dim), lo, hi)
                children.append(c1)
                children.append(c2)

            # Evaluate children
            child_F = []
            for cx in children:
                if self.eval_count >= self.budget or time.time() > deadline:
                    break
                try:
                    cf = eval_real(cx)
                    add_to_pop(cx, cf)
                    child_F.append(cf)
                except Exception:
                    break

            # Local refine the best nondominated points
            if time.time() < deadline and self.eval_count < self.budget:
                fronts = nondominated_sort(np.array(F))
                if fronts:
                    best_front = fronts[0][:min(5, len(fronts[0]))]
                    for idx in best_front:
                        if self.eval_count >= self.budget or time.time() > deadline:
                            break
                        rx, rf = local_refine(pop[idx], F[idx], n_iter=6)
                        # Replace or add
                        if np.all(rf <= F[idx]) and np.any(rf < F[idx]):
                            pop[idx] = rx
                            F[idx] = rf
                        else:
                            add_to_pop(rx, rf)

            # Trim population if too large
            if len(pop) > 120:
                fronts = nondominated_sort(np.array(F))
                keep = set()
                for front in fronts:
                    if len(keep) + len(front) <= 120:
                        keep.update(front)
                    else:
                        break
                if len(keep) < 120:
                    # Add by crowding distance from next fronts
                    all_remaining = [i for i in range(len(pop)) if i not in keep]
                    cd_all = np.zeros(len(pop))
                    for front in fronts:
                        cd_front = crowding_distance(np.array(F), front)
                        for j, idx in enumerate(front):
                            cd_all[idx] = cd_front[j]
                    remaining_sorted = sorted(all_remaining, key=lambda i: -cd_all[i])
                    keep.update(remaining_sorted[:120 - len(keep)])
                pop = [pop[i] for i in sorted(keep)]
                F = [F[i] for i in sorted(keep)]

        return None
# EVOLVE-BLOCK-END
