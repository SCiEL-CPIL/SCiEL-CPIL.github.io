"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II with structured initialization near WFG9 Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 310.0
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj

        all_F = []
        used = 0

        def evaluate(x):
            nonlocal used
            x = np.clip(x, lo, hi)
            f = np.array(problem(x))
            if np.all(np.isfinite(f)):
                all_F.append(f)
            used += 1
            return f

        # Structured initialization: sample near origin (WFG9 Pareto front region)
        # Mix: 50% near origin (radius 0.3), 30% medium (radius 0.7), 20% full range
        pop_size = min(40, max(10, self.budget // 5))

        X = []
        for i in range(pop_size):
            if time.time() > deadline:
                break
            r = rng.random()
            if r < 0.5:
                radius = 0.3
            elif r < 0.8:
                radius = 0.7
            else:
                radius = 1.0
            direction = rng.standard_normal(dim)
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction /= norm
            u = rng.random()
            actual_r = radius * (u ** (1.0 / dim))
            x = actual_r * direction
            x = np.clip(x, lo, hi)
            X.append(x)

        F = np.array([evaluate(x) for x in X])

        # NSGA-II
        while used < self.budget:
            if time.time() > deadline:
                break

            # Non-dominated sorting
            n = len(F)
            ranks = np.zeros(n, dtype=int)
            fronts = []
            remaining = set(range(n))

            while remaining:
                # Find non-dominated set in remaining
                front = []
                for i in remaining:
                    dominated = False
                    for j in remaining:
                        if i == j:
                            continue
                        if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                            dominated = True
                            break
                    if not dominated:
                        front.append(i)
                for i in front:
                    ranks[i] = len(fronts)
                    remaining.discard(i)
                fronts.append(front)

            # Crowding distance
            crowding = np.zeros(n)
            for front in fronts:
                if len(front) <= 2:
                    for i in front:
                        crowding[i] = np.inf
                    continue
                for m in range(n_obj):
                    vals = F[front, m]
                    order = np.argsort(vals)
                    crowding[front[order[0]]] = np.inf
                    crowding[front[order[-1]]] = np.inf
                    v_range = vals[order[-1]] - vals[order[0]]
                    if v_range > 0:
                        for k in range(1, len(front) - 1):
                            idx = front[order[k]]
                            crowding[idx] += (vals[order[k + 1]] - vals[order[k - 1]]) / v_range

            # Selection: binary tournament
            def tournament():
                i, j = rng.integers(0, n, size=2)
                if ranks[i] < ranks[j]:
                    return i
                elif ranks[j] < ranks[i]:
                    return j
                else:
                    return i if crowding[i] >= crowding[j] else j

            # Crossover and mutation
            children_X = []
            children_F = []
            n_children = pop_size
            while len(children_X) < n_children:
                if time.time() > deadline:
                    break
                p1 = X[tournament()]
                p2 = X[tournament()]

                # Simulated binary crossover
                if rng.random() < 0.9:
                    child1 = np.zeros(dim)
                    child2 = np.zeros(dim)
                    for d in range(dim):
                        if rng.random() < 0.5:
                            if abs(p1[d] - p2[d]) > 1e-10:
                                beta1 = (p1[d] / p2[d]) ** 2.0 if p2[d] != 0 else 1.0
                                beta2 = (p2[d] / p1[d]) ** 2.0 if p1[d] != 0 else 1.0
                                child1[d] = 0.5 * (p1[d] + p2[d] - beta1 * (p2[d] - p1[d]))
                                child2[d] = 0.5 * (p1[d] + p2[d] + beta2 * (p2[d] - p1[d]))
                            else:
                                child1[d] = p1[d]
                                child2[d] = p2[d]
                        else:
                            child1[d] = p2[d]
                            child2[d] = p1[d]
                else:
                    child1 = p1.copy()
                    child2 = p2.copy()

                # Gaussian mutation
                sigma = 0.1
                for d in range(dim):
                    if rng.random() < 0.3:
                        child1[d] += sigma * rng.standard_normal()
                    if rng.random() < 0.3:
                        child2[d] += sigma * rng.standard_normal()

                child1 = np.clip(child1, lo, hi)
                child2 = np.clip(child2, lo, hi)

                f1 = evaluate(child1)
                f2 = evaluate(child2)
                children_X.append(child1)
                children_F.append(f1)
                children_X.append(child2)
                children_F.append(f2)

                if used >= self.budget:
                    break

            if not children_X:
                break

            # Combine parents and children
            combined_X = X + children_X
            combined_F = np.vstack([F, np.array(children_F)])
            n_combined = len(combined_F)

            # Non-dominated sort on combined
            ranks_c = np.zeros(n_combined, dtype=int)
            fronts_c = []
            remaining_c = set(range(n_combined))
            while remaining_c:
                front = []
                for i in remaining_c:
                    dominated = False
                    for j in remaining_c:
                        if i == j:
                            continue
                        if np.all(combined_F[j] <= combined_F[i]) and np.any(combined_F[j] < combined_F[i]):
                            dominated = True
                            break
                    if not dominated:
                        front.append(i)
                for i in front:
                    ranks_c[i] = len(fronts_c)
                    remaining_c.discard(i)
                fronts_c.append(front)

            # Crowding distance on combined
            crowding_c = np.zeros(n_combined)
            for front in fronts_c:
                if len(front) <= 2:
                    for i in front:
                        crowding_c[i] = np.inf
                    continue
                for m in range(n_obj):
                    vals = combined_F[front, m]
                    order = np.argsort(vals)
                    crowding_c[front[order[0]]] = np.inf
                    crowding_c[front[order[-1]]] = np.inf
                    v_range = vals[order[-1]] - vals[order[0]]
                    if v_range > 0:
                        for k in range(1, len(front) - 1):
                            idx = front[order[k]]
                            crowding_c[idx] += (vals[order[k + 1]] - vals[order[k - 1]]) / v_range

            # Select best pop_size from combined
            selected = []
            for front in fronts_c:
                if len(selected) + len(front) <= pop_size:
                    selected.extend(front)
                else:
                    need = pop_size - len(selected)
                    front_sorted = sorted(front, key=lambda i: crowding_c[i], reverse=True)
                    selected.extend(front_sorted[:need])
                    break

            X = [combined_X[i] for i in selected]
            F = combined_F[selected]

        return all_F
# EVOLVE-BLOCK-END
