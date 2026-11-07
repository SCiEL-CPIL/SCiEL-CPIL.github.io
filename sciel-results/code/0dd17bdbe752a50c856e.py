# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for RE37: Grid init + NSGA-II + Nelder-Mead weighted refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Use the global numpy RNG (seeded by evaluator via np.random.seed)
        class _RNG:
            def __init__(self):
                pass
            def permutation(self, n):
                return np.random.permutation(n)
            def random(self, size=None):
                return np.random.random(size)
            def random_integers(self, high, low=None):
                if low is None:
                    low = 0
                return np.random.randint(low, high)
            def standard_normal(self, size=None):
                return np.random.standard_normal(size)
            def choice(self, a, size=None, replace=True):
                return np.random.choice(a, size=size, replace=replace)

        rng = _RNG()

        # Track all evaluated points
        X_all = []
        F_all = []
        eval_count = 0

        def evaluate(x):
            nonlocal eval_count
            x = np.clip(x, lo, hi)
            f = np.array(problem(x))
            X_all.append(x.copy())
            F_all.append(f.copy())
            eval_count += 1
            return f

        def time_ok():
            return (time.time() - start_time) < deadline and eval_count < budget

        # Phase 1: Systematic initialization
        # Grid: 3^4 = 81 points for systematic coverage
        grid_size = 3
        grid_points = []
        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    for l in range(grid_size):
                        x = np.array([
                            lo[0] + (hi[0] - lo[0]) * i / (grid_size - 1),
                            lo[1] + (hi[1] - lo[1]) * j / (grid_size - 1),
                            lo[2] + (hi[2] - lo[2]) * k / (grid_size - 1),
                            lo[3] + (hi[3] - lo[3]) * l / (grid_size - 1),
                        ])
                        grid_points.append(x)

        # LHS points: 24 points for diversity
        lhs_size = 24
        lhs_points = np.zeros((lhs_size, dim))
        for d in range(dim):
            perm = rng.permutation(lhs_size)
            lhs_points[:, d] = (perm + rng.random(lhs_size)) / lhs_size
        lhs_points = lo + (hi - lo) * lhs_points

        # Evaluate all initial points
        for x in grid_points:
            if not time_ok():
                break
            evaluate(x)

        for x in lhs_points:
            if not time_ok():
                break
            evaluate(x)

        if not time_ok():
            return np.array(X_all)

        # Phase 2: NSGA-II style evolutionary loop
        pop_size = 40

        F_current = np.array(F_all)

        def fast_non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_dom = np.zeros(n, dtype=int)
            rank = np.zeros(n, dtype=int)

            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dom[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_dom[i] += 1

            fronts = []
            current_front = [i for i in range(n) if n_dom[i] == 0]
            k = 0
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    rank[i] = k
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                current_front = next_front
                k += 1

            return fronts, rank

        def crowding_distance(F, front):
            if len(front) <= 2:
                return {i: float('inf') for i in front}

            dist = {i: 0.0 for i in front}
            for m in range(n_obj):
                vals = sorted([(F[i][m], i) for i in front])
                dist[vals[0][1]] = float('inf')
                dist[vals[-1][1]] = float('inf')
                fmin = vals[0][0]
                fmax = vals[-1][0]
                if fmax - fmin < 1e-12:
                    continue
                for k in range(1, len(vals) - 1):
                    dist[vals[k][1]] += (vals[k + 1][0] - vals[k - 1][0]) / (fmax - fmin)
            return dist

        # Initialize population from non-dominated points
        fronts, rank = fast_non_dominated_sort(F_current)
        pop_indices = []
        for front in fronts:
            if len(pop_indices) + len(front) <= pop_size:
                pop_indices.extend(front)
            else:
                cd = crowding_distance(F_current, front)
                sorted_front = sorted(front, key=lambda i: cd.get(i, 0), reverse=True)
                remaining = pop_size - len(pop_indices)
                pop_indices.extend(sorted_front[:remaining])
                break

        if len(pop_indices) < pop_size:
            for i in range(len(pop_indices), min(pop_size, len(F_all))):
                pop_indices.append(i)

        X = np.array([X_all[i] for i in pop_indices])
        F = np.array([F_all[i] for i in pop_indices])

        # Evolutionary loop
        while time_ok():
            if eval_count >= budget - 1:
                break

            fronts, rank = fast_non_dominated_sort(F)
            cd = {}
            for front in fronts:
                cd.update(crowding_distance(F, front))

            def tournament():
                i = rng.random_integers(len(F))
                j = rng.random_integers(len(F))
                if rank[i] != rank[j]:
                    return i if rank[i] < rank[j] else j
                return i if cd.get(i, 0) > cd.get(j, 0) else j

            n_children = min(pop_size, budget - eval_count)
            children_X = []
            children_F = []

            for _ in range(n_children):
                if not time_ok():
                    break

                p1 = tournament()
                p2 = tournament()

                # SBX crossover
                if rng.random() < 0.9:
                    child = np.zeros(dim)
                    for d in range(dim):
                        if rng.random() < 0.5:
                            u = rng.random()
                            if u <= 0.5:
                                beta = (2 * u) ** (1 / (2 + 15))
                            else:
                                beta = (1 / (2 * (1 - u))) ** (1 / (2 + 15))
                            if X[p1][d] <= X[p2][d]:
                                child[d] = 0.5 * ((1 + beta) * X[p1][d] + (1 - beta) * X[p2][d])
                            else:
                                child[d] = 0.5 * ((1 + beta) * X[p2][d] + (1 - beta) * X[p1][d])
                        else:
                            child[d] = X[p1][d] if rng.random() < 0.5 else X[p2][d]
                else:
                    child = X[p1].copy()

                # Polynomial mutation
                for d in range(dim):
                    if rng.random() < 0.9:
                        u = rng.random()
                        if u < 0.5:
                            delta = (2 * u) ** (1 / (2 + 20)) - 1
                        else:
                            delta = 1 - (2 * (1 - u)) ** (1 / (2 + 20))
                        child[d] += delta * (hi[d] - lo[d])

                child = np.clip(child, lo, hi)
                fc = evaluate(child)
                children_X.append(child)
                children_F.append(fc)

            if not children_X:
                break

            children_X = np.array(children_X)
            children_F = np.array(children_F)

            X_all_gen = np.vstack([X, children_X])
            F_all_gen = np.vstack([F, children_F])

            fronts_all, rank_all = fast_non_dominated_sort(F_all_gen)

            new_X = []
            new_F = []
            for front in fronts_all:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(X_all_gen[idx])
                        new_F.append(F_all_gen[idx])
                else:
                    remaining = pop_size - len(new_X)
                    cd_front = crowding_distance(F_all_gen, front)
                    sorted_front = sorted(front, key=lambda i: cd_front.get(i, 0), reverse=True)
                    for idx in sorted_front[:remaining]:
                        new_X.append(X_all_gen[idx])
                        new_F.append(F_all_gen[idx])
                    break

            if len(new_X) < pop_size:
                for front in fronts_all:
                    for idx in front:
                        if len(new_X) >= pop_size:
                            break
                        if not any(np.allclose(X_all_gen[idx], nx) for nx in new_X):
                            new_X.append(X_all_gen[idx])
                            new_F.append(F_all_gen[idx])
                    if len(new_X) >= pop_size:
                        break

            X = np.array(new_X[:pop_size])
            F = np.array(new_F[:pop_size])

        # Phase 3: Nelder-Mead local refinement on weighted scalarizations
        remaining = budget - eval_count
        if remaining > 5 and time_ok():
            F_all_arr = np.array(F_all)
            fronts_all, _ = fast_non_dominated_sort(F_all_arr)
            nd_indices = fronts_all[0] if fronts_all else list(range(len(F_all)))

            # Generate diverse weight vectors
            weights = []
            # Systematic simplex lattice
            for w1 in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
                for w2 in [0.1, 0.2, 0.3, 0.4, 0.5]:
                    w3 = 1.0 - w1 - w2
                    if w3 >= 0.1:
                        weights.append([w1, w2, w3])

            # Add random weights for diversity
            for _ in range(8):
                w = rng.random(n_obj)
                w /= w.sum()
                weights.append(w)

            weights = np.array(weights)

            # Nelder-Mead implementation
            def nelder_mead(x0, w, max_iter=30):
                """Nelder-Mead local search on weighted sum scalarization."""
                if not time_ok():
                    return x0

                def scalar_obj(x):
                    if not time_ok():
                        return np.inf
                    f = evaluate(x)
                    return np.dot(f, w)

                x_best = x0.copy()
                f_best = scalar_obj(x0)

                # Create initial simplex
                simplex = [x_best.copy()]
                f_simplex = [f_best]

                for i in range(dim):
                    x_new = x_best.copy()
                    x_new[i] += 0.05 * (hi[i] - lo[i])
                    x_new = np.clip(x_new, lo, hi)
                    simplex.append(x_new)
                    f_simplex.append(scalar_obj(x_new))

                    if not time_ok():
                        break

                # Iterate
                for it in range(max_iter):
                    if not time_ok():
                        break

                    # Sort by objective value
                    order = np.argsort(f_simplex)
                    simplex = [simplex[i] for i in order]
                    f_simplex = [f_simplex[i] for i in order]

                    # Check convergence
                    if abs(f_simplex[-1] - f_simplex[0]) < 1e-8:
                        break

                    # Centroid of all but worst
                    centroid = np.mean(simplex[:-1], axis=0)

                    # Reflection
                    x_reflect = centroid + (centroid - simplex[-1])
                    x_reflect = np.clip(x_reflect, lo, hi)
                    f_reflect = scalar_obj(x_reflect)

                    if f_reflect < f_simplex[0]:
                        # Expansion
                        x_expand = centroid + 2 * (centroid - simplex[-1])
                        x_expand = np.clip(x_expand, lo, hi)
                        f_expand = scalar_obj(x_expand)
                        if f_expand < f_reflect:
                            simplex[-1] = x_expand
                            f_simplex[-1] = f_expand
                        else:
                            simplex[-1] = x_reflect
                            f_simplex[-1] = f_reflect
                    elif f_reflect < f_simplex[-2]:
                        simplex[-1] = x_reflect
                        f_simplex[-1] = f_reflect
                    else:
                        # Contraction
                        x_contract = centroid + 0.5 * (simplex[-1] - centroid)
                        x_contract = np.clip(x_contract, lo, hi)
                        f_contract = scalar_obj(x_contract)
                        if f_contract < f_simplex[-1]:
                            simplex[-1] = x_contract
                            f_simplex[-1] = f_contract
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                f_simplex[i] = scalar_obj(simplex[i])

                return simplex[0]

            # For each weight, find best starting point and refine
            for w in weights:
                if not time_ok():
                    break
                if remaining < 15:
                    break

                # Find the point that minimizes the weighted sum
                weighted_sums = F_all_arr @ w
                best_idx = np.argmin(weighted_sums)
                x0 = X_all[best_idx].copy()

                # Run Nelder-Mead
                x_refined = nelder_mead(x0, w, max_iter=25)

            # Phase 4: Final random sampling with quick local refinement
            if time_ok():
                n_final = min(20, budget - eval_count)
                for _ in range(n_final):
                    if not time_ok():
                        break

                    x_rand = lo + (hi - lo) * rng.random(dim)
                    f_rand = evaluate(x_rand)

                    # Quick local refinement: try small perturbations
                    best_x = x_rand.copy()
                    best_f = f_rand.copy()

                    for _ in range(3):
                        if not time_ok():
                            break
                        perturbation = rng.standard_normal(dim) * 0.03
                        new_x = np.clip(best_x + perturbation, lo, hi)
                        new_f = evaluate(new_x)

                        if np.all(new_f <= best_f) and np.any(new_f < best_f):
                            best_x = new_x
                            best_f = new_f

        return np.array(X_all)
# EVOLVE-BLOCK-END
