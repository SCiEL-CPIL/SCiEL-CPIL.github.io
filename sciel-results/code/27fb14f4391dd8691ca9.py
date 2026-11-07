"""Multi-objective optimizer for RE21 combining NSGA-II structure with local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer using NSGA-II-like non-dominated sorting
    with local refinement (Nelder-Mead) on promising points.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 290.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # All evaluated points and their objective values
        all_X = []
        all_F = []
        used = 0

        def evaluate(x):
            nonlocal used
            x = np.asarray(x, dtype=float)
            f = np.array(problem(x), dtype=float)
            used += 1
            all_X.append(x.copy())
            all_F.append(f)
            return f

        def dominates(f1, f2):
            """Returns True if f1 dominates f2 (strictly better in at least one, no worse in any)."""
            return np.all(f1 <= f2) and np.any(f1 < f2)

        def is_non_dominated(F, idx):
            """Check if point idx is non-dominated in F."""
            f = F[idx]
            for j in range(len(F)):
                if j != idx and dominates(F[j], f):
                    return False
            return True

        def fast_non_dominated_sort(F):
            """
            Fast non-dominated sorting.
            Returns list of fronts, each front is a list of indices.
            """
            n = len(F)
            if n == 0:
                return []

            S = [[] for _ in range(n)]  # S[i] = set of solutions dominated by i
            n_dom = np.zeros(n, dtype=int)  # Number of solutions dominating i
            rank = np.zeros(n, dtype=int)

            for p in range(n):
                for q in range(n):
                    if p == q:
                        continue
                    if dominates(F[p], F[q]):
                        S[p].append(q)
                    elif dominates(F[q], F[p]):
                        n_dom[p] += 1

            fronts = []
            current_front = []
            for p in range(n):
                if n_dom[p] == 0:
                    rank[p] = 0
                    current_front.append(p)

            while current_front:
                fronts.append(current_front)
                next_front = []
                for p in current_front:
                    for q in S[p]:
                        n_dom[q] -= 1
                        if n_dom[q] == 0:
                            rank[q] = rank[p] + 1
                            next_front.append(q)
                current_front = next_front

            return fronts

        def crowding_distance(F, front_indices):
            """
            Calculate crowding distance for a front.
            Returns a dictionary mapping index to distance.
            """
            if not front_indices:
                return {}

            m = len(front_indices)
            if m <= 2:
                return {i: float('inf') for i in front_indices}

            dist = {i: 0.0 for i in front_indices}

            for k in range(n_obj):
                # Sort front by k-th objective
                sorted_front = sorted(front_indices, key=lambda i: F[i][k])

                # Boundary points have infinite distance
                dist[sorted_front[0]] = float('inf')
                dist[sorted_front[-1]] = float('inf')

                # Normalize by range
                f_min = F[sorted_front[0]][k]
                f_max = F[sorted_front[-1]][k]
                f_range = f_max - f_min

                if f_range == 0:
                    continue

                for i in range(1, m - 1):
                    dist[sorted_front[i]] += (F[sorted_front[i + 1]][k] - F[sorted_front[i - 1]][k]) / f_range

            return dist

        def tournament_select(pop_indices, rank_map, crowd_dists):
            """
            Binary tournament selection.
            Returns index of the winner.
            """
            i = int(np.random.randint(len(pop_indices)))
            j = int(np.random.randint(len(pop_indices)))
            idx_i = pop_indices[i]
            idx_j = pop_indices[j]

            rank_i = rank_map.get(idx_i, float('inf'))
            rank_j = rank_map.get(idx_j, float('inf'))

            if rank_i < rank_j:
                return idx_i
            elif rank_i > rank_j:
                return idx_j
            else:
                # Same rank, higher crowding distance is better
                dist_i = crowd_dists.get(idx_i, 0.0)
                dist_j = crowd_dists.get(idx_j, 0.0)
                return idx_i if dist_i > dist_j else idx_j

        def nelder_mead_local_search(x0, f0, max_iter=20):
            """
            Simple local refinement using Nelder-Mead-like steps.
            Returns the best point found.
            """
            if used >= budget or time.time() > deadline:
                return x0, f0

            # Initialize simplex
            simplex_X = [x0.copy()]
            simplex_F = [f0.copy()]

            # Create initial simplex
            for d in range(dim):
                x_new = x0.copy()
                step = 0.1 * (hi[d] - lo[d])
                x_new[d] += step
                x_new = np.clip(x_new, lo, hi)
                f_new = evaluate(x_new)
                simplex_X.append(x_new)
                simplex_F.append(f_new)

                if used >= budget or time.time() > deadline:
                    break

            for _ in range(max_iter):
                if used >= budget or time.time() > deadline:
                    break

                # Sort simplex by first objective (proxy for quality)
                order = np.argsort([f[0] for f in simplex_F])
                simplex_X = [simplex_X[i] for i in order]
                simplex_F = [simplex_F[i] for i in order]

                # Compute centroid of all but worst point
                centroid = np.mean(simplex_X[:-1], axis=0)

                # Reflection
                x_ref = centroid + (centroid - simplex_X[-1])
                x_ref = np.clip(x_ref, lo, hi)
                f_ref = evaluate(x_ref)

                if dominates(f_ref, simplex_F[0]):
                    # Expansion
                    x_exp = centroid + 2 * (centroid - simplex_X[-1])
                    x_exp = np.clip(x_exp, lo, hi)
                    f_exp = evaluate(x_exp)
                    if dominates(f_exp, f_ref):
                        simplex_X[-1] = x_exp
                        simplex_F[-1] = f_exp
                    else:
                        simplex_X[-1] = x_ref
                        simplex_F[-1] = f_ref
                elif not dominates(simplex_F[-1], f_ref):
                    # Contraction
                    x_con = centroid + 0.5 * (simplex_X[-1] - centroid)
                    x_con = np.clip(x_con, lo, hi)
                    f_con = evaluate(x_con)
                    if dominates(f_con, simplex_F[-1]):
                        simplex_X[-1] = x_con
                        simplex_F[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, len(simplex_X)):
                            simplex_X[i] = simplex_X[0] + 0.5 * (simplex_X[i] - simplex_X[0])
                            simplex_X[i] = np.clip(simplex_X[i], lo, hi)
                            simplex_F[i] = evaluate(simplex_X[i])
                            if used >= budget or time.time() > deadline:
                                break

            # Return best point from simplex
            best_idx = 0
            for i in range(len(simplex_F)):
                if dominates(simplex_F[i], simplex_F[best_idx]):
                    best_idx = i
            return simplex_X[best_idx], simplex_F[best_idx]

        # --- Main Algorithm ---

        # Population size
        mu = 20

        # 1. Initialize population
        for _ in range(mu):
            if used >= budget or time.time() > deadline:
                break
            x = lo + (hi - lo) * np.random.random(dim)
            f = evaluate(x)

        # 2. Evolution loop
        while used < budget and time.time() < deadline:
            # Convert to arrays for vectorized operations
            F_arr = np.array(all_F)
            X_arr = np.array(all_X)

            # Non-dominated sorting
            fronts = fast_non_dominated_sort(F_arr)

            # Build rank map
            rank_map = {}
            for rank, front in enumerate(fronts):
                for idx in front:
                    rank_map[idx] = rank

            # Crowding distance for the first front (most important)
            crowd_dists = {}
            if fronts:
                crowd_dists = crowding_distance(F_arr, fronts[0])

            # Generate offspring
            n_offspring = min(mu, budget - used)
            if n_offspring <= 0:
                break

            for _ in range(n_offspring):
                if used >= budget or time.time() > deadline:
                    break

                # Select two parents via tournament
                pop_indices = list(range(len(all_X)))
                p1 = tournament_select(pop_indices, rank_map, crowd_dists)
                p2 = tournament_select(pop_indices, rank_map, crowd_dists)

                # Crossover (SBX-like with uniform)
                if np.random.random() < 0.5:
                    alpha = np.random.random(dim)
                    x1 = (1 - alpha) * X_arr[p1] + alpha * X_arr[p2]
                    x2 = (1 - alpha) * X_arr[p2] + alpha * X_arr[p1]
                else:
                    x1 = X_arr[p1].copy()
                    x2 = X_arr[p2].copy()

                # Mutation
                for x in [x1, x2]:
                    x += 0.05 * (hi - lo) * np.random.standard_normal(dim)
                    x = np.clip(x, lo, hi)

                # Evaluate offspring
                f1 = evaluate(x1)
                if used >= budget or time.time() > deadline:
                    break
                f2 = evaluate(x2)

                # Local refinement on the better offspring
                if dominates(f1, f2):
                    best_x, best_f = x1, f1
                else:
                    best_x, best_f = x2, f2

                # Occasionally apply local search
                if np.random.random() < 0.3 and used < budget - 5:
                    refined_x, refined_f = nelder_mead_local_search(best_x, best_f, max_iter=10)
                    # The local search already added to all_X/all_F via evaluate()

        return all_X
# EVOLVE-BLOCK-END
