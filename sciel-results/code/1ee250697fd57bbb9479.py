"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer combining NSGA-II-style search with local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Population size: use a good chunk of budget for diversity
        pop_size = min(60, self.budget // 3)
        # Reserve some evaluations for local search
        local_search_budget = min(40, self.budget // 5)
        main_budget = self.budget - local_search_budget

        # Initial population: mix of random and structured points
        X = np.empty((pop_size, self.dim))
        # First few points: corners and midpoints of the hypercube
        n_structured = min(8, pop_size)
        for i in range(n_structured):
            # Use binary representation for corners
            bits = [(i >> j) & 1 for j in range(self.dim)]
            X[i] = lo + span * np.array(bits)
        # Fill rest with random
        if pop_size > n_structured:
            X[n_structured:] = lo + span * rng.random((pop_size - n_structured, self.dim))

        F = np.array([problem(x) for x in X])
        used = pop_size

        # Helper: check if a point is dominated
        def is_dominated(f, F):
            """Check if f is dominated by any point in F"""
            return np.any(np.all(F <= f, axis=1) & np.any(F < f, axis=1))

        # Helper: get non-dominated indices
        def get_nondominated_indices(F):
            n = len(F)
            nd_idx = []
            for i in range(n):
                if not is_dominated(F[i], F):
                    nd_idx.append(i)
            return np.array(nd_idx) if nd_idx else np.array([0])

        # Helper: crowding distance
        def crowding_distance(F, idx):
            if len(idx) <= 2:
                return np.ones(len(idx))
            cd = np.zeros(len(idx))
            F_sub = F[idx]
            for m in range(self.n_obj):
            # Sort by objective m
                order = np.argsort(F_sub[:, m])
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                for k in range(1, len(order) - 1):
                    cd[order[k]] += (F_sub[order[k + 1], m] - F_sub[order[k - 1], m])
            # Normalize
            if np.any(F_sub.max(axis=0) - F_sub.min(axis=0) > 0):
                cd /= (F_sub.max(axis=0) - F_sub.min(axis=0) + 1e-10)
            return cd

        # Main evolutionary loop
        while used < main_budget:
            # Check time
            if time.time() > deadline:
                break

            # Find non-dominated points
            nd_idx = get_nondominated_indices(F)

            # Select parents: prefer non-dominated, then by crowding distance
            if len(nd_idx) > 0:
                cd = crowding_distance(F, nd_idx)
                # Select parent from non-dominated set, weighted by crowding
                weights = cd + 1e-10
                weights /= weights.sum()
                parent_idx = nd_idx[rng.choice(len(nd_idx), p=weights)]
            else:
                parent_idx = rng.integers(pop_size)

            # Mutation: adaptive step size
            # Start with larger steps, decrease over time
            progress = used / main_budget
            step_size = 0.15 * (1 - progress) + 0.02
            child = X[parent_idx] + step_size * span * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Replace worst point in population
            # Use a simple criterion: sum of normalized objectives
            F_norm = (F - F.min(axis=0)) / (F.max(axis=0) - F.min(axis=0) + 1e-10)
            worst_idx = np.argmax(F_norm.sum(axis=1))
            X[worst_idx] = child
            F[worst_idx] = fc

        # Local search phase: refine the best non-dominated points
        if used < self.budget:
            nd_idx = get_nondominated_indices(F)
            # Pick a few best points for local refinement
            n_local = min(len(nd_idx), local_search_budget // 5)
            if n_local > 0:
                cd = crowding_distance(F, nd_idx)
                # Select points with high crowding distance (spread)
                local_candidates = nd_idx[np.argsort(cd)[-n_local:]]

                for lc_idx in local_candidates:
                    if used >= self.budget or time.time() > deadline:
                        break

                    # Nelder-Mead style local search around this point
                    x0 = X[lc_idx].copy()
                    f0 = F[lc_idx].copy()

                    # Create simplex
                    simplex = [x0.copy()]
                    for d in range(self.dim):
                        x_trial = x0.copy()
                        x_trial[d] += 0.05 * span[d]
                        x_trial = np.clip(x_trial, lo, hi)
                        simplex.append(x_trial)

                    simplex_F = [problem(x) for x in simplex]
                    used += self.dim

                    # Iterate local search
                    for _ in range(10):
                        if used >= self.budget or time.time() > deadline:
                            break

                        # Sort simplex by sum of objectives (simple scalarization)
                        order = np.argsort([np.sum(f) for f in simplex_F])
                        simplex = [simplex[i] for i in order]
                        simplex_F = [simplex_F[i] for i in order]

                        # Centroid of all but worst
                        centroid = np.mean(simplex[:-1], axis=0)

                        # Reflection
                        x_reflect = centroid + 1.0 * (centroid - simplex[-1])
                        x_reflect = np.clip(x_reflect, lo, hi)
                        f_reflect = problem(x_reflect)
                        used += 1

                        if np.sum(f_reflect) < np.sum(simplex_F[-1]):
                            simplex[-1] = x_reflect
                            simplex_F[-1] = f_reflect
                        else:
                            # Contraction
                            x_contract = centroid + 0.5 * (centroid - simplex[-1])
                            x_contract = np.clip(x_contract, lo, hi)
                            f_contract = problem(x_contract)
                            used += 1
                            if np.sum(f_contract) < np.sum(simplex_F[-1]):
                                simplex[-1] = x_contract
                                simplex_F[-1] = f_contract
                            else:
                                # Shrink
                                for i in range(1, len(simplex)):
                                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                    simplex[i] = np.clip(simplex[i], lo, hi)
                                    simplex_F[i] = problem(simplex[i])
                                    used += 1
                                    if used >= self.budget or time.time() > deadline:
                                        break

                    # Update population with best local search result
                    best_local_idx = np.argmin([np.sum(f) for f in simplex_F])
                    if np.sum(simplex_F[best_local_idx]) < np.sum(f0):
                        X[lc_idx] = simplex[best_local_idx]
                        F[lc_idx] = simplex_F[best_local_idx]

        return X
# EVOLVE-BLOCK-END
