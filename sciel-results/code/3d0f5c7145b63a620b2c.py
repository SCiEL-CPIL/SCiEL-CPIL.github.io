import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # 40s safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Hyperparameters
        pop_size = 40
        eta_c = 15.0
        eta_m = 20.0
        p_c = 0.95
        p_m = 1.0 / dim

        used = 0

        def time_ok():
            return time.time() < deadline

        # --- Helpers ---
        def fast_non_dominated_sort(F):
            n = len(F)
            if n == 0:
                return [], []
            S = [set() for _ in range(n)]
            n_dominated = [0] * n
            rank = [0] * n

            for i in range(n):
                for j in range(i + 1, n):
                    fi = F[i]
                    fj = F[j]
                    if np.all(fi <= fj) and np.any(fi < fj):
                        S[i].add(j)
                        n_dominated[j] += 1
                    elif np.all(fj <= fi) and np.any(fj < fi):
                        S[j].add(i)
                        n_dominated[i] += 1

            fronts = []
            current_front = [i for i in range(n) if n_dominated[i] == 0]
            for i in current_front:
                rank[i] = 0

            k = 0
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dominated[j] -= 1
                        if n_dominated[j] == 0:
                            rank[j] = k + 1
                            next_front.append(j)
                current_front = next_front
                k += 1
            return fronts, rank

        def crowding_distance(F, front_indices):
            if len(front_indices) == 0:
                return {}
            if len(front_indices) == 1:
                return {front_indices[0]: float('inf')}

            dist = {i: 0.0 for i in front_indices}
            for m in range(n_obj):
                sorted_front = sorted(front_indices, key=lambda i: F[i, m])
                dist[sorted_front[0]] = float('inf')
                dist[sorted_front[-1]] = float('inf')

                f_min = F[sorted_front[0], m]
                f_max = F[sorted_front[-1], m]
                if f_max == f_min:
                    continue

                for idx in range(1, len(sorted_front) - 1):
                    i = sorted_front[idx]
                    prev_i = sorted_front[idx - 1]
                    next_i = sorted_front[idx + 1]
                    dist[i] += (F[next_i, m] - F[prev_i, m]) / (f_max - f_min)
            return dist

        def hypervolume_contribution(F, ref):
            """Compute hypervolume contribution of each point in F relative to ref.
            Returns array of contributions."""
            n = len(F)
            if n == 0:
                return np.array([])
            # Filter out dominated points for HV computation (optional optimization)
            # Simple O(n^2) HV for small n
            contrib = np.zeros(n)
            for i in range(n):
                # Volume of region dominated by F[i] but not by any other point
                # For 3 objectives, this is complex; use a simpler proxy:
                # Distance from ref minus distance to nearest neighbor in objective space
                # Actually, let's compute proper 3D HV contribution
                pass

            # For simplicity and speed, use a proxy: 
            # contribution[i] = product of (ref[m] - F[i][m]) for m in range(n_obj)
            # This is the volume of the box from F[i] to ref
            # But we need to subtract overlaps. For a quick proxy, just use the box volume
            # weighted by how "unique" the point is.
            volumes = np.zeros(n)
            for i in range(n):
                prod = 1.0
                valid = True
                for m in range(n_obj):
                    if F[i, m] >= ref[m]:
                        valid = False
                        break
                    prod *= (ref[m] - F[i, m])
                volumes[i] = prod if valid else 0.0
            return volumes

        def sbx_crossover(X1, X2):
            C1 = X1.copy()
            C2 = X2.copy()
            if np.random.rand() < p_c:
                for i in range(dim):
                    if np.random.rand() < 0.5:
                        if X1[i] == X2[i]:
                            beta = 1.0
                        else:
                            u = np.random.rand()
                            if u <= 0.5:
                                beta = (2.0 * u) ** (1.0 / (eta_c + 1.0))
                            else:
                                beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta_c + 1.0))
                        C1[i] = 0.5 * ((1.0 + beta) * X1[i] + (1.0 - beta) * X2[i])
                        C2[i] = 0.5 * ((1.0 - beta) * X1[i] + (1.0 + beta) * X2[i])
                        C1[i] = np.clip(C1[i], lo[i], hi[i])
                        C2[i] = np.clip(C2[i], lo[i], hi[i])
            return C1, C2

        def polynomial_mutation(x):
            x_mut = x.copy()
            for i in range(dim):
                if np.random.rand() < p_m:
                    u = np.random.rand()
                    delta1 = (2.0 * u) ** (1.0 / (eta_m + 1.0)) - 1.0
                    delta2 = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta_m + 1.0))
                    xmut1 = x[i] + delta1 * (hi[i] - lo[i])
                    xmut2 = x[i] + delta2 * (hi[i] - lo[i])
                    if np.random.rand() < 0.5:
                        x_mut[i] = np.clip(xmut1, lo[i], hi[i])
                    else:
                        x_mut[i] = np.clip(xmut2, lo[i], hi[i])
            return x_mut

        # --- Initialization ---
        X = lo + (hi - lo) * np.random.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        ref = np.array([3.0, 5.0, 7.0])

        # --- Main Loop ---
        gen = 0
        while used < budget and time_ok():
            gen += 1
            # Non-dominated sort
            fronts, ranks = fast_non_dominated_sort(F)

            # Crowding distance
            crowding = np.zeros(len(F))
            for front in fronts:
                if len(front) > 1:
                    cdist = crowding_distance(F, front)
                    for idx, val in cdist.items():
                        crowding[idx] = val

            # Parent selection: binary tournament with crowding
            def select_parent(ranks, crowding, n):
                idx1 = np.random.randint(n)
                idx2 = np.random.randint(n)
                if ranks[idx1] < ranks[idx2]:
                    return idx1
                elif ranks[idx2] < ranks[idx1]:
                    return idx2
                else:
                    if crowding[idx1] > crowding[idx2]:
                        return idx1
                    elif crowding[idx2] > crowding[idx1]:
                        return idx2
                    else:
                        return idx1 if np.random.rand() < 0.5 else idx2

            # Generate offspring
            offspring_X = []
            offspring_F = []

            n_pairs = pop_size // 2
            for _ in range(n_pairs):
                if used >= budget or not time_ok():
                    break

                p1 = select_parent(ranks, crowding, len(F))
                p2 = select_parent(ranks, crowding, len(F))

                c1, c2 = sbx_crossover(X[p1], X[p2])
                c1 = polynomial_mutation(c1)
                c2 = polynomial_mutation(c2)

                offspring_X.append(c1)
                offspring_F.append(problem(c1))
                used += 1

                if used >= budget or not time_ok():
                    break

                offspring_X.append(c2)
                offspring_F.append(problem(c2))
                used += 1

            if not offspring_X:
                break

            # Combine and select
            combined_X = np.vstack([X, np.array(offspring_X)])
            combined_F = np.vstack([F, np.array(offspring_F)])

            combined_fronts, combined_ranks = fast_non_dominated_sort(combined_F)

            next_X = []
            next_F = []
            remaining = pop_size

            for front in combined_fronts:
                if remaining <= 0:
                    break
                if len(front) <= remaining:
                    for idx in front:
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                    remaining -= len(front)
                else:
                    cdist = crowding_distance(combined_F, front)
                    sorted_indices = sorted(front, key=lambda i: cdist[i], reverse=True)
                    for idx in sorted_indices[:remaining]:
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                    remaining = 0

            X = np.array(next_X)
            F = np.array(next_F)

        # Final polish: if time remains, do a few local refinements
        if used < budget and time_ok():
            # Local refinement: small perturbations of best points
            fronts, ranks = fast_non_dominated_sort(F)
            if fronts:
                best_front = fronts[0]
                for idx in best_front:
                    if used >= budget or not time_ok():
                        break
                    for _ in range(3):
                        if used >= budget or not time_ok():
                            break
                        perturbed = X[idx].copy()
                        for i in range(dim):
                            if np.random.rand() < 0.3:
                                perturbed[i] += np.random.normal(0, 0.05 * (hi[i] - lo[i]))
                                perturbed[i] = np.clip(perturbed[i], lo[i], hi[i])
                        F_new = problem(perturbed)
                        used += 1
                        # Check if it dominates any existing point
                        improved = False
                        for j in range(len(F)):
                            if np.all(F_new <= F[j]) and np.any(F_new < F[j]):
                                improved = True
                                break
                        if improved:
                            # Replace a dominated point or add
                            for j in range(len(F)):
                                if np.all(F[j] <= F_new) and np.any(F[j] < F_new):
                                    F[j] = F_new
                                    X[j] = perturbed
                                    break
                            else:
                                F = np.vstack([F, [F_new]])
                                X = np.vstack([X, [perturbed]])

        return X
