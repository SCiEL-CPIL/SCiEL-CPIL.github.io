"""Multi-objective optimizer for WFG7 using NSGA-II with crowding distance and local refinement."""
import time
import numpy as np


class Optimizer:
    """NSGA-II based multi-objective optimizer for WFG7."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        rng = np.random.default_rng(42)

        # NSGA-II parameters
        pop_size = 30
        F = 0.8  # Differential weight for mutation
        CR = 0.9  # Crossover rate

        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F_pop = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        # Archive for non-dominated solutions (final result)
        archive_X = list(X)
        archive_F = list(F_pop)

        def _fast_nondominated_sort(F):
            """Fast non-dominated sorting. Returns list of fronts (lists of indices)."""
            n = len(F)
            S = [[] for _ in range(n)]  # Dominated set
            n_d = np.zeros(n, dtype=int)  # Number of dominators
            fronts = []
            
            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_d[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_d[i] += 1
            
            current_front = [i for i in range(n) if n_d[i] == 0]
            fronts.append(current_front)
            
            while current_front:
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_d[j] -= 1
                        if n_d[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                current_front = next_front
            
            return fronts

        def _crowding_distance(F, indices):
            """Compute crowding distance for a set of indices in F."""
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)
            
            dist = np.zeros(n)
            m = len(indices)
            for k in range(n_obj):
                # Sort by kth objective
                sorted_idx = sorted(range(m), key=lambda i: F[indices[i]][k])
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                
                f_min = F[indices[sorted_idx[0]]][k]
                f_max = F[indices[sorted_idx[-1]]][k]
                range_k = f_max - f_min
                if range_k < 1e-15:
                    continue
                    
                for i in range(1, m - 1):
                    dist[sorted_idx[i]] += (
                        F[indices[sorted_idx[i + 1]]][k] - F[indices[sorted_idx[i - 1]]][k]
                    ) / range_k
            
            return dist

        def _dominates(f1, f2):
            return np.all(f1 <= f2) and np.any(f1 < f2)

        # NSGA-II main loop
        while used < budget and time.time() < deadline - 30:
            # Create offspring population
            offspring_X = []
            offspring_F = []
            
            for _ in range(pop_size):
                if used >= budget or time.time() > deadline - 10:
                    break
                
                # Tournament selection (size 3) for two parents
                parent1, parent2 = None, None
                for _ in range(3):
                    idx = rng.integers(pop_size)
                    if parent1 is None or rng.random() < 0.5:
                        # Keep parent1 if it has higher rank (lower index in fronts)
                        # For simplicity, just random selection with tournament
                        pass
                
                # Simple binary tournament
                idx1, idx2 = rng.choice(pop_size, size=2, replace=False)
                # For now, just pick randomly, tournament would require rank info
                # Let's do a proper tournament using the current population
                p1, p2 = None, None
                for _ in range(2):
                    candidates = rng.choice(pop_size, size=2, replace=False)
                    # Compare using dominance - for speed, just pick the one with better sum
                    if np.sum(F_pop[candidates[0]]) < np.sum(F_pop[candidates[1]]):
                        parent = candidates[0]
                    else:
                        parent = candidates[1]
                    if p1 is None:
                        p1 = parent
                    else:
                        p2 = parent
                        break
                
                if p2 is None:
                    p2 = p1
                
                # Differential mutation
                idx3 = rng.integers(pop_size)
                v = X[p1] + F * (X[p2] - X[idx3])
                
                # Binomial crossover
                u = X[p1].copy()
                j_rand = rng.integers(dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        u[j] = v[j]
                
                u = np.clip(u, lo, hi)
                
                # Evaluate
                f_u = problem(u)
                used += 1
                
                offspring_X.append(u)
                offspring_F.append(f_u)

            if not offspring_X:
                break

            offspring_F = np.array(offspring_F)
            
            # Combine parent and offspring
            combined_X = np.vstack([X, np.array(offspring_X)])
            combined_F = np.vstack([F_pop, offspring_F])
            
            # Fast non-dominated sorting
            fronts = _fast_nondominated_sort(combined_F)
            
            # Select next population
            next_X = []
            next_F = []
            for front in fronts:
                if len(next_F) + len(front) <= pop_size:
                    for idx in front:
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                else:
                    # Need to select by crowding distance
                    remaining = pop_size - len(next_F)
                    if remaining > 0:
                        # Compute crowding distance within this front
                        front_indices = list(front)
                        front_F = np.array([combined_F[i] for i in front_indices])
                        cd = _crowding_distance(combined_F, front_indices)
                        sorted_by_cd = sorted(range(len(front_indices)), key=lambda i: cd[i], reverse=True)
                        for i in range(remaining):
                            idx = front_indices[sorted_by_cd[i]]
                            next_X.append(combined_X[idx])
                            next_F.append(combined_F[idx])
                    break
            
            if len(next_X) == pop_size:
                X = np.array(next_X)
                F_pop = np.array(next_F)
            
            # Update archive with non-dominated solutions
            for i in range(len(F_pop)):
                is_nd = True
                for af in archive_F:
                    if _dominates(af, F_pop[i]):
                        is_nd = False
                        break
                if is_nd:
                    # Remove dominated points from archive
                    new_archive_F = []
                    new_archive_X = []
                    for j in range(len(archive_F)):
                        if not _dominates(F_pop[i], archive_F[j]):
                            new_archive_F.append(archive_F[j])
                            new_archive_X.append(archive_X[j])
                    archive_F = new_archive_F
                    archive_X = new_archive_X
                    archive_F.append(F_pop[i])
                    archive_X.append(X[i])

        # Local refinement using Nelder-Mead on best archive points
        if used < budget - 10 and time.time() < deadline - 20:
            # Select diverse points from archive
            sorted_indices = np.argsort(np.sum(archive_F, axis=1))
            candidates = []
            for idx in sorted_indices[:5]:
                if len(candidates) >= 3:
                    break
                is_diverse = True
                for c in candidates:
                    if np.linalg.norm(archive_X[idx] - c) < 0.1 * np.linalg.norm(hi - lo):
                        is_diverse = False
                        break
                if is_diverse:
                    candidates.append(archive_X[idx])
            
            for x0 in candidates:
                if used >= budget or time.time() > deadline - 5:
                    break
                
                # Simple Nelder-Mead for multi-objective (using sum as scalarization)
                x = x0.copy()
                f = problem(x)
                used += 1
                
                simplex_X = [x.copy()]
                simplex_F = [f.copy()]
                for d in range(min(dim, 8)):  # Limit simplex size
                    x_d = x.copy()
                    x_d[d] += 0.05 * (hi[d] - lo[d])
                    x_d = np.clip(x_d, lo, hi)
                    f_d = problem(x_d)
                    used += 1
                    simplex_X.append(x_d)
                    simplex_F.append(f_d)
                
                for _ in range(5):
                    if used >= budget or time.time() > deadline - 5:
                        break
                    order = np.argsort(np.sum(simplex_F, axis=1))
                    simplex_X = [simplex_X[i] for i in order]
                    simplex_F = [simplex_F[i] for i in order]
                    
                    centroid = np.mean(simplex_X[:-1], axis=0)
                    xr = centroid + (centroid - simplex_X[-1])
                    xr = np.clip(xr, lo, hi)
                    fr = problem(xr)
                    used += 1
                    
                    if np.sum(fr) < np.sum(simplex_F[0]):
                        xe = centroid + 2 * (centroid - simplex_X[-1])
                        xe = np.clip(xe, lo, hi)
                        fe = problem(xe)
                        used += 1
                        if np.sum(fe) < np.sum(fr):
                            simplex_X[-1] = xe
                            simplex_F[-1] = fe
                        else:
                            simplex_X[-1] = xr
                            simplex_F[-1] = fr
                    elif np.sum(fr) < np.sum(simplex_F[-1]):
                        simplex_X[-1] = xr
                        simplex_F[-1] = fr
                    else:
                        xc = centroid + 0.5 * (simplex_X[-1] - centroid)
                        xc = np.clip(xc, lo, hi)
                        fc = problem(xc)
                        used += 1
                        if np.sum(fc) < np.sum(simplex_F[-1]):
                            simplex_X[-1] = xc
                            simplex_F[-1] = fc
                        else:
                            for i in range(1, len(simplex_X)):
                                simplex_X[i] = simplex_X[0] + 0.5 * (simplex_X[i] - simplex_X[0])
                                f_shrink = problem(simplex_X[i])
                                used += 1
                                simplex_F[i] = f_shrink
                                if used >= budget or time.time() > deadline - 5:
                                    break
                            break

                best_idx = np.argmin(np.sum(simplex_F, axis=1))
                if _is_non_dominated(archive_F, simplex_F[best_idx]):
                    archive_F = _remove_dominated(archive_F, simplex_F[best_idx])
                    archive_F.append(simplex_F[best_idx])
                    archive_X.append(simplex_X[best_idx])

        return archive_X
