# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for RE37 using Differential Evolution
    with Pareto front maintenance, adaptive mutation, and local search polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        rng = np.random.default_rng(0)

        def dominates(a, b):
            """Returns True if a dominates b (all objectives <= and at least one <)."""
            return np.all(a <= b) and np.any(a < b)

        def fast_non_dominated_sort(F):
            """Fast non-dominated sort. Returns list of fronts (lists of indices)."""
            n = len(F)
            if n == 0:
                return []
            
            S = [[] for _ in range(n)]
            n_doms = [0] * n
            fronts = []
            
            for i in range(n):
                for j in range(i + 1, n):
                    if dominates(F[i], F[j]):
                        S[i].append(j)
                        n_doms[j] += 1
                    elif dominates(F[j], F[i]):
                        S[j].append(i)
                        n_doms[i] += 1
            
            front = [i for i in range(n) if n_doms[i] == 0]
            if not front:
                return [list(range(n))]
            
            fronts.append(front)
            
            while front:
                next_front = []
                for i in front:
                    for j in S[i]:
                        n_doms[j] -= 1
                        if n_doms[j] == 0:
                            next_front.append(j)
                if not next_front:
                    break
                fronts.append(next_front)
                front = next_front
            
            return fronts

        def crowding_distance(front_indices, F):
            """Calculate crowding distance for points in a front."""
            n = len(front_indices)
            if n <= 2:
                return [float('inf')] * n
            
            distances = [0.0] * n
            m = len(F[front_indices[0]])
            
            for k in range(m):
                sorted_indices = sorted(front_indices, key=lambda i: F[i][k])
                
                distances[sorted_indices[0]] = float('inf')
                distances[sorted_indices[-1]] = float('inf')
                
                f_min = F[sorted_indices[0]][k]
                f_max = F[sorted_indices[-1]][k]
                f_range = f_max - f_min
                
                if f_range == 0:
                    continue
                
                for i in range(1, n - 1):
                    idx = sorted_indices[i]
                    distances[idx] += (F[sorted_indices[i + 1]][k] - F[sorted_indices[i - 1]][k]) / f_range
            
            return distances

        # Initialize with stratified sampling
        n_init = 20
        X = []
        F = []
        used = 0

        for _ in range(n_init):
            if used >= budget or time.time() > deadline:
                break
            x = rng.uniform(lo, hi)
            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=float)
            X.append(x)
            F.append(f)
            used += 1

        X = np.array(X)
        F = np.array(F)

        # Maintain an archive of non-dominated points
        archive = []
        
        for i in range(len(X)):
            archive.append((X[i], F[i]))
        
        def update_archive(archive, x_new, f_new):
            """Update archive with new point, removing dominated points."""
            for _, f_arch in archive:
                if dominates(f_arch, f_new):
                    return archive
            
            new_archive = []
            for x_arch, f_arch in archive:
                if not dominates(f_new, f_arch):
                    new_archive.append((x_arch, f_arch))
            
            new_archive.append((x_new, f_new))
            
            return new_archive

        # Phase 1: Differential Evolution
        F_de_start = 0.8
        F_de_end = 0.4
        CR = 1.0
        
        polish_threshold = 0.75 * budget
        
        while used < budget:
            if time.time() > deadline:
                break

            n_pop = len(X)
            if n_pop < 3:
                x = rng.uniform(lo, hi)
                x = np.clip(x, lo, hi)
                f = np.array(problem(x), dtype=float)
                X = np.vstack([X, x.reshape(1, -1)])
                F = np.vstack([F, f.reshape(1, -1)])
                used += 1
                continue

            # Adaptive DE weight
            progress = used / budget
            F_de = F_de_start - (F_de_start - F_de_end) * progress

            if used < polish_threshold:
                # Standard DE Mutation
                if len(archive) >= 3:
                    p1, p2, p3 = rng.choice(len(archive), size=3, replace=False)
                    X_p1 = archive[p1][0]
                    X_p2 = archive[p2][0]
                    X_p3 = archive[p3][0]
                else:
                    p1, p2, p3 = rng.choice(n_pop, size=3, replace=False)
                    X_p1 = X[p1]
                    X_p2 = X[p2]
                    X_p3 = X[p3]

                # Generate mutant
                v = X_p1 + F_de * (X_p2 - X_p3)
                v = np.clip(v, lo, hi)

                # Evaluate
                f_v = np.array(problem(v), dtype=float)
                used += 1

                # Update archive
                archive = update_archive(archive, v, f_v)

                # Add to population
                X = np.vstack([X, v.reshape(1, -1)])
                F = np.vstack([F, f_v.reshape(1, -1)])

                # Periodically prune population to maintain diversity
                if len(X) > 50:
                    fronts = fast_non_dominated_sort(F)
                    if fronts:
                        nds_indices = fronts[0]
                        X = X[nds_indices]
                        F = F[nds_indices]
            else:
                # Phase 2: Local Search Polish
                if len(archive) == 0:
                    x = rng.uniform(lo, hi)
                    x = np.clip(x, lo, hi)
                    f = np.array(problem(x), dtype=float)
                    used += 1
                    continue

                idx = rng.choice(len(archive))
                x_base, f_base = archive[idx]
                
                # Small perturbation with adaptive step size
                step_size = 0.05 * (hi - lo) * (1.0 - 0.5 * progress)
                v = x_base + rng.normal(0, step_size, dim)
                v = np.clip(v, lo, hi)
                
                # Evaluate
                f_v = np.array(problem(v), dtype=float)
                used += 1
                
                # Update archive
                archive = update_archive(archive, v, f_v)

        return X
# EVOLVE-BLOCK-END
