import numpy as np
import time
import random

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        def non_dominated_sort(F):
            n = len(F)
            if n == 0:
                return np.array([], dtype=int), np.array([])
            
            ranks = np.zeros(n, dtype=int)
            crowding = np.zeros(n)
            
            # Use numpy for faster dominance check
            front_mask = np.ones(n, dtype=bool)
            rank = 0
            while front_mask.any():
                front_indices = np.where(front_mask)[0]
                front_F = F[front_indices]
                
                # Find non-dominated within front
                nd_mask = np.ones(len(front_indices), dtype=bool)
                for i in range(len(front_indices)):
                    if not nd_mask[i]:
                        continue
                    # Check if any other point dominates this one
                    dominated = np.any(
                        np.all(front_F <= front_F[i], axis=1) & 
                        np.any(front_F < front_F[i], axis=1) & 
                        (np.arange(len(front_indices)) != i)
                    )
                    if dominated:
                        nd_mask[i] = False
                
                nd_in_front = front_indices[nd_mask]
                
                for i in nd_in_front:
                    ranks[i] = rank
                
                if len(nd_in_front) > 1:
                    for k in range(n_obj):
                        sorted_front = sorted(nd_in_front, key=lambda x: F[x, k])
                        f_min = F[sorted_front[0], k]
                        f_max = F[sorted_front[-1], k]
                        if f_max <= f_min:
                            continue
                        for idx, i in enumerate(sorted_front):
                            if idx == 0 or idx == len(sorted_front) - 1:
                                crowding[i] = np.inf
                            else:
                                crowding[i] += (F[sorted_front[idx+1], k] - F[sorted_front[idx-1], k]) / (f_max - f_min)
                
                rank += 1
                front_mask[nd_in_front] = False
            
            return ranks, crowding
        
        def tournament_select(X, F, ranks, k=2):
            indices = np.random.randint(0, len(F), size=k)
            best = indices[0]
            for i in indices[1:]:
                if ranks[i] < ranks[best]:
                    best = i
            return X[best]
        
        def sbcrossover(x1, x2, eta=20.0):
            x = np.zeros_like(x1)
            for i in range(dim):
                if np.random.rand() < 0.5:
                    if abs(x1[i] - x2[i]) < 1e-14:
                        x[i] = x1[i]
                    else:
                        u = np.random.rand()
                        if u <= 0.5:
                            beta = (2 * u) ** (1.0 / (eta + 1))
                        else:
                            beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                        x[i] = 0.5 * ((x1[i] + x2[i]) - beta * (x1[i] - x2[i]))
                else:
                    if abs(x1[i] - x2[i]) < 1e-14:
                        x[i] = x1[i]
                    else:
                        u = np.random.rand()
                        if u <= 0.5:
                            beta = (2 * u) ** (1.0 / (eta + 1))
                        else:
                            beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                        x[i] = 0.5 * ((x1[i] + x2[i]) + beta * (x1[i] - x2[i]))
            return np.clip(x, lo, hi)
        
        def polynomial_mutation(x, pmut=1.0/dim, eta=15.0):
            x_new = x.copy()
            for i in range(dim):
                if np.random.rand() < pmut:
                    range_i = hi[i] - lo[i]
                    if range_i < 1e-14:
                        continue
                    y = (x_new[i] - lo[i]) / range_i
                    if y < 0: y = 0
                    if y > 1: y = 1
                    u = np.random.rand()
                    if u < 0.5:
                        delta1 = (1 + 2 * (y - 0)) ** (eta + 1)
                        delta = (1 / delta1) ** (1 / (eta + 1)) - 1
                        y_new = y + delta * (1 - y)
                    else:
                        delta2 = (1 + 2 * (1 - y)) ** (eta + 1)
                        delta = 1 - (1 / delta2) ** (1 / (eta + 1))
                        y_new = y + delta * y
                    x_new[i] = lo[i] + y_new * range_i
            return np.clip(x_new, lo, hi)
        
        def update_archive(archive_X, archive_F, new_X, new_F):
            # Combine archive and new points
            X_all = np.vstack([archive_X, new_X])
            F_all = np.vstack([archive_F, new_F])
            
            n = len(X_all)
            is_nd = np.ones(n, dtype=bool)
            
            # Vectorized dominance check for archive update
            # A point i is dominated if there exists j such that F_all[j] <= F_all[i] and F_all[j] != F_all[i]
            for i in range(n):
                if not is_nd[i]:
                    continue
                # Check if any other point dominates i
                # Dominates: F[j] <= F[i] for all obj, and F[j] < F[i] for at least one obj
                dominated_by = np.all(F_all <= F_all[i], axis=1) & np.any(F_all < F_all[i], axis=1)
                dominated_by[i] = False # Self-dominance is false
                if np.any(dominated_by):
                    is_nd[i] = False
            
            return X_all[is_nd], F_all[is_nd]
        
        def local_refinement(problem, lo, hi, archive_X, archive_F, remaining, deadline):
            """Local refinement using scalarized objectives."""
            dim = hi.shape[0]
            n_obj = archive_F.shape[1]
            n_archive = len(archive_X)
            if n_archive == 0:
                return archive_X, archive_F

            # Allocate budget among archive points
            per_point = max(1, remaining // n_archive)
            
            for idx in range(n_archive):
                if time.time() > deadline:
                    break
                if remaining <= 0:
                    break

                for _ in range(min(per_point, remaining)):
                    if time.time() > deadline:
                        break
                    if remaining <= 0:
                        break
                        
                    # Random weighted sum direction
                    weights = np.random.dirichlet(np.ones(n_obj))
                    base = archive_X[idx].copy()
                    step = 0.05 * (hi - lo)
                    candidate = base + step * np.random.randn(dim)
                    candidate = np.clip(candidate, lo, hi)

                    fc = problem(candidate)
                    remaining -= 1

                    # Accept if non-dominated or improves scalarized objective
                    scalar_old = np.dot(weights, archive_F[idx])
                    scalar_new = np.dot(weights, fc)

                    if scalar_new < scalar_old:
                        archive_X[idx] = candidate
                        archive_F[idx] = fc
                    else:
                        # Check if candidate is non-dominated w.r.t the whole archive
                        # If it is, add it (and potentially remove dominated archive points)
                        # Simplified: just check if it's better than the specific point or non-dominated globally
                        # For efficiency, we only add if it's not dominated by the current best point in that direction
                        # A more rigorous check:
                        is_dominated = False
                        for j in range(len(archive_F)):
                            if np.all(archive_F[j] <= fc) and np.any(archive_F[j] < fc):
                                is_dominated = True
                                break
                        if not is_dominated:
                            # Add to archive
                            archive_X = np.vstack([archive_X, candidate])
                            archive_F = np.vstack([archive_F, fc])
                            # Optional: prune dominated points from archive to keep it manageable
                            # But this is O(N^2), so we might skip pruning if archive grows too large
                            if len(archive_X) > 200:
                                # Keep only non-dominated
                                archive_X, archive_F = update_archive(archive_X, archive_F, np.empty((0, dim)), np.empty((0, n_obj)))

            return archive_X, archive_F

        # Initialize
        pop_size = 50
        X = np.random.uniform(lo, hi, size=(pop_size, dim))
        F = np.array([problem(x) for x in X])
        evaluations = pop_size
        
        # Compute initial ranks
        ranks, crowding = non_dominated_sort(F)
        
        # Initialize Archive
        archive_X, archive_F = update_archive(np.empty((0, dim)), np.empty((0, n_obj)), X, F)
        
        while evaluations < self.budget:
            if time.time() > deadline:
                break
            
            n_off = min(pop_size, self.budget - evaluations)
            X_off = []
            for _ in range(n_off):
                p1 = tournament_select(X, F, ranks)
                p2 = tournament_select(X, F, ranks)
                child = sbcrossover(p1, p2)
                child = polynomial_mutation(child)
                X_off.append(child)
            
            X_off = np.array(X_off)
            F_off = np.array([problem(x) for x in X_off])
            evaluations += n_off
            
            X_all = np.vstack([X, X_off])
            F_all = np.vstack([F, F_off])
            
            # Update Archive with offspring (and parents, though parents are already there)
            archive_X, archive_F = update_archive(archive_X, archive_F, X_off, F_off)
            
            ranks, crowding = non_dominated_sort(F_all)
            sort_idx = np.lexsort(((-crowding), ranks))
            X = X_all[sort_idx[:pop_size]]
            F = F_all[sort_idx[:pop_size]]
        
        # Use remaining budget for local refinement of archive points
        remaining = self.budget - evaluations
        if remaining > 0 and len(archive_X) > 0:
            archive_X, archive_F = local_refinement(problem, lo, hi, archive_X, archive_F, remaining, deadline)

        return np.zeros((1, dim))
