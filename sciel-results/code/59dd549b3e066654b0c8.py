# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Decomposition-based multi-objective optimizer (NSGA-III style) for RE34."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _generate_reference_points(self, n_obj, n_points):
        """Generate uniformly distributed reference points on the simplex."""
        if n_obj == 3:
            # Use a systematic grid approach for 3 objectives
            # Generate points on the simplex x+y+z = 1 with x,y,z >= 0
            points = []
            # Try different levels of subdivision
            for k in range(1, 15):
                for i in range(k + 1):
                    for j in range(k + 1 - i):
                        l = k - i - j
                        if l >= 0:
                            point = np.array([i / k, j / k, l / k])
                            if point[0] + point[1] + point[2] == 1.0:
                                # Avoid duplicates
                                is_dup = False
                                for p in points:
                                    if np.allclose(p, point, atol=1e-10):
                                        is_dup = True
                                        break
                                if not is_dup:
                                    points.append(point)
                if len(points) >= n_points:
                    break
            if len(points) < n_points:
                # Add more points with finer grid
                for k in range(15, 30):
                    for i in range(k + 1):
                        for j in range(k + 1 - i):
                            l = k - i - j
                            if l >= 0:
                                point = np.array([i / k, j / k, l / k])
                                if point[0] + point[1] + point[2] == 1.0:
                                    is_dup = False
                                    for p in points:
                                        if np.allclose(p, point, atol=1e-10):
                                            is_dup = True
                                            break
                                    if not is_dup:
                                        points.append(point)
                    if len(points) >= n_points:
                        break
            points = np.array(points[:n_points])
            return points
        else:
            # Generic: use Dirichlet sampling
            points = np.random.dirichlet(np.ones(n_obj), size=n_points)
            return points

    def _associate_points(self, F, ref_points, f_min, f_range):
        """Associate each point with the nearest reference point."""
        n = len(F)
        m = len(ref_points)
        if n == 0:
            return np.array([], dtype=int), np.zeros(n)
        
        # Normalize objectives
        F_norm = (F - f_min) / (f_range + 1e-10)
        
        # Compute distances to each reference point
        # For each point, compute the distance to each reference point
        # Using the angle-based or Euclidean distance in normalized space
        associations = np.zeros(n, dtype=int)
        distances = np.zeros(n)
        
        for i in range(n):
            # Compute distance from F_norm[i] to each reference point
            # We use the projection-based distance
            dists = np.zeros(m)
            for j in range(m):
                # Compute the distance
                diff = F_norm[i] - ref_points[j]
                dists[j] = np.linalg.norm(diff)
            best_j = np.argmin(dists)
            associations[i] = best_j
            distances[i] = dists[best_j]
        
        return associations, distances

    def _fast_non_dominated_sort(self, F):
        """Fast non-dominated sorting."""
        n = len(F)
        if n == 0:
            return []
        
        S = [[] for _ in range(n)]  # S[i] = indices dominated by i
        n_dom = np.zeros(n, dtype=int)  # Number of points that dominate i
        rank = np.zeros(n, dtype=int)
        
        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_dom[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    n_dom[i] += 1
        
        current_front = []
        for i in range(n):
            if n_dom[i] == 0:
                current_front.append(i)
                rank[i] = 0
        
        fronts = []
        front_idx = 0
        while current_front:
            fronts.append(current_front)
            next_front = []
            for i in current_front:
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        next_front.append(j)
                        rank[j] = front_idx + 1
            current_front = next_front
            front_idx += 1
        
        return fronts

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        ref = np.array([1705.0, 12.0, 0.3])

        rng = np.random

        # Generate reference points for decomposition
        n_ref = 50
        ref_points = self._generate_reference_points(n_obj, n_ref)

        # --- Phase 1: Initial sampling ---
        n_init = min(40, budget)
        X = np.zeros((n_init, dim))
        for d in range(dim):
            perm = rng.permutation(n_init)
            X[:, d] = lo[d] + (hi[d] - lo[d]) * (perm + rng.uniform(size=n_init)) / n_init
        X = np.clip(X, lo, hi)

        F = np.array([problem(x) for x in X])
        used = len(X)

        # Track all evaluated points
        all_X = X.copy()
        all_F = F.copy()

        # --- Phase 2: Evolutionary search ---
        # Maintain a population of size pop_size
        pop_size = min(30, budget - used)
        if pop_size < 5:
            pop_size = max(5, budget - used)

        # Initialize population from initial points
        # Select diverse points
        if len(F) > pop_size:
            # Use simple diversity-based selection
            selected = []
            remaining = list(range(len(F)))
            # Start with the best point
            sums = np.sum(F, axis=1)
            best_idx = np.argmin(sums)
            selected.append(best_idx)
            remaining.remove(best_idx)
            
            while len(selected) < pop_size and remaining:
                # Select the point farthest from already selected
                best_dist = -1
                best_idx = -1
                for idx in remaining:
                    min_dist = min(np.linalg.norm(F[idx] - F[s]) for s in selected)
                    if min_dist > best_dist:
                        best_dist = min_dist
                        best_idx = idx
                if best_idx >= 0:
                    selected.append(best_idx)
                    remaining.remove(best_idx)
            
            pop_X = F[selected]  # This is wrong, should be X
            pop_X = X[selected]
            pop_F = F[selected]
        else:
            pop_X = X.copy()
            pop_F = F.copy()

        # Main evolutionary loop
        while used < budget:
            if time.time() > deadline:
                break

            # Compute objective ranges for normalization
            f_min = np.min(pop_F, axis=0)
            f_max = np.max(pop_F, axis=0)
            f_range = f_max - f_min

            # Associate population points with reference points
            associations, distances = self._associate_points(pop_F, ref_points, f_min, f_range)

            # Select parents based on reference point association
            # For each reference point, select the best point associated with it
            parents_X = []
            parents_F = []
            for rp_idx in range(len(ref_points)):
                # Find points associated with this reference point
                mask = associations == rp_idx
                if np.any(mask):
                    # Select the point with minimum distance to the reference point
                    candidate_indices = np.where(mask)[0]
                    dists = distances[mask]
                    best_local = candidate_indices[np.argmin(dists)]
                    parents_X.append(pop_X[best_local])
                    parents_F.append(pop_F[best_local])
            
            if len(parents_X) == 0:
                # Fallback: random selection
                parents_X = [pop_X[rng.randint(len(pop_X))]]
                parents_F = [pop_F[rng.randint(len(pop_F))]]

            parents_X = np.array(parents_X)
            parents_F = np.array(parents_F)

            # Generate children
            n_children = min(pop_size, budget - used)
            children_X = np.zeros((n_children, dim))
            children_F = np.zeros((n_children, n_obj))

            for c in range(n_children):
                if used + c >= budget or time.time() > deadline:
                    break

                # Select two parents
                p1_idx = rng.randint(len(parents_X))
                p2_idx = rng.randint(len(parents_X))
                p1 = parents_X[p1_idx]
                p2 = parents_X[p2_idx]

                # SBX crossover
                child = np.zeros(dim)
                for d in range(dim):
                    if rng.uniform() < 0.5:
                        # Crossover
                        if rng.uniform() < 0.5:
                            child[d] = p1[d]
                        else:
                            child[d] = p2[d]
                    else:
                        # SBX
                        if rng.uniform() < 0.5:
                            beta = 1.0 + 2.0 * rng.uniform()
                            alpha = 1.0 / (beta + 1.0)
                            child[d] = (p1[d] + p2[d]) / 2.0 + alpha * (p1[d] - p2[d])
                        else:
                            beta = 1.0 + 2.0 * rng.uniform()
                            alpha = 1.0 / (beta + 1.0)
                            child[d] = (p1[d] + p2[d]) / 2.0 - alpha * (p1[d] - p2[d])

                # Mutation
                for d in range(dim):
                    if rng.uniform() < 0.5:  # 50% mutation rate
                        # Polynomial mutation
                        eta = 20.0
                        delta = rng.uniform()
                        if delta < 0.5:
                            mut = (2.0 * delta) ** (1.0 / (eta + 1.0)) - 1.0
                        else:
                            mut = 1.0 - (2.0 * (1.0 - delta)) ** (1.0 / (eta + 1.0))
                        child[d] = child[d] + mut * (hi[d] - lo[d])

                child = np.clip(child, lo, hi)
                children_X[c] = child
                children_F[c] = problem(child)
                used += 1

                all_X = np.vstack([all_X, [child]])
                all_F = np.vstack([all_F, [children_F[c]]])

            # Combine population and children
            combined_X = np.vstack([pop_X, children_X[:used - len(pop_X) - len(X) + len(pop_X)]])
            # Actually, let's be more careful
            n_new = used - len(X) - len(pop_X)
            if n_new > 0:
                new_children_X = children_X[:n_new]
                new_children_F = children_F[:n_new]
                combined_X = np.vstack([pop_X, new_children_X])
                combined_F = np.vstack([pop_F, new_children_F])
            else:
                combined_X = pop_X
                combined_F = pop_F

            # Non-dominated sorting
            fronts = self._fast_non_dominated_sort(combined_F)

            # Select next population
            next_pop_X = []
            next_pop_F = []
            for front in fronts:
                if len(next_pop_X) + len(front) <= pop_size:
                    for idx in front:
                        next_pop_X.append(combined_X[idx])
                        next_pop_F.append(combined_F[idx])
                else:
                    # Need to select from this front using reference point association
                    remaining = pop_size - len(next_pop_X)
                    front_X = combined_X[front]
                    front_F = combined_F[front]
                    
                    f_min_front = np.min(front_F, axis=0)
                    f_range_front = np.max(front_F, axis=0) - f_min_front
                    assoc, dist = self._associate_points(front_F, ref_points, f_min_front, f_range_front)
                    
                    # For each reference point, select the closest point
                    selected_in_front = []
                    for rp_idx in range(len(ref_points)):
                        mask = assoc == rp_idx
                        if np.any(mask):
                            candidates = np.where(mask)[0]
                            d = dist[mask]
                            best = candidates[np.argmin(d)]
                            if best not in selected_in_front:
                                selected_in_front.append(best)
                        if len(selected_in_front) >= remaining:
                            break
                    
                    if len(selected_in_front) < remaining:
                        # Fill with random points from the front
                        remaining_indices = [i for i in range(len(front)) if i not in selected_in_front]
                        if remaining_indices:
                            extra = rng.choice(remaining_indices, size=remaining - len(selected_in_front), replace=False)
                            selected_in_front.extend(extra.tolist())
                    
                    for idx in selected_in_front[:remaining]:
                        next_pop_X.append(front_X[idx])
                        next_pop_F.append(front_F[idx])
                    break

            if len(next_pop_X) > 0:
                pop_X = np.array(next_pop_X[:pop_size])
                pop_F = np.array(next_pop_F[:pop_size])
            else:
                # Fallback: keep the best points
                sums = np.sum(combined_F, axis=1)
                top_idx = np.argsort(sums)[:pop_size]
                pop_X = combined_X[top_idx]
                pop_F = combined_F[top_idx]

        # --- Phase 3: Local refinement on top points ---
        if used < budget and len(pop_F) > 0:
            # Sort by sum of objectives and refine the best few
            sums = np.sum(pop_F, axis=1)
            top_k = min(4, len(pop_X))
            top_idx = np.argsort(sums)[:top_k]

            for pi in top_idx:
                if used >= budget or time.time() > deadline - 5:
                    break
                parent = pop_X[pi].copy()

                # Simple coordinate descent refinement
                for coord in range(dim):
                    if used >= budget or time.time() > deadline - 3:
                        break

                    best_f = problem(parent).copy()
                    best_x = parent.copy()

                    for delta in [-0.1, -0.05, -0.02, 0.02, 0.05, 0.1]:
                        if used >= budget or time.time() > deadline - 2:
                            break
                        trial = parent.copy()
                        trial[coord] = np.clip(parent[coord] + delta * (hi[coord] - lo[coord]), lo[coord], hi[coord])
                        if np.allclose(trial, parent):
                            continue
                        ft = problem(trial)
                        used += 1
                        if np.all(ft <= best_f) and np.any(ft < best_f):
                            best_f = ft
                            best_x = trial

                    if not np.allclose(best_x, parent):
                        parent = best_x
                        all_X = np.vstack([all_X, [parent]])
                        all_F = np.vstack([all_F, [best_f]])

        return all_X

# EVOLVE-BLOCK-END
