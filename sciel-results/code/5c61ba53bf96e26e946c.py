import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        deadline = time.time() + 325.0
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        all_X = []
        all_F = []
        used = 0

        def evaluate(x):
            nonlocal used
            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=float)
            used += 1
            all_X.append(x.copy())
            all_F.append(f)
            return f

        def dominates(f1, f2):
            return np.all(f1 <= f2) and np.any(f1 < f2)

        def non_dominated_sort(F):
            n = len(F)
            if n == 0:
                return []

            S = [[] for _ in range(n)]
            n_dom = np.zeros(n, dtype=int)
            rank = np.zeros(n, dtype=int)

            for p in range(n):
                for q in range(n):
                    if p == q:
                        continue
                    if dominates(F[p], F[q]):
                        S[p].append(q)
                        n_dom[q] += 1
                    elif dominates(F[q], F[p]):
                        S[q].append(p)
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
            if not front_indices:
                return {}

            m = len(front_indices)
            if m <= 2:
                return {i: float('inf') for i in front_indices}

            dist = {i: 0.0 for i in front_indices}

            for k in range(n_obj):
                sorted_front = sorted(front_indices, key=lambda i: F[i][k])

                dist[sorted_front[0]] = float('inf')
                dist[sorted_front[-1]] = float('inf')

                f_min = F[sorted_front[0]][k]
                f_max = F[sorted_front[-1]][k]
                f_range = f_max - f_min

                if f_range == 0:
                    continue

                for i in range(1, m - 1):
                    dist[sorted_front[i]] += (F[sorted_front[i+1]][k] - F[sorted_front[i-1]][k]) / f_range

            return dist

        def simplex_local_search(center, max_evals=8, step_size=None):
            """Local Nelder-Mead style simplex search around center."""
            if step_size is None:
                step_size = 0.1 * (hi - lo)
            
            # Build initial simplex
            simplex = [center.copy()]
            for j in range(dim):
                v = center.copy()
                v[j] += step_size[j]
                v = np.clip(v, lo, hi)
                simplex.append(v)
            
            # Evaluate all simplex vertices
            simplex_F = []
            for v in simplex:
                if used >= budget or time.time() > deadline:
                    break
                simplex_F.append(evaluate(v))
            
            if len(simplex_F) < 2:
                return
            
            # Simplex iterations
            max_iter = max_evals - len(simplex_F)
            for _ in range(max(0, max_iter)):
                if used >= budget or time.time() > deadline:
                    break
                
                # Sort by first objective (as proxy for quality)
                order = np.argsort([f[0] for f in simplex_F])
                simplex = [simplex[i] for i in order]
                simplex_F = [simplex_F[i] for i in order]
                
                worst = simplex[-1]
                worst_F = simplex_F[-1]
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                alpha = 1.0
                reflect = centroid + alpha * (centroid - worst)
                reflect = np.clip(reflect, lo, hi)
                ref_F = evaluate(reflect)
                
                if ref_F[0] < simplex_F[0][0]:
                    # Expansion
                    gamma = 2.0
                    expand = centroid + gamma * (centroid - worst)
                    expand = np.clip(expand, lo, hi)
                    exp_F = evaluate(expand)
                    if exp_F[0] < ref_F[0]:
                        simplex[-1] = expand
                        simplex_F[-1] = exp_F
                    else:
                        simplex[-1] = reflect
                        simplex_F[-1] = ref_F
                elif ref_F[0] < simplex_F[-2][0]:
                    simplex[-1] = reflect
                    simplex_F[-1] = ref_F
                else:
                    # Contraction
                    beta = 0.5
                    contract = centroid + beta * (worst - centroid)
                    contract = np.clip(contract, lo, hi)
                    con_F = evaluate(contract)
                    if con_F[0] < worst_F[0]:
                        simplex[-1] = contract
                        simplex_F[-1] = con_F
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], lo, hi)
                            simplex_F[i] = evaluate(simplex[i])
        
        # --- Grid-based decomposition strategy ---
        # Partition 4D space into hypercubes, optimize each with local simplex
        
        # Number of grid points per dimension (2^4 = 16 cells)
        grid_per_dim = 2
        n_cells = grid_per_dim ** dim
        
        # Generate cell centers
        cell_centers = []
        for idx in range(n_cells):
            center = np.zeros(dim)
            temp = idx
            for j in range(dim):
                pos = temp % grid_per_dim
                temp //= grid_per_dim
                # Place center in middle of cell
                frac = (pos + 0.5) / grid_per_dim
                center[j] = lo[j] + frac * (hi[j] - lo[j])
            cell_centers.append(center)
        
        # Shuffle cells for diversity
        np.random.shuffle(cell_centers)
        
        # Phase 1: Evaluate all cell centers (16 evaluations)
        for center in cell_centers:
            if used >= budget or time.time() > deadline:
                break
            evaluate(center)
        
        # Phase 2: Local simplex search on each cell, allocating remaining budget
        remaining_budget = budget - used
        evals_per_cell = max(1, remaining_budget // n_cells)
        
        for center in cell_centers:
            if used >= budget or time.time() > deadline:
                break
            # Reserve some budget for exploration
            local_budget = min(evals_per_cell, budget - used)
            if local_budget > 1:
                step_size = (hi - lo) / grid_per_dim
                simplex_local_search(center, max_evals=local_budget, step_size=step_size)
        
        # Phase 3: Use any remaining budget for random exploration in promising regions
        while used < budget and time.time() < deadline:
            # Pick a random point, biased toward existing good solutions
            if all_X and np.random.rand() < 0.7:
                # Perturb a random existing point
                base_idx = np.random.randint(len(all_X))
                x = all_X[base_idx].copy()
                noise = 0.05 * (hi - lo) * np.random.randn(dim)
                x += noise
                x = np.clip(x, lo, hi)
            else:
                x = lo + (hi - lo) * np.random.random(dim)
            evaluate(x)
        
        return all_X
