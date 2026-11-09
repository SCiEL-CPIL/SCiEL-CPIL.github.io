"""Multi-objective optimizer for ZDT6."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        budget = self.budget
        
        # ZDT6 structure: g(x) = 1 + sum(x[1:])
        # Pareto front: x[1:] = (g-1)/(n-1) for each component, g in [1, n]
        # x[0] controls f1
        
        def zdt6_front_point(t, rng):
            """Generate a point near the Pareto front for parameter t in [0,1]."""
            g_target = 1.0 + t * (n - 1)  # g ranges from 1 to n
            x = np.zeros(n)
            # Set x[1:] to achieve the target g
            x[1:] = (g_target - 1.0) / (n - 1)
            # x[0] is free; set it to a value that gives good f1
            # f1 = 1 + 10*(n-1) + sum_{i=1}^{n-1} (xi-0.5)^2*(1+10*sin(6*pi*xi+pi*f1))
            # For Pareto optimality, we want x[0] to minimize f1 contribution
            # x[0] = 0 gives (0-0.5)^2 * (1+10*sin(pi*f1)) which varies
            # Let's try a few values
            x[0] = rng.random()
            return np.clip(x, lo, hi)
        
        rng = np.random.default_rng(0)
        
        # Initial population: structured + random
        pop_size = min(50, budget)
        X = np.zeros((pop_size, n))
        
        # Structured points along the front
        n_structured = pop_size // 2
        for i in range(n_structured):
            t = (i + 0.5) / n_structured
            X[i] = zdt6_front_point(t, rng)
        
        # Random points
        for i in range(n_structured, pop_size):
            X[i] = lo + (hi - lo) * rng.random(n)
        
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        def non_dominated_sort(F):
            """Return list of fronts, each a list of indices."""
            m = len(F)
            domination_count = np.zeros(m, dtype=int)
            dominated_set = [[] for _ in range(m)]
            fronts = []
            
            for i in range(m):
                for j in range(m):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        dominated_set[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        domination_count[i] += 1
            
            current_front = [i for i in range(m) if domination_count[i] == 0]
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in dominated_set[i]:
                        domination_count[j] -= 1
                        if domination_count[j] == 0:
                            next_front.append(j)
                current_front = next_front
            
            # Add any remaining (shouldn't happen)
            assigned = set()
            for front in fronts:
                for idx in front:
                    assigned.add(idx)
            remaining = [i for i in range(m) if i not in assigned]
            if remaining:
                fronts.append(remaining)
            
            return fronts
        
        def crowding_distance(F, indices):
            """Compute crowding distance for given indices."""
            if len(indices) <= 2:
                return np.full(len(indices), np.inf)
            
            m = len(indices)
            cd = np.zeros(m)
            F_sub = F[indices]
            
            for k in range(self.n_obj):
                order = np.argsort(F_sub[:, k])
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                f_range = F_sub[order[-1], k] - F_sub[order[0], k]
                if f_range > 1e-15:
                    for i in range(1, m - 1):
                        cd[order[i]] += (F_sub[order[i+1], k] - F_sub[order[i-1], k]) / f_range
            return cd
        
        while used < budget:
            if time.time() > deadline:
                break
            
            fronts = non_dominated_sort(F)
            
            # Build next generation
            next_X = []
            next_F = []
            next_indices = []
            
            # Fill with fronts
            for front in fronts:
                if len(next_X) + len(front) <= pop_size:
                    next_X.extend(X[front].tolist())
                    next_F.extend(F[front].tolist())
                    next_indices.extend(front)
                else:
                    # Need to select from this front using crowding distance
                    remaining = pop_size - len(next_X)
                    cd = crowding_distance(F, front)
                    order = np.argsort(cd)[::-1]
                    selected = [front[i] for i in order[:remaining]]
                    next_X.extend(X[selected].tolist())
                    next_F.extend(F[selected].tolist())
                    next_indices.extend(selected)
                    break
            
            X = np.array(next_X)
            F = np.array(next_F)
            
            # Create offspring
            n_offspring = pop_size
            offspring_X = np.zeros((n_offspring, n))
            
            for i in range(n_offspring):
                if time.time() > deadline:
                    break
                
                # Tournament selection
                p1_idx, p2_idx = self._tournament_select(F, fronts, rng)
                
                # Crossover (SBX-like)
                child = self._crossover(X[p1_idx], X[p2_idx], lo, hi, rng)
                
                # Mutation
                child = self._mutate(child, lo, hi, rng)
                
                offspring_X[i] = child
            
            # Evaluate offspring
            offspring_F = np.array([problem(x) for x in offspring_X])
            used += n_offspring
            
            # Combine parent and offspring
            all_X = np.vstack([X, offspring_X])
            all_F = np.vstack([F, offspring_F])
            
            # Select next generation
            all_fronts = non_dominated_sort(all_F)
            next_X = []
            next_F = []
            
            for front in all_fronts:
                if len(next_X) + len(front) <= pop_size:
                    next_X.extend(all_X[front].tolist())
                    next_F.extend(all_F[front].tolist())
                else:
                    remaining = pop_size - len(next_X)
                    cd = crowding_distance(all_F, front)
                    order = np.argsort(cd)[::-1]
                    selected = [front[i] for i in order[:remaining]]
                    next_X.extend(all_X[selected].tolist())
                    next_F.extend(all_F[selected].tolist())
                    break
            
            X = np.array(next_X)
            F = np.array(next_F)
        
        return X
        
    def _tournament_select(self, F, fronts, rng, k=2):
        """Tournament selection based on non-dominated rank."""
        # Assign ranks
        ranks = np.zeros(len(F), dtype=int)
        for rank, front in enumerate(fronts):
            for idx in front:
                ranks[idx] = rank
        
        candidates = rng.integers(0, len(F), size=k)
        best = candidates[0]
        for c in candidates[1:]:
            if ranks[c] < ranks[best]:
                best = c
        return best, best  # Return same for simplicity, or pick two
        
    def _crossover(self, p1, p2, lo, hi, rng):
        """SBX crossover."""
        child = np.zeros_like(p1)
        for i in range(len(p1)):
            if rng.random() < 0.5:
                child[i] = p1[i]
            else:
                child[i] = p2[i]
        return np.clip(child, lo, hi)
    
    def _mutate(self, x, lo, hi, rng):
        """Polynomial-like mutation."""
        x = x.copy()
        for i in range(len(x)):
            if rng.random() < 0.1:  # mutation probability
                span = hi[i] - lo[i]
                x[i] = x[i] + 0.1 * span * rng.standard_normal()
        return np.clip(x, lo, hi)
# EVOLVE-BLOCK-END
