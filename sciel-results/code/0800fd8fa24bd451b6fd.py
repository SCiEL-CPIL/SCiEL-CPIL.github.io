# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """NSGA-II for multi-objective optimization (DTLZ4)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # NSGA-II parameters
        pop_size = 40
        eta_crossover = 20.0  # High eta for SBX to keep parents close
        eta_mutation = 20.0   # High eta for PM to make small changes
        
        # Initial population
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def fast_non_dominated_sort(F):
            """Returns list of fronts, each front is a list of indices."""
            n = len(F)
            S = [set() for _ in range(n)]  # S[i] = set of indices dominated by i
            n_d = [0] * n  # n_d[i] = number of individuals dominating i
            fronts = []
            
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    # Check if i dominates j
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].add(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_d[i] += 1
            
            current_front = []
            for i in range(n):
                if n_d[i] == 0:
                    current_front.append(i)
            
            k = 0
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_d[j] -= 1
                        if n_d[j] == 0:
                            next_front.append(j)
                current_front = next_front
                k += 1
            return fronts

        def crowding_distance(F, indices):
            """Calculates crowding distance for individuals in 'indices'."""
            if not indices:
                return np.zeros(0)
            n = len(indices)
            dist = np.zeros(n)
            if n <= 2:
                return dist
            
            m = self.n_obj
            for m_idx in range(m):
                # Sort indices by objective m_idx
                sorted_indices = sorted(indices, key=lambda idx: F[idx, m_idx])
                dist_sorted = np.zeros(n)
                
                # Map sorted positions to original indices in the subset
                # Actually, we just compute distance for the subset
                obj_vals = F[sorted_indices, m_idx]
                if obj_vals[-1] - obj_vals[0] > 1e-10:
                    dist_sorted[0] = float('inf')
                    dist_sorted[-1] = float('inf')
                    for k in range(1, n - 1):
                        dist_sorted[k] = (obj_vals[k+1] - obj_vals[k-1]) / (obj_vals[-1] - obj_vals[0])
                else:
                    dist_sorted[:] = float('inf')
                
                # Assign back to dist array corresponding to sorted_indices
                for pos, idx in enumerate(sorted_indices):
                    dist[indices.index(idx)] += dist_sorted[pos]
            
            return dist

        def sbx_crossover(x1, x2):
            """Simulated Binary Crossover."""
            if rng.random() < 0.9: # Crossover probability
                child1 = np.empty(self.dim)
                child2 = np.empty(self.dim)
                for i in range(self.dim):
                    if rng.random() < 0.5:
                        if x1[i] < x2[i]:
                            c1, c2 = self._sbx_single(x1[i], x2[i], lo[i], hi[i], eta_crossover)
                        else:
                            c1, c2 = self._sbx_single(x2[i], x1[i], lo[i], hi[i], eta_crossover)
                    else:
                        c1, c2 = x1[i], x2[i]
                    child1[i] = c1
                    child2[i] = c2
                return child1, child2
            else:
                return x1.copy(), x2.copy()

        def _sbx_single(x1, x2, lo_b, hi_b, eta):
            """Single gene SBX."""
            if abs(x1 - x2) < 1e-14:
                return x1, x2
            
            u = rng.random()
            if u <= 0.5:
                beta = (2 * u) ** (1.0 / (eta + 1))
            else:
                beta = (1 / (2 * (1 - u))) ** (1.0 / (eta + 1))
            
            c1 = 0.5 * ((1 + beta) * x1 + (1 - beta) * x2)
            c2 = 0.5 * ((1 - beta) * x1 + (1 + beta) * x2)
            
            # Clip to bounds
            c1 = np.clip(c1, lo_b, hi_b)
            c2 = np.clip(c2, lo_b, hi_b)
            
            # If clipping causes issues, swap or adjust
            if c1 == c2:
                return x1, x2
            
            return c1, c2

        def polynomial_mutation(x, lo_b, hi_b, eta):
            """Polynomial Mutation."""
            y = x.copy()
            for i in range(self.dim):
                if rng.random() < 0.9 / self.dim: # Mutation probability
                    u = rng.random()
                    if u < 0.5:
                        delta1 = (2 * u) ** (1.0 / (eta + 1)) - 1
                    else:
                        delta1 = 1 - (2 * (1 - u)) ** (1.0 / (eta + 1))
                    
                    yl = (x[i] - lo_b[i]) / (hi_b[i] - lo_b[i])
                    y = x[i] + delta1 * (hi_b[i] - lo_b[i]) * (
                        0.5 * (1 + np.sqrt(1 + (eta + 1) * yl)) ** (-(eta + 1)) +
                        0.5 * (1 - np.sqrt(1 + (eta + 1) * (1 - yl))) ** (-(eta + 1))
                    )
                    y = np.clip(y, lo_b[i], hi_b[i])
            return y

        # Main loop
        while used < self.budget:
            # Create offspring population
            offspring_X = np.empty((pop_size, self.dim))
            
            for i in range(pop_size):
                # Tournament selection for parent 1
                idx1 = self._tournament_select(F, pop_size, rng)
                # Tournament selection for parent 2
                idx2 = self._tournament_select(F, pop_size, rng)
                
                p1 = X[idx1]
                p2 = X[idx2]
                
                c1, c2 = sbx_crossover(p1, p2)
                c1 = polynomial_mutation(c1, lo, hi, eta_mutation)
                c2 = polynomial_mutation(c2, lo, hi, eta_mutation)
                
                if used + 2 <= self.budget:
                    offspring_X[i] = c1
                    f1 = problem(c1)
                    used += 1
                    # Store temporarily, we'll combine later
                    # For simplicity in this loop structure, let's store F_offspring
                    if not hasattr(self, '_f_off'):
                        self._f_off = []
                    self._f_off.append(f1)
                else:
                    break
            
            # If we broke early, adjust
            if used < self.budget:
                 # Fill remaining with copies or just stop
                 pass
            
            # We need to be careful with evaluation count.
            # Let's restructure to evaluate exactly pop_size offspring per generation.
            
            # Restart logic for clarity and strict budget adherence
            break # The above loop was messy. Let's implement a cleaner generation loop.

        # Re-implementing the main loop properly to ensure exact budget usage
        # Reset used count? No, we already used initial pop.
        # Let's do a clean implementation below.
        
        # --- Clean Implementation ---
        # Reset state for clean loop
        # (The code above is discarded, replacing with clean logic)
        
        # We'll just use the initial X and F.
        # Loop for generations
        while used + pop_size <= self.budget:
            offspring_X = np.empty((pop_size, self.dim))
            offspring_F = np.empty((pop_size, self.n_obj))
            
            for i in range(pop_size):
                # Selection
                idx1 = self._tournament_select(F, pop_size, rng)
                idx2 = self._tournament_select(F, pop_size, rng)
                
                p1 = X[idx1]
                p2 = X[idx2]
                
                c1, c2 = sbx_crossover(p1, p2)
                c1 = polynomial_mutation(c1, lo, hi, eta_mutation)
                
                # Evaluate only c1 to save budget? No, standard NSGA-II evaluates both.
                # But budget is tight. 200 total. 40 initial. 160 left.
                # 4 generations * 40 offspring = 160.
                # If we generate 2 offspring per pair, we need 20 pairs.
                # Let's generate 1 offspring per iteration in the loop to control budget strictly.
                
                # Actually, standard NSGA-II generates N offspring.
                # We will generate 1 offspring per loop iteration, selecting 2 parents.
                # This uses 1 eval per iteration.
                
                offspring_X[i] = c1
                offspring_F[i] = problem(c1)
                used += 1
            
            # Combine parent and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            
            # Environmental Selection
            fronts = fast_non_dominated_sort(combined_F)
            
            new_X = []
            new_F = []
            for front in fronts:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                else:
                    # Need to select from this front based on crowding distance
                    dist = crowding_distance(combined_F, front)
                    # Sort front indices by distance descending
                    sorted_front = sorted(front, key=lambda idx: dist[front.index(idx)], reverse=True)
                    needed = pop_size - len(new_X)
                    for idx in sorted_front[:needed]:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                    break
            
            X = np.array(new_X)
            F = np.array(new_F)
            
            if used >= self.budget:
                break

        return X

    def _tournament_select(self, F, pop_size, rng):
        """Binary tournament selection."""
        i1 = rng.integers(pop_size)
        i2 = rng.integers(pop_size)
        
        # Simple dominance check
        if np.all(F[i1] <= F[i2]) and np.any(F[i1] < F[i2]):
            return i1
        elif np.all(F[i2] <= F[i1]) and np.any(F[i2] < F[i1]):
            return i2
        
        # If non-dominated, pick randomly (or use crowding, but simple random is okay for small pops)
        return i1 if rng.random() < 0.5 else i2
# EVOLVE-BLOCK-END
