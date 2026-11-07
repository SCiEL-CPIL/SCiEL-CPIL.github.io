import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        N = 20  # Population size
        self.evals = 0
        
        # Helper to check budget
        def can_eval():
            return self.evals < self.budget
        
        # Initial population
        X = lo + (hi - lo) * np.random.rand(N, self.dim)
        F = np.array([problem(x) for x in X])
        self.evals += N
        
        # NSGA-II loop
        while self.evals < self.budget:
            # Generate offspring
            Q = np.zeros_like(X)
            for i in range(N):
                # Crossover
                if np.random.rand() < 0.9:
                    p1_idx = np.random.randint(N)
                    p2_idx = np.random.randint(N)
                    if p1_idx == p2_idx:
                        p2_idx = (p2_idx + 1) % N
                    c1, c2 = self.sbx_crossover(X[p1_idx], X[p2_idx], lo, hi)
                    # Mutation
                    c1 = self.polynomial_mutation(c1, lo, hi)
                    c2 = self.polynomial_mutation(c2, lo, hi)
                    Q[i] = c1
                    if self.evals + 1 < self.budget:
                         # We need to evaluate both? 
                         # Actually, we generate N offspring. 
                         # If we evaluate 2 per pair, we use 2N evals per generation.
                         # But we only have budget for N offspring per generation if we count 1 eval per offspring.
                         # Wait, standard NSGA-II generates N offspring from N parents.
                         # If we do 1 crossover producing 2 children, we need N/2 crossovers.
                         # Let's simplify: Generate N offspring by selecting 2 parents per offspring?
                         # Or just use 2-point crossover for N/2 pairs to get N children.
                         # Let's stick to generating N children.
                         # If we do N/2 crossovers, we get N children.
                         # We evaluate all N children.
                         # So 1 generation costs N evaluations.
                         pass
                
            # Let's restructure the offspring generation to be cleaner and strictly N evals.
            Q = np.zeros_like(X)
            for i in range(N):
                p1_idx = np.random.randint(N)
                p2_idx = np.random.randint(N)
                if p1_idx == p2_idx:
                    p2_idx = (p2_idx + 1) % N
                c1, c2 = self.sbx_crossover(X[p1_idx], X[p2_idx], lo, hi)
                # Use c1 as the offspring
                c1 = self.polynomial_mutation(c1, lo, hi)
                Q[i] = c1
            
            F_q = np.array([problem(x) for x in Q])
            self.evals += N
            
            # Combine
            X_all = np.vstack((X, Q))
            F_all = np.vstack((F, F_q))
            
            # Selection
            X, F = self.nsga2_select(X_all, F_all, N)
        
        # Return the non-dominated set from the final population
        # Actually, the task says "Every returned objective vector is recorded". 
        # The evaluator records the return value of __call__? 
        # "The return value of __call__ is ignored." 
        # "Every returned objective vector is recorded" - this likely means the evaluator intercepts calls to `problem`?
        # No, "The evaluator calls ... and calls it once." 
        # "Every returned objective vector is recorded" usually implies the evaluator wraps the `problem` function or checks the state.
        # But the contract says: "Every returned objective vector is recorded and the return value of __call__ is ignored."
        # This implies the evaluator is listening to the `problem` calls? 
        # "problem(x) clips ... and returns 3 objective values."
        # If the evaluator records every call to `problem`, then we just need to make the calls.
        # The return value of `__call__` is ignored, so we can return anything.
        # But to be safe, we should return the final population or non-dominated set.
        
        # Identify non-dominated in final F
        fronts = self.non_dominated_sort(F)
        if fronts:
            nd_indices = fronts[0]
            return X[nd_indices]
        return X

    def sbx_crossover(self, x1, x2, lo, hi):
        """Simulated Binary Crossover"""
        c1 = x1.copy()
        c2 = x2.copy()
        eta = 15.0
        for i in range(self.dim):
            if np.random.rand() < 0.5:
                if abs(x1[i] - x2[i]) > 1e-14:
                    if x1[i] < x2[i]:
                        beta = 1 + (2 * (x1[i] - lo[i]) / (x2[i] - x1[i]))
                        alpha = 2 - beta ** (-(eta + 1))
                        if np.random.rand() <= 1.0 / alpha:
                            beta_q = (0.5 * beta) ** (1.0 / (eta + 1))
                        else:
                            beta_q = (0.5 / (2 - beta)) ** (1.0 / (eta + 1))
                        c1[i] = 0.5 * (x1[i] + x2[i] - beta_q * (x2[i] - x1[i]))
                    else:
                        beta = 1 + (2 * (hi[i] - x2[i]) / (x1[i] - x2[i]))
                        alpha = 2 - beta ** (-(eta + 1))
                        if np.random.rand() <= 1.0 / alpha:
                            beta_q = (0.5 * beta) ** (1.0 / (eta + 1))
                        else:
                            beta_q = (0.5 / (2 - beta)) ** (1.0 / (eta + 1))
                        c2[i] = 0.5 * (x1[i] + x2[i] + beta_q * (x1[i] - x2[i]))
        return c1, c2

    def polynomial_mutation(self, x, lo, hi):
        """Polynomial Mutation"""
        eta = 20.0
        x_mut = x.copy()
        for i in range(self.dim):
            if np.random.rand() < 1.0 / self.dim:
                delta1 = (x_mut[i] - lo[i]) / (hi[i] - lo[i])
                delta2 = (hi[i] - x_mut[i]) / (hi[i] - lo[i])
                rand = np.random.rand()
                if rand < 0.5:
                    xy = 1.0 - delta1
                    mut_pow = 2.0 + (1.0 / (xy * (eta + 1.0)))
                    deltaq = (2.0 * rand + (1.0 - 2.0 * rand) * (1.0 - (1.0 - rand) ** mut_pow)) ** (1.0 / mut_pow) - 1.0
                else:
                    xy = 1.0 - delta2
                    mut_pow = 2.0 + (1.0 / (xy * (eta + 1.0)))
                    deltaq = 1.0 - (2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (1.0 - (1.0 - rand) ** mut_pow)) ** (1.0 / mut_pow)
                x_mut[i] += deltaq * (hi[i] - lo[i])
                x_mut[i] = np.clip(x_mut[i], lo[i], hi[i])
        return x_mut

    def non_dominated_sort(self, F):
        """Returns list of fronts, each front is a list of indices"""
        n = len(F)
        S = [[] for _ in range(n)]  # Dominated sets
        n_dom = np.zeros(n, dtype=int)  # Dominance count
        fronts = []
        current_front = []
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    n_dom[i] += 1
            if n_dom[i] == 0:
                current_front.append(i)
        
        while current_front:
            fronts.append(current_front)
            next_front = []
            for i in current_front:
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        next_front.append(j)
            current_front = next_front
            
        return fronts

    def crowding_distance(self, F, indices):
        """Calculates crowding distance for individuals in a front"""
        n = len(indices)
        if n == 0:
            return np.array([])
        if n <= 2:
            return np.full(n, np.inf)
            
        distances = np.zeros(n)
        F_front = F[indices]
        
        for m in range(self.n_obj):
            # Sort by objective m
            sorted_indices = np.argsort(F_front[:, m])
            distances[sorted_indices[0]] = np.inf
            distances[sorted_indices[-1]] = np.inf
            
            range_val = F_front[sorted_indices[-1], m] - F_front[sorted_indices[0], m]
            if range_val == 0:
                continue
                
            for i in range(1, n - 1):
                dist = (F_front[sorted_indices[i + 1], m] - F_front[sorted_indices[i - 1], m]) / range_val
                distances[sorted_indices[i]] += dist
                
        return distances

    def nsga2_select(self, X_all, F_all, N):
        """Selects N individuals from X_all/F_all using NSGA-II selection"""
        fronts = self.non_dominated_sort(F_all)
        X_new = []
        F_new = []
        
        for front in fronts:
            if len(X_new) + len(front) <= N:
                # Add all
                X_new.extend(X_all[front])
                F_new.extend(F_all[front])
            else:
                # Need to select from this front
                remaining = N - len(X_new)
                distances = self.crowding_distance(F_all, front)
                # Sort indices by distance descending
                sorted_front = np.array(front)[np.argsort(distances)[::-1]]
                X_new.extend(X_all[sorted_front[:remaining]])
                F_new.extend(F_all[sorted_front[:remaining]])
                break
                
        if len(X_new) < N:
            # Should not happen if fronts cover all, but just in case
            # Fill with random
            while len(X_new) < N:
                idx = np.random.randint(len(X_all))
                X_new.append(X_all[idx])
                F_new.append(F_all[idx])
                
        return np.array(X_new), np.array(F_new)
