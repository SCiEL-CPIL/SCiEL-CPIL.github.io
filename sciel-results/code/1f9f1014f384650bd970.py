import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.population_size = 50  # Keep small to allow many generations within budget
        self.crowding_distance = None

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Initial population: Uniform random in bounds
        # We use the seeded RNG from the evaluator
        pop = np.random.uniform(lo, hi, size=(self.population_size, self.dim))
        F = np.array([problem(x) for x in pop])
        
        eval_count = self.population_size
        all_points = [pop]
        all_F = [F]
        
        # NSGA-II Loop
        while eval_count < self.budget:
            if time.time() > deadline:
                break
                
            # Combine parents and offspring (we generate offspring equal to pop size)
            # Actually, standard NSGA-II:
            # 1. Evaluate parents
            # 2. Create offspring
            # 3. Evaluate offspring
            # 4. Combine parents + offspring
            # 5. Non-dominated sort
            # 6. Select next generation based on ranks and crowding distance
            
            # Generate offspring
            offspring = self._generate_offspring(pop, lo, hi)
            
            # Check budget
            if eval_count + self.population_size > self.budget:
                # Only evaluate as many as we can
                n_to_eval = self.budget - eval_count
                offspring = offspring[:n_to_eval]
            
            F_off = np.array([problem(x) for x in offspring])
            eval_count += len(offspring)
            
            all_points.append(offspring)
            all_F.append(F_off)
            
            # Combine
            X_combined = np.vstack([pop, offspring])
            F_combined = np.vstack([F, F_off])
            
            # Non-dominated sort
            ranks = self._non_dominated_sort(F_combined)
            
            # Calculate crowding distance for each rank
            # Select next generation
            new_pop = []
            for i in range(len(ranks)):
                idx_in_rank = np.where(ranks == i)[0]
                if len(new_pop) + len(idx_in_rank) <= self.population_size:
                    new_pop.extend(idx_in_rank)
                else:
                    # Need to select some from this rank based on crowding distance
                    n_needed = self.population_size - len(new_pop)
                    # Calculate crowding distance for these points
                    F_rank = F_combined[idx_in_rank]
                    cd = self._crowding_distance(F_rank)
                    # Sort by crowding distance descending
                    sorted_idx = np.argsort(-cd)
                    selected_in_rank = idx_in_rank[sorted_idx[:n_needed]]
                    new_pop.extend(selected_in_rank)
                    break
            
            pop = X_combined[new_pop]
            F = F_combined[new_pop]
            
            # Stop if we've hit the budget exactly or deadline
            if eval_count >= self.budget:
                break

        return pop

    def _generate_offspring(self, pop, lo, hi):
        """Simple SBX crossover and Polynomial mutation"""
        n_pop = len(pop)
        offspring = np.zeros_like(pop)
        
        # Crossover rate 0.9, Mutation rate 1/dim
        # For simplicity and speed, we use a simple blend or just mutation
        # Given 22 dims and 200 budget, complex operators might be overkill.
        # Let's use a simple random restart + local perturbation strategy (Memetic-ish)
        # Or just simple DE-like mutation.
        
        # Let's use Differential Evolution mutation: x_new = x_i + F * (x_r1 - x_r2)
        F_mut = 0.5
        for i in range(n_pop):
            r1, r2 = np.random.choice(n_pop, 2, replace=False)
            # Ensure r1, r2 != i ideally, but with small pop it's ok
            if r1 == i: r1 = (r1 + 1) % n_pop
            if r2 == i: r2 = (r2 + 1) % n_pop
            if r1 == r2: r2 = (r2 + 1) % n_pop
            
            mutant = pop[i] + F_mut * (pop[r1] - pop[r2])
            
            # Clip to bounds
            mutant = np.clip(mutant, lo, hi)
            
            # Polynomial Mutation
            # Simple Gaussian mutation scaled by bounds
            # This is robust
            sigma = 0.1 * (hi - lo)
            mutant = mutant + np.random.normal(0, sigma)
            mutant = np.clip(mutant, lo, hi)
            
            offspring[i] = mutant
            
        return offspring

    def _non_dominated_sort(self, F):
        """Returns an array of ranks for each point"""
        n = len(F)
        ranks = np.zeros(n, dtype=int)
        S = [[] for _ in range(n)] # S[i] = indices dominated by i
        n_dominated = np.zeros(n, dtype=int) # n_dominated[i] = count of points i dominates
        
        for i in range(n):
            for j in range(i + 1, n):
                if self._dominates(F[i], F[j]):
                    S[i].append(j)
                    n_dominated[j] += 1
                elif self._dominates(F[j], F[i]):
                    S[j].append(i)
                    n_dominated[i] += 1
        
        front = 0
        current_front = []
        for i in range(n):
            if n_dominated[i] == 0:
                ranks[i] = 0
                current_front.append(i)
        
        while current_front:
            next_front = []
            for i in current_front:
                for j in S[i]:
                    n_dominated[j] -= 1
                    if n_dominated[j] == 0:
                        ranks[j] = 1 if front == 0 else front + 1 # Wait, rank is 0, 1, 2...
                        # Actually rank 0 is first front.
                        ranks[j] = front + 1
                        next_front.append(j)
            front += 1
            current_front = next_front
            
        return ranks

    def _dominates(self, a, b):
        """Returns True if a dominates b (all <= and at least one <)"""
        if np.any(a > b):
            return False
        if np.all(a == b):
            return False
        return True

    def _crowding_distance(self, F):
        """Calculates crowding distance for a set of points in the same front"""
        n = len(F)
        if n <= 2:
            return np.full(n, np.inf)
            
        cd = np.zeros(n)
        for k in range(self.n_obj):
            # Sort by objective k
            idx = np.argsort(F[:, k])
            cd[idx[0]] = np.inf
            cd[idx[-1]] = np.inf
            f_range = F[idx[-1], k] - F[idx[0], k]
            if f_range == 0:
                continue
            for i in range(1, n - 1):
                cd[idx[i]] += (F[idx[i+1], k] - F[idx[i-1], k]) / f_range
                
        return cd
