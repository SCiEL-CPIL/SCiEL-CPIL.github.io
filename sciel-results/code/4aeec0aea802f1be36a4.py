import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s kill
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        max_evals = self.budget
        
        # NSGA-II parameters
        # Budget 200. 
        # Let's use a population size of 50.
        # Generations: 200 / 50 = 4 generations.
        # This allows for 4 rounds of selection, crossover, mutation, and ranking.
        
        pop_size = 50
        n_gen = max_evals // pop_size
        
        # Initialize population
        X = lo + (hi - lo) * np.random.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        all_X = X.copy()
        all_F = F.copy()
        
        for gen in range(n_gen):
            if time.time() > deadline:
                break
            
            # Combine parents and offspring
            # We will generate offspring of size pop_size
            # Total population size 2*pop_size for ranking
            
            # Selection: Tournament selection based on non-dominated sorting and crowding distance
            # For simplicity and speed, we can just do random selection from the current non-dominated front 
            # or use a simple tournament. Given the small budget, a simple random selection from the 
            # current best non-dominated set is often effective for DTLZ2.
            
            # Let's implement a simple non-dominated sort to get the front.
            def dominates(a, b):
                return np.all(a <= b) and np.any(a < b)
            
            # Get non-dominated front from all_F
            front_indices = []
            for i in range(len(all_F)):
                is_dominated = False
                for j in range(len(all_F)):
                    if i != j and dominates(all_F[j], all_F[i]):
                        is_dominated = True
                        break
                if not is_dominated:
                    front_indices.append(i)
            
            if len(front_indices) == 0:
                front_indices = list(range(len(all_F)))
            
            # Select parents from the front
            # If front is small, sample with replacement. Otherwise, tournament.
            # Let's just sample uniformly from the front for simplicity and diversity.
            parent_indices = np.random.choice(front_indices, size=pop_size, replace=True)
            parents_X = all_X[parent_indices]
            parents_F = all_F[parent_indices]
            
            # Generate offspring
            # Simple mutation: Gaussian perturbation
            # Since DTLZ2 has a known structure, we can also bias towards the Pareto front region
            # if we knew it, but as a black box, we just mutate.
            # However, we can use a "smart" initialization for the first generation if we detect structure?
            # No, stick to black box.
            
            # Crossover: Blend crossover
            offspring_X = np.zeros_like(parents_X)
            for i in range(0, pop_size, 2):
                if i + 1 >= pop_size:
                    offspring_X[i] = parents_X[i]
                else:
                    alpha = np.random.random(self.dim)
                    offspring_X[i] = alpha * parents_X[i] + (1 - alpha) * parents_X[i+1]
                    offspring_X[i+1] = (1 - alpha) * parents_X[i] + alpha * parents_X[i+1]
            
            # Mutation
            # Standard deviation: 1/sqrt(dim) is a common choice, scaled by bounds
            # We need to ensure we stay within bounds or clip.
            # Let's use a small mutation rate and step size.
            # Step size: 0.1 * (hi - lo)
            mutation_mask = np.random.random((pop_size, self.dim)) < 0.2
            mutation_step = 0.1 * (hi - lo)
            mutation_values = np.random.normal(0, 1, (pop_size, self.dim)) * mutation_step
            offspring_X[mutation_mask] += mutation_values[mutation_mask]
            
            # Clip to bounds
            offspring_X = np.clip(offspring_X, lo, hi)
            
            # Evaluate offspring
            offspring_F = np.array([problem(x) for x in offspring_X])
            used += pop_size
            
            # Combine and rank
            combined_X = np.vstack([all_X, offspring_X])
            combined_F = np.vstack([all_F, offspring_F])
            
            # Non-dominated sorting
            # We want to keep the best pop_size points.
            # Simple approach: Find the first non-dominated front. If it's <= pop_size, keep it.
            # If > pop_size, we need to truncate. 
            # For DTLZ2, the front is smooth, so we can just keep the first pop_size non-dominated points.
            # To ensure diversity, we should ideally use crowding distance, but let's try a simpler 
            # "random from front" or "grid" approach for truncation if needed.
            
            # Let's just find the non-dominated points in the combined set.
            new_front_indices = []
            for i in range(len(combined_F)):
                is_dominated = False
                for j in range(len(combined_F)):
                    if i != j and dominates(combined_F[j], combined_F[i]):
                        is_dominated = True
                        break
                if not is_dominated:
                    new_front_indices.append(i)
            
            # If we have more than pop_size non-dominated points, we need to select pop_size of them.
            # A simple heuristic: select them uniformly at random to maintain diversity.
            if len(new_front_indices) > pop_size:
                selected_indices = np.random.choice(new_front_indices, size=pop_size, replace=False)
            else:
                # If we have fewer, we need to fill with dominated points? 
                # Or just keep the front and add best dominated?
                # Usually, we keep the front. If it's smaller, we can add points from the next front.
                # For simplicity, let's just keep the front and pad with random points from the combined set 
                # if necessary, though this might degrade quality.
                # Better: Keep all non-dominated, and if we need more, pick from the rest.
                remaining_indices = [i for i in range(len(combined_F)) if i not in new_front_indices]
                if len(new_front_indices) < pop_size:
                    need = pop_size - len(new_front_indices)
                    extra_indices = np.random.choice(remaining_indices, size=need, replace=False)
                    selected_indices = np.concatenate([new_front_indices, extra_indices])
                else:
                    selected_indices = new_front_indices
            
            # Update population
            all_X = combined_X[selected_indices]
            all_F = combined_F[selected_indices]
            
            # Check time
            if time.time() > deadline:
                break
        
        # Final evaluation of the best points? 
        # The problem statement says "Every returned objective vector is recorded".
        # So we just need to make sure we have called problem() for the points we want recorded.
        # The loop above has already called problem() for all points in all_X and all_F?
        # No, all_F contains the results from previous evaluations.
        # The evaluator records *every* call to problem().
        # So the history of calls is what matters.
        # We have made `used` calls.
        # We should ensure we don't exceed the budget.
        
        # The current implementation re-evaluates offspring, which is correct.
        # The points in all_F are the results of those evaluations.
        # The evaluator records them.
        
        # We don't need to return anything specific, but we must not crash.
        return all_F
