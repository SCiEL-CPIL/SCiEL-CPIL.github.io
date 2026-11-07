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
        used = 0
        
        # We will use a simple NSGA-II-like approach
        # Population size chosen to be small enough to fit in budget
        # but large enough to maintain diversity.
        # Budget 200. If pop_size=40, we can do 5 generations.
        # If pop_size=50, we can do 4 generations.
        pop_size = 50
        
        # Initialize population randomly
        X = lo + (hi - lo) * np.random.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used += pop_size
        
        all_F = F.copy()
        
        # Elitism: Keep the best points found so far
        # We will maintain a list of all non-dominated points or just all points
        # Since the score is based on hypervolume of ALL recorded vectors,
        # we should record every evaluation.
        
        def record(x, f):
            nonlocal used
            # The problem call is already made, so we just increment used
            # but in this structure, we call problem inside the loop.
            # Let's refactor to track used correctly.
            pass

        # Let's restructure to track evaluations strictly.
        # The previous block already did pop_size evals.
        # We need to be careful with the 'used' counter.
        
        # Actually, the previous block `F = np.array([problem(x) for x in X])` 
        # consumed `pop_size` evaluations.
        
        # List of all points evaluated (X, F)
        all_X = X.copy()
        all_F = F.copy()
        
        # Simple NSGA-II loop
        # We have max_evals - used remaining.
        # Each generation requires pop_size evaluations.
        # generations = (max_evals - used) // pop_size
        
        while used < max_evals:
            if time.time() > deadline:
                break
            
            # 1. Combine parents and offspring (if we had offspring, but here we generate new)
            # Since we don't have a complex offspring mechanism yet, let's do:
            # - Select parents from current population
            # - Create offspring
            # - Evaluate offspring
            # - Select next population from parents + offspring
            
            # To keep it simple and effective within the budget:
            # We will maintain a population of size `pop_size`.
            # At each step, we generate `pop_size` new candidates, evaluate them,
            # and then select the best `pop_size` from the combined set of 
            # previous population and new candidates based on non-domination.
            
            # However, this doubles the number of points we need to sort.
            # With pop_size=50, combined=100. Sorting 100 points in 3D is fast.
            
            # Generate offspring
            # Use a simple mutation/crossover
            # For simplicity and speed, let's use a "random restart with local bias" 
            # or simple crossover + mutation.
            
            # Let's use a simple approach: 
            # Pick two parents, crossover, mutate.
            # Or even simpler: since DTLZ2 is separable, random search with 
            # elitism works surprisingly well if we keep the best points.
            
            # Let's implement a simple (mu+lambda) EA with mu=pop_size, lambda=pop_size
            # Selection: Non-dominated sorting + crowding distance (approximated)
            
            # To save time and complexity, let's just keep the best `pop_size` 
            # points from the union of the current population and the new 
            # offspring, based on a simple hypervolume contribution or 
            # non-domination.
            
            # Given the time constraint and the fact that DTLZ2 is well-behaved,
            # a simple "keep best N by non-domination" is sufficient.
            
            # Create new population candidates
            new_X = np.zeros((pop_size, self.dim))
            
            for i in range(pop_size):
                # Crossover: Pick two parents
                p1_idx = np.random.randint(0, pop_size)
                p2_idx = np.random.randint(0, pop_size)
                
                # Simple uniform crossover
                mask = np.random.random(self.dim) > 0.5
                child = np.where(mask, X[p1_idx], X[p2_idx])
                
                # Mutation: Gaussian noise
                # Scale noise based on bounds
                sigma = 0.1 * (hi - lo)
                child += sigma * np.random.standard_normal(self.dim)
                child = np.clip(child, lo, hi)
                
                new_X[i] = child
            
            # Evaluate new population
            new_F = np.array([problem(x) for x in new_X])
            used += pop_size
            
            # Combine
            combined_X = np.vstack([X, new_X])
            combined_F = np.vstack([F, new_F])
            
            # Update global records
            all_X = np.vstack([all_X, new_X])
            all_F = np.vstack([all_F, new_F])
            
            # Selection: Choose the best pop_size from combined
            # We need a fast selection method.
            # Non-dominated sorting is O(N^2). N=100. 10,000 ops. Very fast.
            
            # Function to check if a dominates b
            # a dominates b if a_i <= b_i for all i and a_j < b_j for some j
            
            # Let's implement a simple non-dominated sort to get the first front.
            # If the first front has more than pop_size, we need to trim.
            # If less, we add from the second front.
            
            # For simplicity and speed, let's use a greedy selection based on 
            # non-domination and distance to the ideal point (0,0,0).
            
            # Actually, for DTLZ2, the Pareto front is a quarter sphere.
            # Points closer to the origin (in objective space) are generally better 
            # in terms of HV contribution if they are non-dominated.
            
            # Let's just keep the points that are non-dominated in the combined set.
            # If there are too many, we keep the ones with the smallest sum of objectives.
            
            dominated = np.zeros(len(combined_F), dtype=bool)
            for i in range(len(combined_F)):
                if dominated[i]:
                    continue
                for j in range(len(combined_F)):
                    if i == j or dominated[j]:
                        continue
                    # Check if i dominates j
                    if np.all(combined_F[i] <= combined_F[j]) and np.any(combined_F[i] < combined_F[j]):
                        dominated[j] = True
            
            # Get non-dominated indices
            non_dom_idx = np.where(~dominated)[0]
            
            # If we have more non-dominated points than pop_size, we need to select
            # If fewer, we fill with dominated points (closest to non-dominated)
            
            if len(non_dom_idx) > pop_size:
                # Select best pop_size from non-dominated
                # Sort by sum of objectives (proxy for distance to origin)
                sums = np.sum(combined_F[non_dom_idx], axis=1)
                # We want the smallest sums
                # But we also want diversity. 
                # For DTLZ2, the front is smooth. 
                # Let's just take the ones with smallest sums.
                # This might cluster them. 
                # Better: Randomly sample from non-dominated? No, we want the "best" ones.
                # Actually, for HV, we want points that spread out.
                # Let's use a simple grid or just keep the first pop_size found?
                # Let's sort by the angle in the first two objectives to ensure spread.
                
                # Calculate angles for f1, f2
                # f1 = sin, f2 = cos. angle = atan2(f1, f2)
                # This assumes the standard DTLZ2 front.
                # To be black-box, let's just use random selection from the non-dominated set?
                # No, that's bad.
                
                # Let's use a simple "max-min distance" selection on the non-dominated set.
                # Or simply: sort by f1 and take evenly spaced ones?
                
                # Given the time, let's just take the non-dominated points with the 
                # smallest norm (distance to 0). This biases towards the "center" of the front.
                # This is not ideal for HV which wants spread.
                
                # Alternative: Keep the existing population X if it's good, and 
                # only replace the worst ones.
                
                # Let's simplify: If we have > pop_size non-dominated, we just 
                # randomly select pop_size of them. This maintains diversity.
                selected_idx = np.random.choice(non_dom_idx, pop_size, replace=False)
            elif len(non_dom_idx) < pop_size:
                # Need to fill with dominated points
                # Pick the dominated points that are "closest" to the non-dominated front
                # Approximation: smallest sum of objectives among dominated
                dom_idx = np.where(dominated)[0]
                if len(dom_idx) > 0:
                    sums = np.sum(combined_F[dom_idx], axis=1)
                    # Sort dominated by sum
                    sorted_dom = dom_idx[np.argsort(sums)]
                    needed = pop_size - len(non_dom_idx)
                    selected_idx = np.concatenate([non_dom_idx, sorted_dom[:needed]])
                else:
                    # Should not happen
                    selected_idx = np.arange(pop_size)
            else:
                selected_idx = non_dom_idx
            
            # Update population
            X = combined_X[selected_idx]
            F = combined_F[selected_idx]
            
            # Check time
            if time.time() > deadline:
                break
            
        return np.array(all_X)
