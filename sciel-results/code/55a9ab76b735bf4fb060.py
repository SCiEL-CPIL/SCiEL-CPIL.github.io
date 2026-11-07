"""Multi-objective optimizer on RE34 using Hypervolume-Indicative Evolutionary Strategy."""
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
        deadline = start_time + 290  # Safety margin for 300s limit
        
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Use a fixed seed for reproducibility within the run, 
        # but the evaluator sets np.random.seed(seed) before calling us.
        # We should respect the global state or use our own rng?
        # The prompt says "evaluator calls np.random.seed(seed)". 
        # Using np.random directly ensures we use the seeded state.
        
        # Initialize population
        pop_size = 50
        if budget < pop_size:
            pop_size = max(10, budget // 2)
            
        X = lo + (hi - lo) * np.random.rand(pop_size, dim)
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Reference point for hypervolume calculation (from task description)
        # The task says reference point is [1705.0, 12.0, 0.3].
        # However, for selection during optimization, we can use a dynamic reference 
        # or just use the known one to guide towards the "interesting" part of the front.
        # Using the actual reference point helps align the search with the scoring metric.
        ref_point = np.array([1705.0, 12.0, 0.3])
        
        # Helper to calculate hypervolume contribution roughly or use a simple indicator
        # For speed, we can use a simplified HV contribution or just use the non-dominated set.
        # Given 3 objectives, we can compute exact HV of the non-dominated front efficiently.
        
        def compute_hv(points, ref):
            """Compute hypervolume for 3 objectives. Points must be non-dominated."""
            if len(points) == 0:
                return 0.0
            # Sort by first objective
            points = sorted(points, key=lambda p: p[0])
            hv = 0.0
            # For 3D, we can iterate and compute the volume of boxes
            # This is O(n^2) or O(n log n) depending on implementation.
            # For n~50, O(n^2) is fine.
            
            # Standard 3D HV algorithm:
            # Sort by f0.
            # For each point i, find the next point j that is "higher" in f1 or f2?
            # Actually, simpler:
            # The HV is the sum of volumes of boxes defined by each point and the reference.
            # But they overlap.
            # A common efficient way for small n:
            # 1. Sort by f0 ascending.
            # 2. Keep track of the "best" (min) f1 and f2 seen so far? No, that's for 2D.
            
            # Let's use a simple O(n^2) check for dominance to ensure points are non-dominated first,
            # then use a simple union of boxes or a specific 3D algorithm.
            
            # Since n is small (<=50), we can use a simple approach:
            # Sort by f0.
            # Iterate i from 0 to n-1.
            # The contribution of point i is (ref[0] - p_i[0]) * (area in f1-f2 plane covered by p_i)
            # This is complex.
            
            # Alternative: Use the "Slicing" method or just use a heuristic indicator for selection
            # that correlates with HV.
            
            # For selection, we don't need exact HV, just a good indicator.
            # We can use the sum of distances to the ideal point or similar.
            # But let's try to be a bit smarter.
            
            # Let's just use the non-dominated set size and spread as a proxy if HV is too slow?
            # No, let's implement a simple exact 3D HV for the archive.
            
            # Algorithm for 3D HV:
            # 1. Sort points by f0 ascending.
            # 2. Initialize hv = 0.
            # 3. For i in range(n):
            #    Let p_i be the i-th point.
            #    Let f1_max = -inf, f2_max = -inf (actually we need the "envelope")
            #    This is getting complicated to implement correctly from scratch quickly.
            
            # Let's use a simpler indicator: 
            # Indicator = sum of (ref - f_i) for all non-dominated points? No.
            
            # Let's stick to a robust, simpler selection: 
            # 1. Identify non-dominated points.
            # 2. Among non-dominated, pick those with the largest "gap" to neighbors (crowding distance) 
            #    or simply use random selection from non-dominated with a bias towards lower sums.
            
            # Actually, the previous DE failed because it didn't maintain diversity well.
            # Let's use a simple Archive-based approach:
            # 1. Maintain an archive of non-dominated points.
            # 2. Generate offspring by mutating random parents from the archive (or population).
            # 3. Update archive.
            
            pass # This helper is not used directly in the loop below, replaced by logic in loop.

        # Simple Non-Dominated Sort and Selection
        def is_dominated(a, b):
            """Returns True if a dominates b (a <= b and a < b in at least one)."""
            return np.all(a <= b) and np.any(a < b)

        # Main Loop
        while used < budget:
            if time.time() > deadline:
                break
            
            # Determine number of offspring to generate in this batch
            # Small batches allow more frequent archive updates
            batch_size = min(5, budget - used)
            
            for _ in range(batch_size):
                if used >= budget:
                    break
                if time.time() > deadline:
                    break
                    
                # Selection of Parents
                # We want to select parents that are diverse and good.
                # Simple strategy: Pick 2 parents randomly from the current population,
                # but with a bias towards non-dominated points.
                
                # Identify non-dominated indices in current population
                non_dom_idx = []
                for i in range(pop_size):
                    dominated = False
                    for j in range(pop_size):
                        if i != j and is_dominated(F[i], F[j]):
                            dominated = True
                            break
                    if not dominated:
                        non_dom_idx.append(i)
                
                # If no non-dominated found (shouldn't happen), use all
                if not non_dom_idx:
                    non_dom_idx = list(range(pop_size))
                
                # Pick 2 parents from non-dominated set (or all if too few)
                # Use random choice
                p1_idx = np.random.choice(non_dom_idx)
                p2_idx = np.random.choice(non_dom_idx)
                
                # Ensure they are distinct if possible
                if p1_idx == p2_idx and len(non_dom_idx) > 1:
                    p2_idx = np.random.choice([x for x in non_dom_idx if x != p1_idx])
                
                # Mutation
                # Use a simple mutation: blend + random noise
                # alpha = np.random.rand()
                # child = alpha * X[p1] + (1-alpha) * X[p2]
                
                # Differential Evolution style mutation is often better for continuous spaces
                # v = X[p1] + F * (X[p2] - X[np.random.choice(pop_size)])
                
                # Let's use a mix:
                # 1. DE mutation
                # 2. Gaussian perturbation
                
                third_idx = np.random.choice(pop_size)
                F_mut = 0.5 + 0.5 * np.random.rand() # Adaptive mutation rate
                CR = 0.5
                
                v = X[p1] + F_mut * (X[p2] - X[third_idx])
                
                # Crossover
                r = np.random.rand(dim)
                j_rand = np.random.randint(dim)
                u = np.where(r <= CR, v, X[p1])
                u[j_rand] = v[j_rand]
                
                # Bounds handling: Reflect
                for d in range(dim):
                    if u[d] < lo[d]:
                        u[d] = 2 * lo[d] - u[d]
                    elif u[d] > hi[d]:
                        u[d] = 2 * hi[d] - u[d]
                    u[d] = np.clip(u[d], lo[d], hi[d])
                
                # Evaluate
                f_u = problem(u)
                used += 1
                
                # Update Population
                # Simple strategy: Replace the worst point in the population?
                # Or use a tournament selection against the current point?
                
                # Let's try: If u is non-dominated in the population, replace a random point.
                # If u is dominated, replace the point it dominates (if any) with some probability, 
                # or the worst point based on a scalarization.
                
                # Scalarization: Sum of normalized objectives? 
                # We don't know the range. Let's use the sum of raw objectives.
                # This is biased if scales differ, but RE34 objectives might be similar scale?
                # Reference: [1705, 12, 0.3]. Scales differ significantly.
                # Better: Use the "distance to reference point" or similar.
                
                # Let's calculate a score for each point in population:
                # score = sum( (ref - f) / ref )  # Normalized improvement
                # Lower is better? No, higher improvement is better.
                # We want to keep points with high "potential".
                
                # Actually, the most robust simple method is:
                # 1. If u dominates any point in pop, replace the first one found.
                # 2. Else, if u is non-dominated, replace a random point (to maintain diversity).
                # 3. Else (u is dominated), replace the point with the worst "sum" if u is better than it.
                
                dominated_idx = None
                for i in range(pop_size):
                    if is_dominated(f_u, F[i]):
                        dominated_idx = i
                        break
                
                if dominated_idx is not None:
                    X[dominated_idx] = u
                    F[dominated_idx] = f_u
                else:
                    # Check if u is non-dominated
                    is_non_dom = True
                    for i in range(pop_size):
                        if is_dominated(F[i], f_u):
                            is_non_dom = False
                            break
                    
                    if is_non_dom:
                        # Replace a random point to ensure diversity
                        replace_idx = np.random.randint(pop_size)
                        X[replace_idx] = u
                        F[replace_idx] = f_u
                    else:
                        # u is dominated. Check if it's better than the worst point by sum.
                        # Calculate sums
                        sums = np.sum(F, axis=1)
                        worst_idx = np.argmin(-sums) # Index of max sum (worst)
                        if np.sum(f_u) < sums[worst_idx]:
                            X[worst_idx] = u
                            F[worst_idx] = f_u

        return X

# EVOLVE-BLOCK-END
