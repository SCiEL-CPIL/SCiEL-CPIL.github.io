import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 290

        N = 20 # Population size
        if N >= budget:
            X = lo + (hi - lo) * np.random.random((budget, dim))
            F = np.array([problem(x) for x in X])
            return X

        # 1. Generate Reference Points
        # For n_obj=3, we can use a grid.
        # Let's use a function to generate uniform points on the simplex.
        ref_points = self.generate_reference_points(n_obj, N)
        
        # 2. Initialize Population with LHS
        X = self.lhs_sample(N, dim, lo, hi)
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = N

        # 3. Main Loop
        while used + N <= budget:
            if time.time() > deadline: break

            # Create Offspring
            offspring_X = np.empty((N, dim))
            # Non-dominated sort for selection
            ranks, fronts = self.non_dominated_sort(F)
            cd = self.crowding_distance(F, fronts)
            
            for i in range(N):
                # Tournament Selection
                p1_idx = self.tournament_select(ranks, cd, 2)
                p2_idx = self.tournament_select(ranks, cd, 2)
                
                x1, x2 = X[p1_idx], X[p2_idx]
                c1, c2 = self.sbx_crossover(x1, x2, lo, hi, 0.9, 20)
                c1 = self.polynomial_mutation(c1, lo, hi, 1.0/dim, 20)
                offspring_X[i] = c1

            F_off = np.array([problem(x) for x in offspring_X], dtype=np.float64)
            used += N

            # Combine
            X_all = np.vstack([X, offspring_X])
            F_all = np.vstack([F, F_off])

            # Non-dominated Sort
            ranks_all, fronts_all = self.non_dominated_sort(F_all)
            
            # Selection with Niche Preservation (NSGA-III style)
            X, F = self.select_population(X_all, F_all, ranks_all, fronts_all, ref_points, N)

        return X

    def generate_reference_points(self, M, N):
        # Generate N points on the (M-1)-simplex
        # For M=3, it's a triangle.
        # We can use the "systematic" method from NSGA-III paper.
        # Or just use random points from Dirichlet and deduplicate?
        # Let's use a grid for M=3.
        # If N is not a perfect grid size, we can add random points.
        # For simplicity, let's use random points from Dirichlet(1,1,1) scaled?
        # No, uniform on simplex is better.
        # Algorithm:
        # 1. Generate random points in [0,1]^M.
        # 2. Normalize so sum is 1.
        # This gives uniform distribution on simplex.
        points = np.random.dirichlet(np.ones(M), size=N)
        return points

    def select_population(self, X, F, ranks, fronts, ref_points, N):
        # Select N individuals
        # 1. Take full fronts
        selected_idx = []
        for front in fronts:
            if len(selected_idx) + len(front) <= N:
                selected_idx.extend(front)
            else:
                # Need to select from this front
                # Use Niche Preservation
                # 1. Normalize F for this front
                # 2. Associate with ref points
                # 3. Pick best
                
                # Get indices of the front
                front_indices = front
                # We need to pick (N - len(selected_idx)) from front_indices
                
                # Normalize objectives for the whole population? Or just the front?
                # NSGA-III normalizes using the ideal point of the whole population.
                z_ideal = np.min(F, axis=0)
                # Normalize
                F_norm = F - z_ideal
                # Avoid division by zero
                F_norm[F_norm == 0] = 1e-10
                
                # Associate solutions in the front with reference points
                # Calculate distance to each reference point
                # dist[i, j] = || F_norm[i] * ref_point[j] || ? No.
                # Angle or Euclidean distance in normalized space.
                # Let's use Euclidean distance between F_norm[i] and ref_point[j] * scale?
                # Standard NSGA-III:
                # 1. Normalize F.
                # 2. For each solution, find the closest reference point.
                
                # We only care about solutions in the last front.
                # Let's compute association for all solutions in the last front.
                
                last_front_indices = front
                n_last = len(last_front_indices)
                n_ref = len(ref_points)
                
                # Calculate distances
                # F_last: (n_last, M)
                # Ref: (n_ref, M)
                # Dist: (n_last, n_ref)
                F_last = F_norm[last_front_indices]
                
                # Normalize ref points? They are already on simplex.
                # F_last is normalized by ideal point.
                # We need to compare them.
                # Usually, we project F_last onto the hyperplane defined by ref points?
                # Simpler: Euclidean distance between F_last[i] and Ref[j].
                # But scales might differ.
                # Let's use the angle between F_last[i] and Ref[j].
                # Or just Euclidean.
                
                dists = np.linalg.norm(F_last[:, np.newaxis, :] - ref_points[np.newaxis, :, :], axis=2)
                
                # For each reference point, find the closest solution
                # And for each solution, find the closest reference point
                
                # We want to select solutions that are well-distributed.
                # Strategy:
                # 1. Count how many solutions are associated with each ref point.
                # 2. Pick solutions from ref points with fewest associations.
                
                # Association:
                # For each solution i in last front, closest ref point is argmin_j dists[i,j]
                closest_ref = np.argmin(dists, axis=1)
                
                # Count associations
                counts = np.zeros(n_ref, dtype=int)
                for cr in closest_ref:
                    counts[cr] += 1
                    
                # We need to select k = N - len(selected_idx) solutions
                k = N - len(selected_idx)
                
                # Sort ref points by count
                ref_order = np.argsort(counts)
                
                selected_in_front = []
                # We will pick one solution per ref point, starting with the least crowded.
                # If we still need more, we pick the next best from the same ref point.
                
                # Let's create a list of (ref_idx, sol_idx_in_front)
                # For each ref point, sort its associated solutions by distance.
                
                # Actually, simpler:
                # Iterate through ref points in order of least count.
                # For each, pick the closest solution not yet picked.
                
                picked_solutions = set()
                
                # To handle "next best" from same ref point, we need to be careful.
                # Let's just pick the closest solution for each ref point,
                # and if we need more, we pick the next closest from the ref points that already have solutions?
                
                # Standard NSGA-III:
                # 1. Associate all solutions.
                # 2. Select solutions from ref points with min count.
                # 3. If still need more, select from ref points with next min count.
                
                # Let's implement this.
                # Create a list of candidates: (count, dist, sol_idx)
                # Sort by count, then dist.
                
                candidates = []
                for i_idx, sol_idx in enumerate(last_front_indices):
                    ref_idx = closest_ref[i_idx]
                    dist = dists[i_idx, ref_idx]
                    candidates.append((counts[ref_idx], dist, sol_idx))
                    
                # Sort
                candidates.sort(key=lambda x: (x[0], x[1]))
                
                # Pick first k
                for i in range(min(k, len(candidates))):
                    selected_idx.append(candidates[i][2])
                    
                break # Exit front loop
            
        return X[selected_idx], F[selected_idx]

    # ... (copy non_dominated_sort, crowding_distance, tournament_select, sbx_crossover, polynomial_mutation, lhs_sample)
