"""Optimized program for WFG9 using MOEA/D with neighborhood mating and external archive."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """MOEA/D-based Multi-Objective Optimizer."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Number of subproblems
        n_sub = 30
        
        # Generate reference points on simplex
        # For 3 objectives, we use a grid on the triangle.
        # H is the number of divisions. n_sub = (H+1)*(H+2)/2
        # H=7 -> 36 points. H=6 -> 28 points.
        # Let's use H=7 for 36 subproblems? Or stick to 30?
        # The prompt suggests a structural change. MOEA/D typically uses a fixed grid.
        # Let's use a grid of H=7 (36 points) to be robust, or H=6 (28 points).
        # 200 budget / 36 ~ 5 generations. 200 / 28 ~ 7 generations.
        # Let's use H=7 (36 subproblems) for better coverage.
        H = 7
        ref_points = self._generate_grid_points(H, self.n_obj)
        n_sub = len(ref_points)
        
        # Build neighborhood for each subproblem
        # Distance between reference points
        dist = np.zeros((n_sub, n_sub))
        for i in range(n_sub):
            for j in range(n_sub):
                dist[i, j] = np.linalg.norm(ref_points[i] - ref_points[j])
        
        # Sort distances for each i
        sorted_indices = np.argsort(dist, axis=1)
        
        # Neighborhood size T
        T = 10
        neighbors = []
        for i in range(n_sub):
            neighbors.append(sorted_indices[i, :T])
        
        # Initialize population
        X = np.zeros((n_sub, self.dim))
        F = np.zeros((n_sub, self.n_obj))
        
        for i in range(n_sub):
            X[i] = lo + (hi - lo) * np.random.rand(self.dim)
            F[i] = problem(X[i])
        
        evaluations = n_sub
        
        # External archive for non-dominated solutions
        archive_X = []
        archive_F = []
        
        # Add initial population to archive
        for i in range(n_sub):
            if self._is_non_dominated(F[i], archive_F):
                archive_X.append(X[i])
                archive_F.append(F[i])
        
        # DE parameters
        CR = 0.5
        F_DE = 0.5
        
        # Main loop
        while evaluations < self.budget:
            if time.time() > deadline:
                break
            
            # Select a subproblem to evolve
            # Random selection is standard, or use a strategy.
            # To ensure all are explored, we can cycle or pick randomly.
            # With small budget, random is fine.
            idx = np.random.randint(n_sub)
            
            # Select parents from neighborhood
            # Pick 2 distinct parents from neighbors[idx]
            # If T < 2, pick from all? T=10 so it's fine.
            # Ensure we don't pick idx itself? MOEA/D usually allows idx.
            # Let's pick 2 distinct indices from neighbors[idx]
            # neighbors[idx] contains idx? Yes, dist[i,i]=0 so it's first.
            # Let's pick 2 distinct parents from neighbors[idx] excluding idx?
            # Standard MOEA/D: pick from neighborhood.
            
            # To avoid picking the same individual twice, we sample 2 distinct indices.
            # If we include idx, we might pick idx.
            # Let's just pick 2 distinct indices from the neighbor list.
            # If the list has duplicates? No, it's a permutation.
            
            # Sample 2 distinct parents
            p1_idx, p2_idx = np.random.choice(neighbors[idx], 2, replace=False)
            
            p1 = X[p1_idx]
            p2 = X[p2_idx]
            current = X[idx]
            
            # DE/rand/1 mutation
            # v = p1 + F * (p2 - current)? Or v = p1 + F * (p2 - p3)?
            # Standard DE/rand/1 uses 3 parents.
            # Let's pick a 3rd parent p3 from neighborhood, distinct from p1, p2.
            # If possible.
            
            # To keep it simple and fast, let's use DE/current-to-p1/1 or similar?
            # Or just DE/rand/1 with 3 parents.
            
            # Let's pick p3.
            candidates = [j for j in neighbors[idx] if j != p1_idx and j != p2_idx]
            if len(candidates) > 0:
                p3_idx = np.random.choice(candidates)
            else:
                p3_idx = p1_idx # Fallback
            
            p3 = X[p3_idx]
            
            v = p1 + F_DE * (p2 - p3)
            
            # Crossover
            child = np.zeros_like(v)
            j_rand = np.random.randint(self.dim)
            for i in range(self.dim):
                if np.random.rand() < CR or i == j_rand:
                    child[i] = v[i]
                else:
                    child[i] = current[i]
            
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            evaluations += 1
            
            # Update using Tchebycheff scalarization
            # Ideal point z
            # We can maintain a running ideal point or compute from population.
            # Computing from population is O(N*3) which is fast.
            z = np.min(F, axis=0)
            
            # Update z if child is better
            z = np.minimum(z, fc)
            
            w = ref_points[idx]
            
            # Calculate fitness for current and child
            curr_fit = self._tchebycheff(F[idx], w, z)
            child_fit = self._tchebycheff(fc, w, z)
            
            # Accept if child is better
            if child_fit <= curr_fit:
                X[idx] = child
                F[idx] = fc
                
                # Update archive
                if self._is_non_dominated(fc, archive_F):
                    # Add to archive
                    archive_X.append(child)
                    archive_F.append(fc)
                    
                    # Remove dominated points from archive
                    new_archive_X = []
                    new_archive_F = []
                    for k in range(len(archive_F)):
                        # If archive point k is dominated by fc, remove it
                        if not (fc <= archive_F[k]).all():
                            # Check if archive_F[k] is dominated by fc
                            # Dominated if fc <= archive_F[k] and fc != archive_F[k]
                            if (fc < archive_F[k]).any() and (fc <= archive_F[k]).all():
                                continue
                            else:
                                new_archive_X.append(archive_X[k])
                                new_archive_F.append(archive_F[k])
                    
                    # Also check if any other archive points dominate each other?
                    # The archive should only contain non-dominated points.
                    # When we add a new point, we remove points it dominates.
                    # Points that dominate the new point were already there, and the new point was rejected if dominated.
                    # So we only need to remove points dominated by the new point.
                    
                    archive_X = new_archive_X
                    archive_F = new_archive_F
                    
                    # Limit archive size?
                    # If archive gets too large, it slows down dominance checks.
                    # With budget 200, max 200 points. 200x200 check is 40k ops, fine.
            
        return X

    def _tchebycheff(self, f, w, z):
        return np.max(w * np.abs(f - z))

    def _is_non_dominated(self, f, archive_F):
        """Check if f is non-dominated by any point in archive_F."""
        for f_other in archive_F:
            if (f_other <= f).all() and (f_other < f).any():
                return False
        return True

    def _generate_grid_points(self, H, n_obj):
        """Generate grid points on simplex for n_obj objectives."""
        if n_obj == 3:
            pts = []
            for i in range(H + 1):
                for j in range(H - i + 1):
                    k = H - i - j
                    pts.append([i / H, j / H, k / H])
            return np.array(pts)
        else:
            # Fallback for other n_obj
            pts = np.random.rand(H * H, n_obj)
            sums = pts.sum(axis=1, keepdims=True)
            pts = pts / sums
            return pts
# EVOLVE-BLOCK-END
