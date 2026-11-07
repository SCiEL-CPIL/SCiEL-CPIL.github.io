import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 290.0 # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        N = 40 # Population size
        # Ensure N doesn't exceed budget
        if N > self.budget:
            N = self.budget
            
        # Initialize
        X = lo + (hi - lo) * np.random.random((N, self.dim))
        F = np.array([problem(x) for x in X])
        evals = N
        
        # Archive to store best non-dominated points found so far
        # We will maintain a list of (x, f) that are non-dominated
        # For efficiency, we can just keep the final population's non-dominated set
        # but an archive is safer for hypervolume.
        
        # Let's stick to standard NSGA-II flow but return the non-dominated set of the final population 
        # OR the archive. 
        # Given the small budget, the final population is likely very close to the front.
        # However, early good points might be lost.
        # Let's maintain a global non-dominated archive.
        
        archive_X = []
        archive_F = []
        
        def update_archive(x, f):
            # Check if x is dominated by any in archive
            for i in range(len(archive_F)):
                if np.all(archive_F[i] <= f) and np.any(archive_F[i] < f):
                    return # x is dominated
            
            # Remove dominated points from archive
            i = 0
            while i < len(archive_F):
                if np.all(f <= archive_F[i]) and np.any(f < archive_F[i]):
                    archive_X.pop(i)
                    archive_F.pop(i)
                else:
                    i += 1
            
            archive_X.append(x)
            archive_F.append(f)

        # Update archive with initial pop
        for i in range(N):
            update_archive(X[i], F[i])

        while evals < self.budget:
            if time.time() > deadline:
                break
                
            # Create offspring
            offspring_X = np.empty((N, self.dim))
            offspring_F = np.empty((N, self.n_obj))
            
            for i in range(N):
                # Selection: Binary Tournament
                idx1 = np.random.randint(N)
                idx2 = np.random.randint(N)
                # Need non-dominated sort and crowding distance for selection
                # This is expensive to do every time. 
                # Standard NSGA-II does it once per generation on the combined pool.
                # So we should generate the whole offspring population first, 
                # then combine, sort, select.
                pass
            
            # Correct NSGA-II Flow:
            # 1. Generate offspring (size N) using current parents
            # 2. Combine P_t and Q_t (size 2N)
            # 3. Non-dominated sort
            # 4. Crowding distance
            # 5. Select next P_{t+1}
            
            # We need to store current parents X, F
            
            # Generate Offspring
            off_X = np.zeros((N, self.dim))
            for i in range(N):
                # Tournament Selection on current X, F
                # We need ranks and crowding for X, F
                # To avoid recalculating every time, we can do it once per generation.
                # But we need to select parents *before* creating offspring.
                # So:
                # 1. Calculate ranks/crowding for current X, F
                # 2. Select N pairs of parents
                # 3. Generate N offspring
                # 4. Evaluate offspring
                # 5. Combine and select next parents
                
                pass
            
            # Let's restructure the loop to match NSGA-II
            # Current Parents: X, F
            
            # 1. Sort Current Population
            ranks, crowding = self._non_dominated_sort(X, F)
            
            # 2. Generate Offspring
            off_X = np.zeros((N, self.dim))
            for i in range(N):
                # Binary Tournament
                p1_idx, p2_idx = self._tournament(ranks, crowding)
                p1 = X[p1_idx]
                p2 = X[p2_idx]
                
                # Crossover (SBX)
                child = self._sbx(p1, p2, lo, hi)
                # Mutation (Poly)
                child = self._poly_mut(child, lo, hi)
                off_X[i] = child
                
            # 3. Evaluate Offspring
            off_F = np.array([problem(x) for x in off_X])
            evals += N
            if evals > self.budget:
                # This shouldn't happen if we check before, but just in case
                break
                
            # 4. Update Archive with Offspring
            for i in range(N):
                update_archive(off_X[i], off_F[i])
                
            # 5. Combine and Select
            # Combine X and off_X
            X_comb = np.vstack([X, off_X])
            F_comb = np.vstack([F, off_F])
            
            ranks_comb, crowding_comb = self._non_dominated_sort(X_comb, F_comb)
            
            # Select N best
            # Sort by rank, then by crowding (descending)
            # We need to pick the first N from this sorted list
            order = self._select_order(ranks_comb, crowding_comb)
            X = X_comb[order[:N]]
            F = F_comb[order[:N]]
            
        # Return non-dominated points from Archive
        # The archive contains all non-dominated points found.
        # We should return them as a list of vectors or an array.
        # The task says "Every returned objective vector is recorded".
        # The return value of __call__ is ignored.
        # But we should return something valid.
        
        return archive_X
