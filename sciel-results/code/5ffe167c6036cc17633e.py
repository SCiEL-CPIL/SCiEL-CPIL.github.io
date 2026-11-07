import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        # Maintain a list of (x, f) for the best non-dominated points
        # We use a simple list and filter dominated points occasionally
        # To be efficient, we keep a small archive of non-dominated points
        # and a larger population of candidates?
        
        # Actually, the "lighter-weight" strategy suggests avoiding full NSGA-II sorting.
        # Let's use a simple "worst-replacement" or "archive" approach.
        # Archive: list of non-dominated points.
        # If new point dominates any in archive, remove dominated.
        # If any in archive dominates new point, discard new point.
        # Else, add new point to archive.
        
        # To avoid O(N^2) blowup in archive size, we can cap the archive or
        # just keep all non-dominated points. With 200 budget, max 200 points.
        # 200^2 = 40,000 checks. Each check is 3 comparisons. This is fast.
        
        archive_x = []
        archive_f = []
        
        used = 0
        
        # 1. Initialization
        # Use Latin Hypercube Sampling (LHS) for initial spread
        pop_size_init = min(20, self.budget)
        X_init = np.zeros((pop_size_init, self.dim))
        for i in range(self.dim):
            samples = np.linspace(0, 1, pop_size_init, endpoint=False)
            np.random.shuffle(samples)
            X_init[:, i] = samples
        X_init = lo + range_ * X_init
        
        for x in X_init:
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
            f = problem(x)
            used += 1
            self._update_archive(x, f)
            
        if used >= self.budget or time.time() > deadline:
            return np.array(archive_x) if archive_x else np.array([lo])

        # 2. Local Search / Exploration
        # We will pick a random point from the archive (or a random point if archive is small)
        # and perturb it.
        # Adaptive step size: start large, decrease if no improvement?
        # Or just use a fixed reasonable step size relative to bounds.
        
        # Let's use a simple random restart + local search hybrid.
        # But the prompt says "single-solution adaptive Gaussian local search".
        # Let's stick to that.
        
        # We need a current solution to refine.
        # Let's pick the best point in the archive based on some heuristic?
        # Or just pick randomly from archive to maintain diversity?
        # Random selection from archive ensures we explore different parts of the front.
        
        # Step size control:
        # Start with sigma = 0.2 * range
        # If a move is accepted (improves hypervolume or is non-dominated), keep sigma or increase slightly?
        # If rejected, decrease sigma?
        # Simplest: Fixed sigma = 0.1 * range per dimension?
        # WFG4 is multimodal. Local search might get stuck.
        # The "worst-replacement" strategy mentioned in insights usually implies a population.
        # But the "single-solution" insight suggests a different path.
        
        # Let's try a "Simulated Annealing" like approach or just random perturbation.
        # Given the budget is small (200), and dim=12, we need efficient exploration.
        
        # Let's implement a simple (1+1) EA with Gaussian mutation.
        # Current best: We don't have a single "best" in multi-objective.
        # We maintain the archive.
        # In each step:
        # 1. Pick a random parent from the archive.
        # 2. Mutate it with Gaussian noise.
        # 3. Clip to bounds.
        # 4. Evaluate.
        # 5. Update archive.
        
        # To avoid getting stuck in a local basin of one point, picking randomly from the archive
        # is good. The archive grows as we find new non-dominated points.
        
        # Step size:
        # Let's use a step size that adapts based on the number of successful moves.
        # Or just a fixed step size that covers a good portion of the space.
        # range_ is the vector of ranges.
        
        sigma = 0.15 # 15% of the range per dimension
        
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Pick a parent from the archive
            # If archive is empty (shouldn't happen), pick random
            if archive_x:
                idx = np.random.randint(0, len(archive_x))
                parent_x = archive_x[idx]
            else:
                parent_x = lo + range_ * np.random.rand(self.dim)
            
            # Gaussian mutation
            # We want the mutation to be relative to the range
            noise = np.random.randn(self.dim) * sigma * range_
            child_x = parent_x + noise
            child_x = np.clip(child_x, lo, hi)
            
            # Evaluate
            f = problem(child_x)
            used += 1
            
            # Update archive
            self._update_archive(child_x, f)
            
            # Optional: Adaptive sigma
            # If we want to be smarter, we could track success rate.
            # But with 200 evals, fixed might be fine.
            # Let's try decreasing sigma slightly over time to refine?
            # No, WFG4 needs exploration. Let's keep it constant or slightly randomize.
            # Actually, let's just keep it constant for simplicity and robustness.
            
        return np.array(archive_x) if archive_x else np.array([lo])

    def _update_archive(self, x, f):
        # Check if x is dominated by any point in archive
        # If yes, discard x.
        # If x dominates any point in archive, remove those points.
        # If neither, add x to archive.
        
        dominated = False
        to_remove = []
        
        for i, (ax, af) in enumerate(zip(self.archive_x, self.archive_f)):
            # Check if ax dominates x
            if np.all(af <= f) and np.any(af < f):
                dominated = True
                break
            # Check if x dominates ax
            if np.all(f <= af) and np.any(f < af):
                to_remove.append(i)
        
        if dominated:
            return
        
        # Remove dominated points
        if to_remove:
            # Efficient removal: create new lists
            # Since archive is small (<200), this is fine
            new_x = []
            new_f = []
            for i in range(len(self.archive_x)):
                if i not in to_remove:
                    new_x.append(self.archive_x[i])
                    new_f.append(self.archive_f[i])
            self.archive_x = new_x
            self.archive_f = new_f
        
        # Add new point
        self.archive_x.append(x)
        self.archive_f.append(f)
