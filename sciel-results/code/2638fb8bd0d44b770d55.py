"""Multi-objective optimizer for RE34 using DE with Crowding Distance and Local Polishing."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using Differential Evolution with Crowding Distance."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin: 300s limit, kill at 320s to be safe
        deadline = 315.0

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)

        # Use a local RNG to ensure reproducibility within the run, 
        # but independent of the global seed set by the evaluator for the problem itself.
        # However, the evaluator sets np.random.seed(seed). 
        # We should use np.random directly or a derived rng.
        # The prompt says: "evaluator calls np.random.seed(seed), creates Optimizer... and calls it once."
        # So np.random is seeded. We can use np.random directly or create a local one.
        # Using np.random is standard for these tasks.
        # Let's use np.random directly to align with the evaluator's seed.
        
        def dominates(a, b):
            """Returns True if a dominates b (a <= b and a < b in at least one obj)"""
            return np.all(a <= b) and np.any(a < b)

        def crowding_distance(F):
            """Calculate crowding distance for a set of objective vectors F (N, n_obj)."""
            n = F.shape[0]
            if n <= 2:
                return np.full(n, np.inf)
            
            cd = np.zeros(n)
            for m in range(n_obj):
                sorted_indices = np.argsort(F[:, m])
                cd[sorted_indices[0]] = np.inf
                cd[sorted_indices[-1]] = np.inf
                if F[sorted_indices[-1], m] - F[sorted_indices[0], m] == 0:
                    continue
                for i in range(1, n - 1):
                    idx = sorted_indices[i]
                    cd[idx] += (F[sorted_indices[i+1], m] - F[sorted_indices[i-1], m]) / (F[sorted_indices[-1], m] - F[sorted_indices[0], m])
            return cd

        # Archive: list of (x, f)
        archive_X = []
        archive_F = []
        
        used = 0

        def add_to_archive(x, f):
            nonlocal archive_X, archive_F, used
            # Check if dominated by existing
            for af in archive_F:
                if dominates(af, f):
                    return False
            
            # Remove dominated
            new_X = []
            new_F = []
            for i in range(len(archive_X)):
                if not dominates(f, archive_F[i]):
                    new_X.append(archive_X[i])
                    new_F.append(archive_F[i])
            
            new_X.append(x)
            new_F.append(f)
            
            archive_X = new_X
            archive_F = new_F
            return True

        def evaluate_and_add(x):
            nonlocal used
            if used >= budget:
                return None
            # Check deadline
            if time.time() - start_time > deadline:
                return None
            
            # Clip to bounds
            x = np.clip(x, lo, hi)
            f = problem(x)
            used += 1
            
            # Handle NaN/Inf just in case, though problem promises finite
            if np.any(~np.isfinite(f)):
                return None
            
            add_to_archive(x, f)
            return x, f

        # 1. Initialization - more efficient
        # Center
        x0 = (lo + hi) / 2.0
        evaluate_and_add(x0)
        
        # Fewer corners, more diverse random points
        patterns = [
            [0,0,0,0,0], [1,1,1,1,1],
            [0,1,0,1,0], [1,0,1,0,1],
            [0,0,1,1,1], [1,1,0,0,0],
        ]
        for p in patterns:
            x = np.array([hi[j] if p[j]==1 else lo[j] for j in range(dim)])
            evaluate_and_add(x)
            
        # Random points to fill initial diversity
        while used < 12:
            x = lo + (hi - lo) * np.random.random(dim)
            evaluate_and_add(x)

        # 2. Main DE Loop
        # Parameters
        F_factor = 0.8
        CR = 0.9
        # We will generate candidates from the archive.
        # If archive is small, we might need to maintain a population.
        # But with 200 budget, archive-based DE is efficient.
        
        # To avoid stagnation if archive converges, we keep a "population" of the last N evaluated non-dominated points?
        # Actually, just sampling from the archive is standard for MOEA/D-like or simple PADE.
        
        while used < budget:
            if time.time() - start_time > deadline:
                break
            
            # Switch to polishing mode if near end
            if used > budget - 30:
                # Polishing: Perturb best points with smaller steps
                if len(archive_X) == 0:
                    # If empty, just random
                    x = lo + (hi - lo) * np.random.random(dim)
                    evaluate_and_add(x)
                    continue
                
                # Select a point to polish
                # Pick the one with max crowding distance for diversity
                if len(archive_X) > 1:
                    F_arr = np.array(archive_F)
                    cd = crowding_distance(F_arr)
                    idx = np.argmax(cd)
                else:
                    idx = 0
                base_x = archive_X[idx]
                
                # Small random perturbation with decreasing step size
                progress = (budget - used) / 30.0  # 1.0 to 0.0
                step = 0.05 * (hi - lo) * progress + 0.01 * (hi - lo)
                x = base_x + np.random.normal(0, 1, dim) * step
                evaluate_and_add(x)
            else:
                # DE Mode
                if len(archive_X) < 3:
                    # Need more diversity, do random
                    x = lo + (hi - lo) * np.random.random(dim)
                    evaluate_and_add(x)
                    continue
                
                # Select 3 distinct parents from archive
                # If archive is small, wrap around or repeat? 
                # Distinct indices is better.
                if len(archive_X) >= 3:
                    idx1, idx2, idx3 = np.random.choice(len(archive_X), 3, replace=False)
                    x1, x2, x3 = archive_X[idx1], archive_X[idx2], archive_X[idx3]
                else:
                    # Fallback if archive < 3 (shouldn't happen after init)
                    idx1, idx2, idx3 = np.random.choice(len(archive_X), 3, replace=True)
                    x1, x2, x3 = archive_X[idx1], archive_X[idx2], archive_X[idx3]

                # DE Mutation with adaptive F
                # Use smaller F as we progress to refine
                progress = used / (budget - 30)
                F_current = F_factor * (1.0 - 0.5 * progress)
                
                trial = x1 + F_current * (x2 - x3)
                
                # Binomial Crossover with a random parent (use x1 as base)
                target = x1
                j_rand = np.random.randint(dim)
                cross_mask = np.random.random(dim) < CR
                cross_mask[j_rand] = True # Ensure at least one gene is mutated
                
                trial = np.where(cross_mask, trial, target)
                
                evaluate_and_add(trial)
                
                # Optional: If archive is growing too large, prune by crowding distance
                if len(archive_X) > 45:
                    # Calculate crowding distance
                    F_arr = np.array(archive_F)
                    cd = crowding_distance(F_arr)
                    # Keep top 35 by crowding distance
                    sort_idx = np.argsort(-cd)
                    keep_idx = sort_idx[:35]
                    archive_X = [archive_X[i] for i in keep_idx]
                    archive_F = [archive_F[i] for i in keep_idx]

        return None
# EVOLVE-BLOCK-END
