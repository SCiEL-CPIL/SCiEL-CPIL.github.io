# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for RE37 using Differential Evolution
    with Pareto front maintenance."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Reference point for hypervolume (given in task)
        # ref = np.array([1.1, 1.3, 1.1])

        rng = np.random.default_rng(0)

        def dominates(a, b):
            """Returns True if a dominates b (all objectives <= and at least one <)."""
            return np.all(a <= b) and np.any(a < b)

        def is_non_dominated(f, F_all):
            """Check if f is non-dominated with respect to F_all."""
            for other in F_all:
                if dominates(other, f):
                    return False
            return True

        # Initialize with stratified sampling (Latin Hypercube-like)
        # Use 40 points for initialization to leave 160 for search
        n_init = 40
        X_init = np.random.uniform(lo, hi, (n_init, dim))
        # Better: Stratified
        # For 4D, 40 points is a bit sparse for 4x4x4x4 grid (256 cells).
        # Let's just use random uniform for init, it's often better for DE.
        # Or, use a simple grid.
        # Let's stick to random uniform for simplicity and robustness.
        
        X = []
        F = []
        used = 0

        # Initialization
        for _ in range(n_init):
            if used >= budget or time.time() > deadline:
                break
            x = rng.uniform(lo, hi)
            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=float)
            X.append(x)
            F.append(f)
            used += 1

        X = np.array(X)
        F = np.array(F)

        # Main Optimization Loop: Differential Evolution
        # DE parameters
        F_de = 0.8  # Differential weight
        CR = 1.0    # Crossover rate (always take mutant if it's non-dominated)
        
        # We will maintain a list of all non-dominated points.
        # Since budget is small (200), we can afford to check against all previous points.
        
        while used < budget:
            if time.time() > deadline:
                break

            # Get current non-dominated set indices
            # We can just use the full F array for selection, but NDS is better.
            # Let's maintain an explicit NDS list for efficiency in selection.
            # However, with N=200, checking all is fast.
            
            # Select 3 distinct parents from the current population
            # Prefer non-dominated parents if possible
            n_pop = len(X)
            if n_pop < 3:
                # If population is too small, add random points
                x = rng.uniform(lo, hi)
                x = np.clip(x, lo, hi)
                f = np.array(problem(x), dtype=float)
                X = np.vstack([X, x.reshape(1, -1)])
                F = np.vstack([F, f.reshape(1, -1)])
                used += 1
                continue

            # Identify non-dominated points
            # This is O(N^2), which is 40000 ops max. Fine.
            nds_indices = []
            for i in range(n_pop):
                is_nd = True
                for j in range(n_pop):
                    if i == j:
                        continue
                    if dominates(F[j], F[i]):
                        is_nd = False
                        break
                if is_nd:
                    nds_indices.append(i)
            
            # If NDS is empty (shouldn't happen), use all
            if not nds_indices:
                nds_indices = list(range(n_pop))
            
            # Select 3 distinct parents from NDS
            # If NDS is small, sample with replacement from NDS or fallback to all
            if len(nds_indices) >= 3:
                p1, p2, p3 = rng.choice(nds_indices, size=3, replace=False)
            else:
                # Fallback: pick from all, ensuring distinctness if possible
                p1, p2, p3 = rng.choice(n_pop, size=3, replace=False)

            # Generate mutant
            v = X[p1] + F_de * (X[p2] - X[p3])
            v = np.clip(v, lo, hi)

            # Evaluate
            f_v = np.array(problem(v), dtype=float)
            used += 1

            # Check dominance
            # 1. Does v dominate any existing point?
            dominated_indices = []
            for i in range(n_pop):
                if dominates(f_v, F[i]):
                    dominated_indices.append(i)
            
            # 2. Is v dominated by any existing point?
            is_v_dominated = False
            for i in range(n_pop):
                if i in dominated_indices:
                    continue
                if dominates(F[i], f_v):
                    is_v_dominated = True
                    break

            if not is_v_dominated:
                # v is non-dominated or dominates some
                # Remove dominated points from population? 
                # To keep population size manageable and quality high, 
                # we can replace dominated points or just add.
                # Adding leads to N=200 max.
                # Let's just add it.
                X = np.vstack([X, v.reshape(1, -1)])
                F = np.vstack([F, f_v.reshape(1, -1)])
                
                # Optional: Prune dominated points to keep "elite" set small?
                # For 200 budget, keeping all is fine.
            else:
                # v is dominated. 
                # We could try a local perturbation around v?
                # Or just discard.
                # To improve exploration, maybe try a smaller step?
                # Let's just discard to save time/compute, as DE is stochastic.
                pass

        return X
# EVOLVE-BLOCK-END
