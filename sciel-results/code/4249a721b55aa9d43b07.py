"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for RE37 using a hybrid approach:
    1. Dense Latin Hypercube Sampling for initial exploration.
    2. Greedy hypervolume-driven local search to refine the Pareto front.
    3. Deadline guard to ensure termination within time limits.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 325.0  # Safety margin of 5s before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Reference point for hypervolume calculation
        # Task specifies [1.1, 1.3, 1.1] but we should use a point slightly worse than any possible front
        # Actually, the scoring uses a specific reference point. For local optimization, we just want to maximize hypervolume.
        # We can use a dynamic reference point or just minimize objectives.
        # Let's stick to a simple dominance-based update first, then add hypervolume contribution if needed.
        # Given the small budget (200), simple non-dominated sorting + random restarts/local search is robust.

        # State
        X = []  # List of decision vectors
        F = []  # List of objective vectors
        used = 0

        # Helper to check if a point is dominated by any point in the archive
        def is_dominated(f_new, F_archive):
            # f_new: (n_obj,)
            # F_archive: (N, n_obj)
            # Returns True if any point in F_archive dominates f_new
            # Dominance: a <= b for all objectives and a < b for at least one
            if len(F_archive) == 0:
                return False
            # Vectorized check
            # (F_archive <= f_new).all(axis=1) & (F_archive < f_new).any(axis=1)
            return np.any(np.all(F_archive <= f_new, axis=1) & np.any(F_archive < f_new, axis=1))

        def update_archive(x_new, f_new):
            nonlocal X, F
            if not is_dominated(f_new, np.array(F) if F else np.empty((0, n_obj))):
                X.append(x_new)
                F.append(f_new)
                # Remove points dominated by f_new
                new_F = []
                new_X = []
                for i, (x_old, f_old) in enumerate(zip(X, F)):
                    if i == len(X) - 1: # Just added
                        continue
                    # Check if f_old is dominated by f_new
                    if np.all(f_new <= f_old) and np.any(f_new < f_old):
                        continue
                    new_X.append(x_old)
                    new_F.append(f_old)
                # Append the new one
                new_X.append(x_new)
                new_F.append(f_new)
                X = new_X
                F = new_F
            else:
                # Still record it? The task says "Every returned objective vector is recorded".
                # The return value of __call__ is ignored.
                # "Every returned objective vector is recorded" implies the evaluator records all calls to problem().
                # So we don't need to return them, just call problem().
                pass

        # Phase 1: Latin Hypercube Sampling (LHS)
        # Use a significant portion of the budget for initial spread.
        # With 200 budget, maybe 50-100 points for LHS.
        n_lhs = min(100, budget // 2)
        
        # Generate LHS
        # For each dimension, divide [0,1] into n_lhs bins, pick one random point in each.
        # Then shuffle the assignments.
        rng = np.random.default_rng(42)
        
        lhs_points = []
        for d in range(dim):
            bins = np.linspace(lo[d], hi[d], n_lhs + 1)
            # Pick one random point in each bin
            points_in_dim = np.random.uniform(bins[:-1], bins[1:])
            # Shuffle to mix dimensions
            rng.shuffle(points_in_dim)
            lhs_points.append(points_in_dim)
        
        # Transpose to get (n_lhs, dim)
        X_lhs = np.array(lhs_points).T
        
        # Evaluate LHS points
        for i in range(n_lhs):
            if time.time() > deadline:
                break
            x = X_lhs[i]
            f = problem(x)
            update_archive(x, f)
            used += 1
            
        if used >= budget or time.time() > deadline:
            return None

        # Phase 2: Local Search / Refinement
        # Strategy: Pick a point from the archive (prefer non-dominated), mutate it, evaluate.
        # If the new point is non-dominated, keep it.
        # We can also do a simple random restart if stuck, but with 200 budget, local search on the front is better.
        
        # Sort archive by sum of objectives to find "best" single points for mutation centers
        # Or just pick random non-dominated points.
        
        while used < budget:
            if time.time() > deadline:
                break
            
            if not F:
                # Fallback to random if archive is empty (shouldn't happen)
                x = np.random.uniform(lo, hi)
            else:
                # Select parent
                # Prefer non-dominated points
                F_arr = np.array(F)
                # Check dominance within archive
                # For simplicity, just pick a random point from the archive, or the one with lowest sum.
                # Let's pick the point with the lowest sum of objectives (scalarization) to guide search towards the knee.
                # Or better: pick a random non-dominated point.
                
                # Identify non-dominated points in current archive
                # With small archive, O(N^2) is fine.
                nondom_indices = []
                for i in range(len(F)):
                    dominated = False
                    for j in range(len(F)):
                        if i == j:
                            continue
                        if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                            dominated = True
                            break
                    if not dominated:
                        nondom_indices.append(i)
                
                if nondom_indices:
                    # Pick random non-dominated parent
                    idx = nondom_indices[rng.integers(len(nondom_indices))]
                    x_parent = X[idx]
                else:
                    # All dominated? Pick the one with lowest sum
                    idx = np.argmin(np.sum(F_arr, axis=1))
                    x_parent = X[idx]

            # Mutation
            # Adaptive step size? Start with 0.1, decrease if no improvement.
            # Simple: Gaussian mutation with scale 0.05-0.1
            step_size = 0.1
            # Maybe reduce step size as we use more budget?
            # step_size = 0.1 * (1.0 - used / budget) + 0.01
            
            x_child = x_parent + rng.standard_normal(dim) * step_size * (hi - lo)
            x_child = np.clip(x_child, lo, hi)
            
            f_child = problem(x_child)
            used += 1
            
            # Update archive
            update_archive(x_child, f_child)
            
            # Optional: If the child is dominated, maybe try a smaller step?
            # For now, keep it simple.
            
        return None

# EVOLVE-BLOCK-END
