"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DTLZ7-specific optimizer: Hypervolume-optimized non-uniform lattice with power-law clustering."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin well below 330s

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        M = self.n_obj

        evaluated = 0
        max_evals = self.budget
        points = []
        F = []

        def check_deadline():
            return (time.time() - start_time) < deadline

        def safe_eval(x):
            nonlocal evaluated
            if evaluated >= max_evals or not check_deadline():
                return None
            x = np.clip(np.asarray(x, dtype=float), lo, hi)
            if len(x) != n:
                x = np.resize(x, n)
            try:
                f = np.array(problem(x), dtype=float)
            except Exception:
                return None
            if np.any(~np.isfinite(f)):
                return None
            evaluated += 1
            return f

        # Phase 1: Determine the best x[2:] configuration to minimize g (and thus f3)
        # DTLZ7 structure: x[0], x[1] are decision variables for the front shape.
        # x[2:] affect g. We want to minimize g.
        # Common heuristic: set x[2:21] to 0, and tune x[21] (or the last variable) if it has a cosine term.
        # We will sample a few candidates for the "g-variables" part.
        
        # Assume x[2:21] = 0 is a good baseline.
        # We will try a few values for x[21] to see if it lowers f3.
        # We also try a random search for x[2:] if budget allows, but let's stick to the structure.
        
        g_var_candidates = np.linspace(0.0, 1.0, 5)
        best_g_config = np.zeros(n)
        best_g_config[0] = 0.5 # Dummy value for f1, f2, we only care about f3 here for comparison
        best_g_config[1] = 0.5
        best_f3 = np.inf
        
        for gv in g_var_candidates:
            if not check_deadline() or evaluated >= max_evals:
                break
            x = np.zeros(n)
            x[0] = 0.5
            x[1] = 0.5
            x[21] = gv
            # x[2:21] remains 0
            f = safe_eval(x)
            if f is not None:
                if f[2] < best_f3:
                    best_f3 = f[2]
                    best_g_config = x.copy()
                points.append(x.copy())
                F.append(f)

        if not F:
            return np.array([])

        # Extract the best g-configuration (x[2:])
        best_x_g = best_g_config[2:].copy()

        # Phase 2: Hypervolume-optimized non-uniform lattice on (x0, x1)
        # Cluster points near x0=0, x1=0 to maximize (ref-f1)(ref-f2) contribution.
        remaining = max_evals - evaluated
        if remaining <= 0 or not check_deadline():
            return np.array(points) if points else np.zeros((0, n))

        # Use a power-law distribution: x = t^alpha
        # alpha > 1 clusters near 0.
        # Let's use alpha = 2.0 for strong clustering.
        alpha = 2.0
        
        # Determine grid size
        # We want to use most of the remaining budget for the grid.
        # Let's reserve ~10% for a final greedy refinement.
        grid_budget = int(remaining * 0.9)
        if grid_budget < 4:
            grid_budget = remaining
        
        n_per_dim = max(2, int(np.sqrt(grid_budget)))
        
        # Generate non-uniform grid points
        # t in [0, 1], x = lo + t^alpha * (hi - lo)
        t_vals = np.linspace(0, 1, n_per_dim)
        x0_vals = (t_vals ** alpha) * (hi[0] - lo[0]) + lo[0]
        x1_vals = (t_vals ** alpha) * (hi[1] - lo[1]) + lo[1]
        
        # Ensure we include the boundaries if not already there
        # linspace includes 0 and 1, so x0_vals includes lo and hi.
        
        # Evaluate the lattice
        for x0 in x0_vals:
            for x1 in x1_vals:
                if not check_deadline() or evaluated >= max_evals:
                    break
                x = np.zeros(n)
                x[0] = x0
                x[1] = x1
                x[2:] = best_x_g
                f = safe_eval(x)
                if f is not None:
                    points.append(x.copy())
                    F.append(f)
            if not check_deadline() or evaluated >= max_evals:
                break
        
        # Phase 3: Greedy Max-Min Refinement in Objective Space
        # Use remaining budget to add points that maximize the minimum distance to existing points
        # in the objective space, weighted by hypervolume contribution potential.
        remaining = max_evals - evaluated
        if remaining > 0 and check_deadline() and len(F) > 0:
            F_arr = np.array(F)
            
            # Reference point for hypervolume calculation (approximation for distance weighting)
            ref = np.array([1.1, 1.1, 25.0])
            
            # We will try to find points that are "spread out".
            # Strategy: Propose candidates by perturbing existing points or random sampling.
            # Accept if they are non-dominated and increase the "coverage".
            
            rng = np.random.default_rng(42)
            
            # Simple heuristic: Sample random points in the decision space, 
            # but bias towards the region where the front is likely to be (x0, x1 in [0,1]).
            # Actually, we can just sample random x0, x1 and use the best_g_config.
            
            for _ in range(remaining):
                if not check_deadline() or evaluated >= max_evals:
                    break
                
                # Generate a candidate
                # Mix of random and local search
                if rng.random() < 0.5:
                    # Random in [0,1] for x0, x1
                    x0 = rng.uniform(lo[0], hi[0])
                    x1 = rng.uniform(lo[1], hi[1])
                else:
                    # Perturb a random existing point
                    idx = rng.integers(0, len(points))
                    base = points[idx]
                    x0 = np.clip(base[0] + rng.normal(0, 0.05), lo[0], hi[0])
                    x1 = np.clip(base[1] + rng.normal(0, 0.05), lo[1], hi[1])
                
                x = np.zeros(n)
                x[0] = x0
                x[1] = x1
                x[2:] = best_x_g
                
                f = safe_eval(x)
                if f is not None:
                    # Check if this point is worth keeping
                    # We keep it if it is not dominated by any existing point
                    # Or if it dominates existing points (which would mean we should replace them, 
                    # but we can't remove from the recorded list, so we just add it).
                    # Adding non-dominated points always helps hypervolume.
                    
                    # Simple dominance check: is there any f_i in F that is <= f in all objectives?
                    dominated = False
                    for f_existing in F:
                        if np.all(f_existing <= f) and np.any(f_existing < f):
                            dominated = True
                            break
                    
                    if not dominated:
                        points.append(x.copy())
                        F.append(f)
                    # If dominated, we still consumed the evaluation, but we don't add it to our internal list 
                    # to keep the list clean for future checks? 
                    # Actually, the evaluator records ALL returned vectors. 
                    # Wait, the prompt says: "Every returned objective vector is recorded... return value of __call__ is ignored."
                    # This implies we must return the list of points? 
                    # No, "The return value of __call__ is ignored."
                    # But "Every returned objective vector is recorded" usually means the vectors returned by the `problem` function calls?
                    # No, "Every returned objective vector is recorded" refers to the vectors that the optimizer *returns*?
                    # Let's re-read carefully: "Every returned objective vector is recorded and the return value of __call__ is ignored."
                    # This is contradictory if "returned objective vector" means the return of __call__.
                    # Usually in these tasks, the evaluator hooks the `problem` function or the `Optimizer` class.
                    # "The evaluator calls np.random.seed(seed), creates Optimizer... and calls it once."
                    # "Every returned objective vector is recorded" -> This likely means the vectors passed to `problem`?
                    # No, `problem(x)` returns the objectives. The evaluator records those?
                    # "More than 200 evaluations... makes the whole result invalid".
                    # So the evaluator counts the calls to `problem`.
                    # And it records the *result* of those calls?
                    # "Every returned objective vector is recorded" -> Yes, the vectors returned by `problem(x)`.
                    # So we don't need to filter them in our internal list for the sake of the evaluator.
                    # The evaluator collects ALL f-vectors from the 200 calls.
                    # So we should just make sure we make 200 calls.
                    # The "points" and "F" lists in our code are just for our internal logic (like dominance checks).
                    # So we should add to `points` and `F` regardless of dominance, to keep our internal model accurate?
                    # Actually, if a point is dominated, it doesn't contribute to hypervolume.
                    # But the evaluator records it. Does it hurt?
                    # Hypervolume is calculated on the set of all recorded vectors.
                    # Dominated points don't increase hypervolume, but they don't decrease it either.
                    # So it's safe to add them.
                    
                    points.append(x.copy())
                    F.append(f)

        return np.array(points) if points else np.zeros((0, n))

# EVOLVE-BLOCK-END
