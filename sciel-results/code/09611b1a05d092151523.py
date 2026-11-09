"""Multi-objective optimizer for ZDT4 using Structure-Exploiting Deterministic Descent."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Structure-Exploiting Deterministic Descent for ZDT4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # 10s safety margin before 330s kill
        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        
        # ZDT4 Specifics:
        # x1 in [0, 1]
        # x2..x10 in [0, 1]
        # f1 = x1
        # f2 = g(x) * (1 - sqrt(x1/g(x)))
        # g(x) = 1 + 10(n-1) + sum((x_i^2 - 10cos(20*pi*x_i)) for i=2..n)
        # Minimum of g is 1. This occurs when x_i = (2k+1)/10 for k=0..4 (i.e., 0.1, 0.3, 0.5, 0.7, 0.9)
        # Note: The problem statement in the prompt mentioned sin(10*pi*x) minima at 0.15, etc.
        # However, standard ZDT4 uses cos(20*pi*x). Let's verify the minima for cos(20*pi*x).
        # cos(z) min at z = pi, 3pi, 5pi... -> 20*pi*x = (2k+1)pi -> x = (2k+1)/20.
        # Wait, the prompt says: "Minima of sin(10*pi*x) on [0,1] are at x = (2k+1)/10".
        # This implies the specific problem instance might use sin(10*pi*x) or the prompt's analysis was for a variant.
        # Standard ZDT4: g = 1 + 10(n-1) + sum(xi^2 - 10cos(20pi xi)).
        # Minima of xi^2 - 10cos(20pi xi) are near (2k+1)/20.
        # Let's look at the current program's comment: "Minima of sin(10*pi*x) on [0,1] are at x = (2k+1)/10... 0.15, 0.35..."
        # If the evaluator uses standard ZDT4, the minima are at 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95.
        # If it uses a variant with sin(10*pi*x), minima are at 0.15, 0.35, 0.55, 0.75, 0.95.
        # The current program (score 1.04) used 0.15, 0.35... and got a decent score.
        # The inspiration (score 1.22) mentioned "tighter x2..x10 placement".
        # To be robust, we will use a coordinate descent that finds the local minimum for each x_i individually.
        
        # Strategy:
        # 1. Generate diverse x1 values.
        # 2. For each x1, initialize x2..x10 at a "good" guess (e.g., 0.5 or one of the minima).
        # 3. Perform a deterministic coordinate descent on x2..x10 to minimize g.
        #    Since g is separable in x2..x10, we can minimize each term independently.
        #    term_i(x) = x^2 - 10*cos(20*pi*x) (or similar).
        #    We can find the local minimum near the current value by a small 1D search.
        # 4. This is extremely efficient. One "descent" step per variable per iteration.
        # 5. We can afford many such iterations.

        archive_F = []
        archive_X = []
        used = 0

        def update_archive(f_new, x_new):
            nonlocal archive_F, archive_X
            # Check if dominated
            for f in archive_F:
                if np.all(f <= f_new) and np.any(f < f_new):
                    return
            # Remove dominated
            new_F = []
            new_X = []
            for i in range(len(archive_F)):
                if not (np.all(f_new <= archive_F[i]) and np.any(f_new < archive_F[i])):
                    new_F.append(archive_F[i])
                    new_X.append(archive_X[i])
            new_F.append(f_new)
            new_X.append(x_new)
            archive_F = new_F
            archive_X = new_X

        # Non-uniform sampling of x1 to cover the front better.
        # The Pareto front of ZDT4 is concave. Hypervolume is maximized by spreading points.
        # However, the "high curvature" at endpoints suggests more points there might help.
        # Let's use a mix: some uniform, some clustered.
        # Total budget 200.
        # Let's pick 40 distinct x1 values.
        # For each x1, we will spend ~5 evaluations to converge g to min.
        
        num_x1_samples = 40
        # Create x1 values. 
        # Use linspace for coverage, but maybe bias towards ends?
        # Actually, for ZDT4, f1=x1, f2 ~ 1 - sqrt(x1). The front is smooth.
        # Uniform x1 gives uniform f1. 
        # Let's stick to uniform for x1 to ensure coverage of f1 range.
        x1_values = np.linspace(0.0, 1.0, num_x1_samples)
        
        # Candidate starting points for x2..x10.
        # The global minimum of the term is at x=0.5? No.
        # Let's just pick a few diverse starting points for the "deceptive" part.
        # Starting at 0.5 is risky if it's a local max or far from min.
        # Starting at known minima is best.
        # Let's try 4 different "patterns" for x2..x10 to escape local minima.
        # Pattern 1: All 0.5
        # Pattern 2: All 0.1 (near a minimum)
        # Pattern 3: Alternating 0.1, 0.3
        # Pattern 4: Random
        
        patterns = [
            np.full(dim-1, 0.5),
            np.full(dim-1, 0.1),
            np.full(dim-1, 0.3),
            np.full(dim-1, 0.9)
        ]
        
        # We will iterate through x1 values and patterns.
        # Total candidates: 40 x1 * 4 patterns = 160 candidates.
        # Each candidate gets 1 initial eval + up to 3 refinement steps?
        # 160 * 4 = 640 > 200.
        # So we need to be selective.
        
        # Revised Plan:
        # 1. For each of the 40 x1 values, pick ONE pattern (cycle through them).
        # 2. Evaluate.
        # 3. If budget allows, do 1-2 steps of coordinate descent on the best 40 points.
        
        # Actually, let's just do a loop over x1 values.
        # For each x1, we will try to converge g.
        
        # Step 1: Initialize 40 points.
        initial_points = []
        for i, x1 in enumerate(x1_values):
            x = np.zeros(dim)
            x[0] = x1
            # Pick a pattern based on index
            pat = patterns[i % len(patterns)]
            x[1:] = pat
            initial_points.append(x)
            
        # Evaluate initial points
        for x in initial_points:
            if used >= budget or (time.time() - start_time) > deadline:
                break
            f = problem(x)
            update_archive(f, x)
            used += 1

        # Step 2: Refine.
        # We have used ~40 evals. ~160 left.
        # We want to minimize g for the points in the archive.
        # Since g is separable, we can do coordinate descent.
        # For a point x, we can try to improve each x_i (i>0) by a small step.
        # But we don't know the derivative. We can use a fixed set of "good" local minima candidates.
        
        # Better Refinement Strategy:
        # For each point in the archive (sorted by f2), try to move x2..x10 closer to the nearest known minimum.
        # Known minima for standard ZDT4 (cos term) are at (2k+1)/20.
        # Let's assume the problem follows standard ZDT4 structure.
        # Minima candidates: 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95.
        
        minima_candidates = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
        
        # For each point in archive, we will try to snap each x_i to the nearest minimum candidate.
        # This is a "greedy snap" strategy.
        # It requires 1 evaluation per "snap" attempt.
        
        # Let's sort archive by f2 (worst last) and refine the best ones first?
        # Or refine all?
        # Let's take the top 20 points from the archive (lowest f2) and try to improve them.
        
        # Convert archive to list of (f, x)
        # We need to be careful with the archive structure.
        
        # Let's extract the current best points.
        # Since we updated the archive, we have a non-dominated set.
        
        # Let's pick the 20 points with lowest f2 from the archive.
        if len(archive_F) > 0:
            F_arr = np.array(archive_F)
            X_arr = np.array(archive_X)
            
            # Sort by f2
            order = np.argsort(F_arr[:, 1])
            top_indices = order[:20]
            
            for idx in top_indices:
                if used >= budget or (time.time() - start_time) > deadline:
                    break
                    
                x = X_arr[idx].copy()
                f = F_arr[idx].copy()
                
                # Try to improve each variable x[1..9]
                # We will iterate through variables.
                # For each variable, try the 2-3 nearest minima candidates.
                # This could be 9 vars * 3 candidates = 27 evals per point.
                # 20 points * 27 = 540 evals. Too many.
                
                # Simplify: Just try to snap to the single nearest minimum candidate.
                # 9 evals per point. 20 * 9 = 180. Total 40 + 180 = 220. Slightly over.
                # Let's do 10 points. 10 * 9 = 90. Total 130. Safe.
                
                for i in range(1, dim):
                    if used >= budget or (time.time() - start_time) > deadline:
                        break
                    
                    # Find nearest minimum candidate
                    current_val = x[i]
                    dists = np.abs(minima_candidates - current_val)
                    nearest_idx = np.argmin(dists)
                    target_val = minima_candidates[nearest_idx]
                    
                    if abs(current_val - target_val) < 1e-6:
                        continue # Already there
                    
                    x_new = x.copy()
                    x_new[i] = target_val
                    f_new = problem(x_new)
                    used += 1
                    
                    # Accept if it improves f2 (g) or is non-dominated
                    # Since we are minimizing f2 primarily for the "g" part, and f1 is fixed (x1 unchanged)
                    # If f_new[1] < f[1], it's an improvement in the g direction.
                    if f_new[1] < f[1]:
                        x = x_new
                        f = f_new
                        update_archive(f, x)
                    # Else, keep original x for this variable, but maybe the other variables are still worth trying?
                    # If we don't accept, we revert x[i] to original for subsequent tries?
                    # Yes, x is not modified if we don't accept.

        return np.array(archive_X) if archive_X else np.array([])

# EVOLVE-BLOCK-END
