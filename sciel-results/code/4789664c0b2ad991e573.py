"""Multi-objective optimizer for ZDT4 using Structure-Exploiting Global Search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Structure-Exploiting Global Search for ZDT4."""

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
        span = hi - lo

        def is_deadline_exceeded():
            return (time.time() - start_time) > deadline

        # Initialize on the known Pareto front structure:
        # For ZDT4, Pareto front is at g=1. 
        # g=1 when x2..x10 are at local minima of sin(10*pi*x).
        # Minima of sin(10*pi*x) on [0,1] are at x = (2k+1)/10 for k=0..4? No, sin(z) min at 3pi/2, 7pi/2...
        # z = 10*pi*x. sin(z) = -1 at z = 3pi/2, 7pi/2, 11pi/2, 15pi/2, 19pi/2.
        # x = 0.15, 0.35, 0.55, 0.75, 0.95.
        # Actually, any local minimum works. Let's pick one set of minima to initialize.
        # x = 0.15, 0.35, 0.55, 0.75, 0.95
        # Let's create 10 diverse base points with different x1 values.
        
        base_points = []
        for i in range(10):
            x = np.zeros(dim)
            x[0] = (i + 0.5) / 10.0  # Spread x1
            # Set x2..x10 to a known local minimum of sin(10*pi*x)
            # Let's use 0.15, 0.35, 0.55, 0.75, 0.95 repeating
            for j in range(1, dim):
                # Use a fixed pattern of minima
                x[j] = [0.15, 0.35, 0.55, 0.75, 0.95][j % 5]
            base_points.append(x)

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

        # Evaluate initial base points
        for x in base_points:
            if used >= budget or is_deadline_exceeded():
                break
            f = problem(x)
            update_archive(f, x)
            used += 1

        # Local search strategy:
        # For each base point, perform a local search to minimize g (and thus f2).
        # ZDT4's g is deceptive. We use a two-phase perturbation.
        # Phase 1: Larger steps to escape local minima.
        # Phase 2: Smaller steps to converge to g=1.
        
        # We will iterate through base points and spend some budget on each.
        # Budget allocation: 200 total. 10 init. 190 left.
        # 19 steps per base point? Or adaptive?
        # Let's do 15 steps per base point initially, then refine best ones.
        
        steps_per_point = 15
        
        # Phase 1: Global exploration around base points
        for i in range(10):
            if used >= budget or is_deadline_exceeded():
                break
            x = base_points[i].copy()
            best_f = problem(x)
            best_x = x.copy()
            used += 1 # Wait, we already evaluated base_points. 
            # Actually, let's re-evaluate to be safe or just use the archive?
            # The archive has the best found so far. Let's use the base point's objective from archive if possible.
            # Simpler: just re-evaluate.
            
            # Wait, I evaluated base_points in the loop above. 
            # Let's store the objective values.
            # Let's restart the logic slightly to be cleaner.
            
        # Revised Strategy:
        # 1. Generate 10 base points on the front.
        # 2. Evaluate them.
        # 3. For each base point, run a local search with 10 evaluations.
        # 4. Collect all results in an archive.
        # 5. If budget remains, run local search on the best points in the archive.
        
        # Reset
        archive_F = []
        archive_X = []
        used = 0
        base_points = []
        for i in range(10):
            x = np.zeros(dim)
            x[0] = (i + 0.5) / 10.0
            for j in range(1, dim):
                x[j] = [0.15, 0.35, 0.55, 0.75, 0.95][j % 5]
            base_points.append(x)

        # Evaluate bases
        base_F = []
        for x in base_points:
            if used >= budget or is_deadline_exceeded():
                break
            f = problem(x)
            base_F.append(f)
            update_archive(f, x)
            used += 1

        # Local Search Function
        def local_search(x_start, f_start, budget_left, scale_start=0.05, scale_end=0.005):
            nonlocal used
            x = x_start.copy()
            f = f_start.copy()
            steps = 10
            for s in range(steps):
                if used >= budget or is_deadline_exceeded():
                    break
                # Interpolate scale
                scale = scale_start * (scale_end / scale_start) ** (s / max(1, steps - 1))
                # Random perturbation
                delta = np.random.randn(dim) * scale
                # Clip to bounds
                x_new = np.clip(x + delta, lo, hi)
                f_new = problem(x_new)
                used += 1
                
                # Update best
                if np.all(f_new <= f) and np.any(f_new < f):
                    x = x_new
                    f = f_new
                    update_archive(f, x)
            return x, f

        # Phase 1: Local search around each base point
        for i in range(10):
            if used >= budget or is_deadline_exceeded():
                break
            # Run local search
            local_search(base_points[i], base_F[i], budget - used)

        # Phase 2: Refine best points in archive
        # Sort archive by f2 (g) to find points closest to Pareto front
        if archive_F:
            # Convert to numpy for sorting
            F_arr = np.array(archive_F)
            X_arr = np.array(archive_X)
            # Sort by f2 (index 1)
            order = np.argsort(F_arr[:, 1])
            # Take top 5
            for idx in order[:5]:
                if used >= budget or is_deadline_exceeded():
                    break
                # Run a finer local search
                local_search(X_arr[idx], F_arr[idx], budget - used, scale_start=0.02, scale_end=0.001)

        return np.array(archive_X) if archive_X else np.array([])

# EVOLVE-BLOCK-END
