"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer: Tchebycheff decomposition with local search."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290.0
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Number of weight vectors for decomposition
        # For 2 objectives, we can use a 1D grid
        n_weights = min(50, max(20, budget // 4))
        
        # Generate weight vectors for 2 objectives: lambda1 + lambda2 = 1
        # Use diverse weights from 0.05 to 0.95
        weights = np.linspace(0.05, 0.95, n_weights)
        W = np.column_stack([weights, 1.0 - weights])
        
        # Initial archive: empty list of (x, f)
        archive_X = []
        archive_F = []
        eval_count = 0

        def evaluate(x):
            nonlocal eval_count
            if eval_count >= budget:
                return None
            x = np.clip(x, lo, hi)
            f = np.array(problem(x))
            eval_count += 1
            return x, f

        def update_archive(x, f):
            """Add point to archive if it's not dominated and remove dominated points."""
            # Check if new point is dominated by any archive point
            for i in range(len(archive_F)):
                if np.all(archive_F[i] <= f) and np.any(archive_F[i] < f):
                    return  # New point is dominated
            
            # Remove archive points dominated by new point
            new_archive_X = []
            new_archive_F = []
            for i in range(len(archive_F)):
                if not (np.all(f <= archive_F[i]) and np.any(f < archive_F[i])):
                    new_archive_X.append(archive_X[i])
                    new_archive_F.append(archive_F[i])
            
            new_archive_X.append(x)
            new_archive_F.append(f)
            
            archive_X[:] = new_archive_X
            archive_F[:] = new_archive_F

        # Initialize: evaluate one point per weight vector
        for i in range(n_weights):
            if eval_count >= budget or time.time() > deadline:
                break
            # Initialize at random point or center
            x = lo + rng.random(dim) * span
            res = evaluate(x)
            if res is None:
                break
            x, f = res
            update_archive(x, f)
        
        if not archive_F:
            return np.array([])

        # Main optimization loop: for each weight vector, do local search
        # We have remaining budget for local search
        remaining = budget - eval_count
        if remaining <= 0:
            return np.array(archive_F) if archive_F else np.array([])

        # Adaptive step size based on objective spread
        if len(archive_F) > 1:
            f_arr = np.array(archive_F)
            obj_spread = np.ptp(f_arr, axis=0)
            base_step = 0.1 * np.where(obj_spread > 1e-10, 1.0, 0.1)
        else:
            base_step = 0.1 * np.ones(n_obj)

        # For each weight vector, perform a few local search steps
        # Distribute remaining budget among weight vectors
        steps_per_weight = max(1, remaining // n_weights)
        
        # Sort weights to explore systematically
        weight_indices = list(range(n_weights))
        rng.shuffle(weight_indices)
        
        for wi in weight_indices:
            if eval_count >= budget or time.time() > deadline:
                break
            
            w = W[wi]
            
            # Find best archive point for this weight vector (Tchebycheff scalarization)
            # min_i max_j (w_j * |f_j(x) - z*_j|)
            # z* is the ideal point (minimum of each objective)
            if len(archive_F) > 0:
                f_arr = np.array(archive_F)
                z_star = f_arr.min(axis=0)
                
                # Find best point for this weight
                best_idx = 0
                best_scalar = np.inf
                for i in range(len(archive_F)):
                    scalar = np.max(w * np.abs(f_arr[i] - z_star))
                    if scalar < best_scalar:
                        best_scalar = scalar
                        best_idx = i
                
                current_x = archive_X[best_idx].copy()
                current_f = archive_F[best_idx].copy()
            else:
                current_x = lo + 0.5 * span
                res = evaluate(current_x)
                if res is None:
                    break
                current_x, current_f = res
                update_archive(current_x, current_f)
                continue

            # Local search: try a few perturbations
            for step in range(steps_per_weight):
                if eval_count >= budget or time.time() > deadline:
                    break
                
                # Adaptive step size: shrink as we progress
                progress = eval_count / budget
                step_size = base_step * (1.0 - 0.5 * progress)
                
                # Gaussian perturbation
                x_new = current_x + step_size * span * rng.standard_normal(dim)
                x_new = np.clip(x_new, lo, hi)
                
                res = evaluate(x_new)
                if res is None:
                    break
                x_new, f_new = res
                
                # Check if improvement for this weight vector
                if len(archive_F) > 0:
                    f_arr = np.array(archive_F)
                    z_star = f_arr.min(axis=0)
                    scalar_old = np.max(w * np.abs(current_f - z_star))
                    scalar_new = np.max(w * np.abs(f_new - z_star))
                    
                    if scalar_new < scalar_old:
                        current_x = x_new
                        current_f = f_new
                        update_archive(x_new, f_new)
                else:
                    update_archive(x_new, f_new)
                    current_x = x_new
                    current_f = f_new

        # Return the archive (all non-dominated points found)
        if archive_F:
            return np.array(archive_F)
        return np.array([])
# EVOLVE-BLOCK-END
