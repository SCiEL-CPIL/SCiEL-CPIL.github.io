"""Derivative-Free Trust-Region optimizer with local quadratic surrogates for BBOB f18."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DFTR optimizer: builds local quadratic models from directional probes."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 275  # safety margin before 300s kill

        best_x = np.zeros(dim)
        best_f = np.inf
        total_calls = 0

        def safe_call(x):
            nonlocal total_calls, best_x, best_f
            if total_calls >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            fx = float(func(x))
            total_calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # ---- Phase 1: Coarse exploration with stratified sampling ----
        # Use a small set of well-distributed points to find a good starting region
        n_explore = min(30, budget)
        # Generate stratified points: divide each dimension into 3 strata
        # and pick one random point per stratum combination, but only a subset
        n_strata_per_dim = 3
        # Generate a grid of n_strata_per_dim^dim points and randomly sample from them
        grid_coords = np.array(np.meshgrid(*[np.linspace(-5, 5, n_strata_per_dim) for _ in range(dim)])).T.reshape(-1, dim)
        # Add small random jitter to avoid exactly on-grid points
        jitter = np.random.randn(len(grid_coords), dim) * 0.5
        candidates = grid_coords + jitter
        np.random.shuffle(candidates)
        
        for pt in candidates[:n_explore]:
            if total_calls >= budget or time.time() > deadline:
                break
            safe_call(pt)

        if total_calls >= budget or time.time() > deadline:
            return best_x, best_f

        # ---- Phase 2: Derivative-Free Trust-Region with quadratic surrogates ----
        # Build a local quadratic model using directional derivatives
        # and solve for the trust-region subproblem
        
        x = best_x.copy()
        f_x = best_f
        trust_radius = 2.0  # initial trust radius
        min_radius = 1e-4
        max_radius = 5.0
        
        # Number of probe points per iteration
        n_probes = 2 * dim + 1  # 11 points for dim=5
        
        while total_calls < budget and time.time() < deadline:
            if trust_radius < min_radius:
                break
            
            # Generate probe points around x using a spherical design
            # For dim=5, use 2*dim+1 = 11 points: center + 5 forward + 5 backward
            # Actually, let's use a more efficient design
            
            # Generate probe directions
            # Use a simple approach: perturb each coordinate by ±trust_radius
            # This gives 2*dim + 1 = 11 points
            probe_points = [x.copy()]
            for d in range(dim):
                for sign in [1, -1]:
                    p = x.copy()
                    p[d] += sign * trust_radius
                    probe_points.append(p)
            
            # Evaluate probe points
            probe_vals = []
            for p in probe_points:
                fp = safe_call(p)
                probe_vals.append(fp)
                
                if total_calls >= budget or time.time() > deadline:
                    break
            
            if total_calls >= budget or time.time() > deadline:
                break
            
            # If any probe is inf, skip this iteration
            if any(np.isinf(v) for v in probe_vals):
                trust_radius *= 0.5
                continue
            
            # Build quadratic model from probe values
            # f(x + d) ≈ f_x + g^T d + 0.5 d^T H d
            # We have 11 equations (including center) and 1 + 5 + 10 = 16 unknowns
            # This is underdetermined, so we need a regularized approach
            
            # Alternative: use a simpler linear + quadratic model
            # Fit f(x + d) ≈ f_x + g^T d + 0.5 * sum_i H_ii * d_i^2 (diagonal Hessian)
            # This gives 1 + 5 + 5 = 11 unknowns, matching 11 equations
            
            # Set up linear system: A * params = b
            # params = [g_1, ..., g_5, H_11, ..., H_55]
            # For each probe point d:
            #   f(x+d) - f_x = g^T d + 0.5 * sum_i H_ii * d_i^2
            
            A = np.zeros((len(probe_points) - 1, 2 * dim))
            b = np.zeros(len(probe_points) - 1)
            
            for i in range(1, len(probe_points)):
                d = probe_points[i] - x
                for j in range(dim):
                    A[i - 1, j] = d[j]
                    A[i - 1, dim + j] = 0.5 * d[j] ** 2
                b[i - 1] = probe_vals[i] - f_x
            
            # Solve least squares
            try:
                params, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            except np.linalg.LinAlgError:
                trust_radius *= 0.5
                continue
            
            g = params[:dim]
            H_diag = params[dim:]
            
            # Check for improvement: move in direction of -g, scaled by trust radius
            # Simple steepest descent step within trust region
            g_norm = np.linalg.norm(g)
            if g_norm < 1e-12:
                # At a local minimum according to the model
                trust_radius *= 0.5
                continue
            
            # Step direction: -g / ||g||
            step_dir = -g / g_norm
            
            # Choose step size: min(trust_radius, ||g|| / max(|H_diag| + eps))
            # Use a simple heuristic
            h_max = np.max(np.abs(H_diag)) + 1e-8
            step_size = min(trust_radius, g_norm / (2 * h_max))
            step_size = max(step_size, trust_radius * 0.1)
            
            # Try the step
            x_new = x + step_size * step_dir
            f_new = safe_call(x_new)
            
            if total_calls >= budget or time.time() > deadline:
                break
            
            if np.isinf(f_new):
                trust_radius *= 0.5
                continue
            
            # Accept or reject based on actual vs predicted improvement
            predicted_improvement = -step_size * g_norm
            actual_improvement = f_x - f_new
            
            if actual_improvement > 0.1 * predicted_improvement:
                # Good step: accept and possibly expand trust region
                x = x_new
                f_x = f_new
                if actual_improvement > 0.5 * predicted_improvement:
                    trust_radius = min(trust_radius * 1.5, max_radius)
                else:
                    trust_radius *= 1.2
            else:
                # Bad step: shrink trust region
                trust_radius *= 0.5
            
            # Update best if improved
            if f_x < best_f:
                best_f = f_x
                best_x = x.copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END
