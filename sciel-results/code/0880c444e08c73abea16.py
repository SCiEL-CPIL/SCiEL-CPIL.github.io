# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Derivative-free optimizer using a Local Quadratic Model (LQM) approach.
    This method fits a quadratic surrogate to the function values in a local 
    neighborhood and minimizes that surrogate to propose the next step.
    This is particularly effective for anisotropic functions like Bent Cigar 
    as it captures the local curvature (Hessian) without needing gradients.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 290.0
        dim = self.dim
        budget = self.budget
        calls = 0
        best_x = np.zeros(dim)
        best_f = float('inf')
        
        # Wrapper to count calls and handle issues
        def wrapper(x):
            nonlocal calls
            if calls >= budget:
                return float('inf')
            if time.time() > deadline:
                return float('inf')
            calls += 1
            try:
                return float(func(x))
            except Exception:
                return float('inf')

        # Initial point
        x_center = np.zeros(dim)
        f_center = wrapper(x_center)
        best_f = f_center
        best_x = x_center.copy()
        
        # Trust region radius
        # Bent Cigar has a very flat direction. We start with a moderate radius.
        radius = 1.0 
        
        # Number of points to sample for the quadratic model
        # For a full quadratic in d dims, we need 1 + d + d(d+1)/2 points.
        # For d=5, that's 1 + 5 + 15 = 21 points.
        # We add a few extra for robustness/noise handling.
        n_model_points = 2 * dim + 1  # 11 points for d=5? 
        # Actually, 2d+1 is the minimum for a diagonal quadratic. 
        # For a full quadratic, we need 21. Let's use 25 to be safe.
        n_model_points = 25

        rng = np.random.default_rng(42)

        while calls < budget and time.time() < deadline:
            if time.time() > deadline:
                break
            
            # 1. Generate sample points around x_center
            # We use the center, +/- axes, and random points on a sphere
            points = [x_center]
            for i in range(dim):
                for sign in [1, -1]:
                    p = x_center.copy()
                    p[i] += sign * radius
                    points.append(p)
            
            # Add random points to ensure full quadratic coverage
            while len(points) < n_model_points:
                # Random direction on sphere
                v = rng.normal(size=dim)
                v /= (np.linalg.norm(v) + 1e-10)
                p = x_center + v * radius
                points.append(p)
            
            # Evaluate all points
            values = []
            for p in points:
                # Clip to bounds [-5, 5]
                p_clipped = np.clip(p, -5, 5)
                val = wrapper(p_clipped)
                values.append(val)
                
                # Update best
                if val < best_f:
                    best_f = val
                    best_x = p_clipped.copy()
            
            # 2. Fit quadratic model
            # Model: f(x) = c + g^T(x - x_center) + 0.5 * (x - x_center)^T H (x - x_center)
            # We fit coefficients for the expanded form relative to x_center.
            # Let z = x - x_center.
            # f(z) = c + sum(g_i z_i) + sum(h_ii z_i^2) + sum(h_ij z_i z_j for i<j)
            
            # Construct design matrix A
            # Rows correspond to points, columns to parameters:
            # 1 (const), z_1..z_d (linear), z_1^2..z_d^2 (quad diag), z_i z_j (quad off-diag)
            
            d = dim
            n_params = 1 + d + d + d*(d-1)//2
            A = np.zeros((len(points), n_params))
            y = np.array(values)
            
            for idx, p in enumerate(points):
                z = p - x_center
                A[idx, 0] = 1.0
                for i in range(d):
                    A[idx, 1 + i] = z[i]
                for i in range(d):
                    A[idx, 1 + d + i] = z[i] * z[i]
                # Off-diagonal terms
                col_idx = 1 + d + d
                for i in range(d):
                    for j in range(i + 1, d):
                        A[idx, col_idx] = z[i] * z[j]
                        col_idx += 1
            
            # Solve least squares A * theta = y
            try:
                theta, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)
            except Exception:
                # If fit fails, shrink radius and continue
                radius *= 0.5
                continue

            # Extract coefficients
            c = theta[0]
            g = theta[1 : 1 + d]
            h_diag = theta[1 + d : 1 + 2 * d]
            h_off = theta[1 + 2 * d : ]
            
            # Reconstruct Hessian H
            H = np.zeros((d, d))
            for i in range(d):
                H[i, i] = h_diag[i]
            col_idx = 0
            for i in range(d):
                for j in range(i + 1, d):
                    val = h_off[col_idx]
                    H[i, j] = val
                    H[j, i] = val
                    col_idx += 1
            
            # 3. Minimize the quadratic model
            # We want to minimize q(z) = c + g^T z + 0.5 z^T H z
            # Gradient: g + H z = 0  =>  z = -H^-1 g
            # If H is singular, use pseudo-inverse or regularize
            
            # Regularize H to ensure positive definiteness if possible, or just use pinv
            # Add small ridge for stability
            H_reg = H + 1e-6 * np.eye(d)
            
            try:
                z_step = np.linalg.solve(H_reg, -g)
            except np.linalg.LinAlgError:
                z_step = np.linalg.pinv(H_reg) @ (-g)
            
            # 4. Check if step is within trust region
            step_norm = np.linalg.norm(z_step)
            if step_norm > radius:
                # Project step to boundary of trust region
                if step_norm > 1e-10:
                    z_step = z_step * (radius / step_norm)
                else:
                    z_step = np.zeros(d)
            
            x_new = x_center + z_step
            x_new = np.clip(x_new, -5, 5)
            
            # 5. Evaluate new point
            f_new = wrapper(x_new)
            
            if f_new < f_center:
                # Accept step
                x_center = x_new
                f_center = f_new
                # Keep or slightly expand radius if improvement is significant
                # For Bent Cigar, we might want to expand in flat directions, 
                # but simple expansion is safer.
                # Let's keep radius constant or slightly increase.
                radius *= 1.1 
            else:
                # Reject step, shrink radius
                radius *= 0.7
                
            # Safety: ensure radius doesn't become too small
            if radius < 1e-4:
                # If radius is tiny, we are stuck. Try a random kick?
                # Or just break.
                # Let's try a random kick to escape local minima/plateaus
                kick = rng.normal(size=d) * 0.1
                x_kick = x_center + kick
                x_kick = np.clip(x_kick, -5, 5)
                f_kick = wrapper(x_kick)
                if f_kick < f_center:
                    x_center = x_kick
                    f_center = f_kick
                    best_f = min(best_f, f_kick)
                    best_x = x_kick.copy()
                radius = 0.5
                continue

        return best_x, best_f
# EVOLVE-BLOCK-END
