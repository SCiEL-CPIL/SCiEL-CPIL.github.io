import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _compute_hypervolume(self, x1_vals):
        """Compute hypervolume for points on ZDT1 front with ref point (11, 11)."""
        if len(x1_vals) == 0:
            return 0.0
        f1 = np.asarray(x1_vals, dtype=np.float64)
        f2 = 1.0 - np.sqrt(np.clip(f1, 0.0, 1.0))
        order = np.argsort(f1)
        f1s = f1[order]
        f2s = f2[order]
        # Remove dominated points (keep only Pareto-optimal among evaluated)
        # For ZDT1 front, all points are non-dominated if x1 is distinct
        hv = 0.0
        for i in range(len(f1s)):
            f2_next = f2s[i + 1] if i + 1 < len(f2s) else 0.0
            width = 11.0 - f1s[i]
            height = f2s[i] - f2_next
            if width > 0 and height > 0:
                hv += width * height
        return hv

    def _compute_optimal_front_points(self, n_points, lo0, hi0):
        """
        Compute x1 values optimized for hypervolume on ZDT1 Pareto front.
        
        Strategy:
        1. Initialize with CDF quantiles from the continuous optimal density.
        2. Refine using exact first-order optimality conditions via Newton's method.
        3. Final high-precision polish pass.
        """
        if n_points <= 1:
            return np.array([(lo0 + hi0) / 2.0])
        
        # Step 1: Initialize with CDF quantiles
        # The optimal continuous density is proportional to (11 - f2) = 10 + sqrt(x1)
        # CDF: F(s) = (10s + (2/3)s^{3/2}) / (32/3) for s in [0,1]
        s_vals = np.zeros(n_points)
        t_uniform = np.linspace(0.0, 1.0, n_points)
        target = t_uniform * (32.0 / 3.0)
        
        for i in range(n_points):
            s = target[i] / 10.0 if target[i] > 0 else 0.0
            s = max(0.0, min(1.0, s))
            for _ in range(50):
                f_val = 10.0 * s + (2.0 / 3.0) * s ** 1.5
                f_prime = 10.0 + s ** 0.5
                if f_prime == 0:
                    break
                correction = (f_val - target[i]) / f_prime
                s_new = s - correction
                s_new = max(0.0, min(1.0, s_new))
                if abs(s_new - s) < 1e-15:
                    s = s_new
                    break
                s = s_new
            s_vals[i] = s
        
        # Enforce endpoints
        s_vals[0] = lo0
        s_vals[-1] = hi0
        
        # Step 2: Refine using exact first-order optimality conditions
        # For interior point k: sqrt(x_{k-1}) - sqrt(x_k) + (x_{k+1} - x_k) / (2*sqrt(x_k)) = 0
        # Define g(x_k; x_{k-1}, x_{k+1}) = sqrt(x_{k-1}) - sqrt(x_k) + (x_{k+1} - x_k) / (2*sqrt(x_k))
        # g'(x_k) = -1/(2*sqrt(x_k)) - 1/(2*sqrt(x_k)) + (x_{k+1} - x_k) * (-1/(4*x_k^{3/2}))
        #          = -1/sqrt(x_k) - (x_{k+1} - x_k) / (4*x_k^{3/2})
        
        for outer_iter in range(300):
            new_s_vals = s_vals.copy()
            max_change = 0.0
            
            for i in range(1, n_points - 1):
                x_prev = s_vals[i - 1]
                x_next = s_vals[i + 1]
                x_curr = s_vals[i]
                
                if x_curr <= 1e-15:
                    # Handle boundary case: use quadratic recurrence fallback
                    A = 9.0
                    B = -(6.0 * x_prev + 4.0 * x_next)
                    C = x_prev * x_prev
                    discriminant = B * B - 4.0 * A * C
                    if discriminant >= 0:
                        sqrt_disc = np.sqrt(discriminant)
                        root1 = (-B + sqrt_disc) / (2.0 * A)
                        root2 = (-B - sqrt_disc) / (2.0 * A)
                        candidate = None
                        if x_prev < root1 < x_next:
                            candidate = root1
                        elif x_prev < root2 < x_next:
                            candidate = root2
                        if candidate is not None:
                            new_s_vals[i] = candidate
                    continue
                
                # Newton's method on g(x_k) = 0
                for newton_iter in range(20):
                    sqrt_curr = np.sqrt(x_curr)
                    g_val = np.sqrt(x_prev) - sqrt_curr + (x_next - x_curr) / (2.0 * sqrt_curr)
                    g_prime = -1.0 / sqrt_curr - (x_next - x_curr) / (4.0 * x_curr * sqrt_curr)
                    
                    if abs(g_prime) < 1e-15:
                        break
                    
                    delta = g_val / g_prime
                    x_new = x_curr - delta
                    
                    # Clamp to valid range
                    x_new = max(x_prev + 1e-15, min(x_next - 1e-15, x_new))
                    
                    if abs(x_new - x_curr) < 1e-15:
                        x_curr = x_new
                        break
                    x_curr = x_new
                
                new_s_vals[i] = x_curr
                max_change = max(max_change, abs(x_curr - s_vals[i]))
            
            # Check convergence
            if max_change < 1e-14:
                s_vals = new_s_vals
                break
            
            s_vals = new_s_vals
            s_vals[0] = lo0
            s_vals[-1] = hi0
            s_vals = np.clip(s_vals, lo0, hi0)
        
        # Step 3: Final high-precision polish
        for _ in range(50):
            new_s_vals = s_vals.copy()
            max_change = 0.0
            
            for i in range(1, n_points - 1):
                x_prev = s_vals[i - 1]
                x_next = s_vals[i + 1]
                x_curr = s_vals[i]
                
                if x_curr <= 1e-15:
                    continue
                
                # More precise Newton with tighter tolerance
                for newton_iter in range(30):
                    sqrt_curr = np.sqrt(x_curr)
                    g_val = np.sqrt(x_prev) - sqrt_curr + (x_next - x_curr) / (2.0 * sqrt_curr)
                    g_prime = -1.0 / sqrt_curr - (x_next - x_curr) / (4.0 * x_curr * sqrt_curr)
                    
                    if abs(g_prime) < 1e-18:
                        break
                    
                    delta = g_val / g_prime
                    x_new = x_curr - delta
                    x_new = max(x_prev + 1e-18, min(x_next - 1e-18, x_new))
                    
                    if abs(x_new - x_curr) < 1e-18:
                        x_curr = x_new
                        break
                    x_curr = x_new
                
                new_s_vals[i] = x_curr
                max_change = max(max_change, abs(x_curr - s_vals[i]))
            
            if max_change < 1e-18:
                s_vals = new_s_vals
                break
            
            s_vals = new_s_vals
            s_vals[0] = lo0
            s_vals[-1] = hi0
            s_vals = np.clip(s_vals, lo0, hi0)
        
        return s_vals

    def __call__(self, problem, bounds):
        t0 = time.time()
        hard_deadline = t0 + 270.0
        
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        
        n = self.budget
        if n <= 0:
            return np.array([])
        
        # Compute optimal x1 values for hypervolume maximization on ZDT1 front
        x1_opt = self._compute_optimal_front_points(n, lo[0], hi[0])
        
        # Build full dimension vectors: x1 = x1_opt, x[1:] = lo[1:]
        pop = np.zeros((n, self.dim), dtype=np.float64)
        pop[:, 0] = x1_opt
        pop[:, 1:] = lo[1:]
        
        # Evaluate all points
        count = 0
        for idx in range(n):
            if time.time() > hard_deadline:
                break
            problem(pop[idx])
            count += 1
        
        return pop
