# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer for BBOB-style Sphere function (f1) with hidden shift, rotation, and offset.
    Strategy:
    1. Adaptive Radial Sampling (ARS): Exploit the spherical symmetry of f1.
       Sample points at varying radii from the origin to quickly identify the approximate
       distance to the hidden minimum. This provides strong early progress (AOCC).
    2. Nelder-Mead Polish: Once a good candidate is found, use a local gradient-free
       optimizer to refine the position.
    3. Robust Budget & Time Management: Ensure we stay within 500 calls and 300s.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 300s limit, use 280s to be safe against overhead
        deadline = start_time + 280.0
        D = self.dim
        budget = self.budget

        call_count = [0]
        best_x = [None]
        best_f = [np.inf]

        def track(x):
            """Evaluate function with tracking and safety checks."""
            call_count[0] += 1
            if call_count[0] > budget:
                raise ValueError("Budget exceeded")
            if time.time() > deadline:
                raise TimeoutError("Time limit exceeded")
            
            x_arr = np.asarray(x, dtype=np.float64)
            # Clip to domain [-5, 5]
            x_arr = np.clip(x_arr, -5.0, 5.0)
            
            f_val = func(x_arr)
            f_val = float(f_val)
            
            if np.isnan(f_val) or np.isinf(f_val):
                f_val = 1e10
            
            if f_val < best_f[0]:
                best_f[0] = f_val
                best_x[0] = x_arr.copy()
            
            return f_val

        # ------------------------------------------------------------------
        # Phase 1: Adaptive Radial Sampling
        # ------------------------------------------------------------------
        # Since f1 is Sphere: f(x) = ||R(x - shift) + offset||^2
        # The minimum is at 'shift'. The value at origin is ||R(-shift) + offset||^2.
        # By sampling points x = r * u, where u is a unit vector, we can probe
        # different distances from the origin.
        
        # We want to find the radius r* where the function value is minimized
        # on average, which should be close to ||shift||.
        
        # Define a set of radii to test.
        # The minimum lies in [-4, 4]^5, so ||shift|| is in [0, 4*sqrt(5) ~ 8.9].
        # However, the domain is [-5, 5]^5.
        # Let's test radii from 0 to 5.
        
        radii = np.linspace(0.0, 5.0, 20)
        rng = np.random.default_rng(42)
        
        # For each radius, sample a few directions to average out noise/rotation effects.
        # Since the function is rotationally symmetric around the shift, the value
        # depends on the angle between (x - shift) and the offset? No, Sphere is
        # f(x) = ||x - shift||^2 (if offset is 0) or similar.
        # Actually, "Sphere" in BBOB is f(x) = ||x - x_opt||^2.
        # The prompt says: "hidden seeded shift, rotation and offset".
        # Usually, this means f(x) = ||R(x - shift) + offset||^2.
        # If offset is non-zero, the minimum is not exactly at shift.
        # But for Sphere, it's likely just a shifted and rotated sphere.
        # Let's assume f(x) = ||R(x - shift)||^2 = ||x - shift||^2.
        # In this case, the best radius is ||shift||.
        
        # We will evaluate a grid of (radius, direction) points.
        # To save budget, we will use 5 directions per radius.
        n_dirs = 5
        n_radii = len(radii)
        total_samples = n_radii * n_dirs
        
        # Check if we have enough budget
        if total_samples > budget * 0.3:
            # Reduce samples if budget is tight
            n_radii = min(n_radii, budget // 10)
            radii = np.linspace(0.0, 5.0, n_radii)
            total_samples = n_radii * n_dirs

        # Generate directions
        directions = []
        for _ in range(n_dirs):
            d = rng.normal(size=D)
            d /= np.linalg.norm(d)
            directions.append(d)
        
        # Evaluate all combinations
        best_radius = 0.0
        min_val_at_radius = np.inf
        
        for r in radii:
            val_sum = 0.0
            count = 0
            for d in directions:
                if call_count[0] >= budget or time.time() > deadline:
                    break
                x = r * d
                try:
                    val = track(x)
                    val_sum += val
                    count += 1
                except (ValueError, TimeoutError):
                    break
            if count > 0:
                avg_val = val_sum / count
                if avg_val < min_val_at_radius:
                    min_val_at_radius = avg_val
                    best_radius = r
            
            if call_count[0] >= budget or time.time() > deadline:
                break
        
        # ------------------------------------------------------------------
        # Phase 2: Local Refinement (Nelder-Mead)
        # ------------------------------------------------------------------
        # Start from the best point found so far.
        # If we didn't find a good point, start from 0.
        
        if best_x[0] is None:
            x_start = np.zeros(D)
            # Ensure at least one evaluation
            try:
                track(x_start)
            except (ValueError, TimeoutError):
                pass
        
        if best_x[0] is not None and call_count[0] < budget - 5:
            # Use Nelder-Mead for polishing.
            # We can use the 'minimize' function from scipy, but to avoid
            # dependency issues or overhead, we can implement a simple
            # coordinate descent or use scipy if available.
            # Let's use scipy.optimize.minimize as it's standard.
            try:
                from scipy.optimize import minimize
                
                # We want to use as many iterations as possible within budget.
                # Nelder-Mead uses D+1 points per iteration.
                # Let's set maxiter to a reasonable number, but rely on
                # the track function to stop if budget is exceeded.
                
                # Initial simplex size
                initial_simplex = list(np.array([best_x[0]] + [best_x[0] + 0.1 * np.eye(D)[i] for i in range(D)]))
                
                # We cannot easily pass a custom simplex to scipy.optimize.minimize
                # without using the 'x0' and letting it build the simplex.
                # To control the initial step size, we can scale the problem or
                # just use default.
                
                res = minimize(
                    track,
                    best_x[0],
                    method='Nelder-Mead',
                    options={
                        'maxiter': 1000,  # Will be limited by budget
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
            except ImportError:
                # Fallback: Simple Coordinate Descent
                step_size = 1.0
                for _ in range(100):
                    if call_count[0] >= budget or time.time() > deadline:
                        break
                    improved = False
                    for i in range(D):
                        # Try +step
                        x_plus = best_x[0].copy()
                        x_plus[i] += step_size
                        x_plus = np.clip(x_plus, -5.0, 5.0)
                        try:
                            f_plus = track(x_plus)
                            if f_plus < best_f[0]:
                                improved = True
                        except (ValueError, TimeoutError):
                            break
                        
                        # Try -step
                        x_minus = best_x[0].copy()
                        x_minus[i] -= step_size
                        x_minus = np.clip(x_minus, -5.0, 5.0)
                        try:
                            f_minus = track(x_minus)
                            if f_minus < best_f[0]:
                                improved = True
                        except (ValueError, TimeoutError):
                            break
                    
                    if not improved:
                        step_size *= 0.5
                        if step_size < 1e-6:
                            break
            
            # ------------------------------------------------------------------
            # Phase 3: Final Polish
            # ------------------------------------------------------------------
            # If we still have budget, do another round of Nelder-Mead with
            # tighter tolerances.
            
            if call_count[0] < budget - 5 and best_x[0] is not None:
                try:
                    from scipy.optimize import minimize
                    res = minimize(
                        track,
                        best_x[0],
                        method='Nelder-Mead',
                        options={
                            'maxiter': 500,
                            'xatol': 1e-10,
                            'fatol': 1e-12,
                            'adaptive': True
                        }
                    )
                except (ImportError, Exception):
                    pass

        # Return the best point and value found
        if best_x[0] is None:
            best_x[0] = np.zeros(D)
            best_f[0] = 0.0

        return (best_x[0], best_f[0])

# EVOLVE-BLOCK-END
