"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    start_time = time.time()
    # Evaluator kills at 150s. Keep a safety margin.
    deadline = start_time + 145.0

    def time_left():
        return deadline - time.time()

    def norm(vec):
        return np.linalg.norm(vec)

    def safe_F(x):
        try:
            y = F(x)
            y = np.asarray(y, dtype=float)
            if y.shape != (2,) or not np.all(np.isfinite(y)):
                return None
            return y
        except Exception:
            return None

    def objective(x):
        Fx = safe_F(x)
        if Fx is None:
            return np.inf
        return 0.5 * np.dot(Fx, Fx)

    def is_good(x, Fx=None):
        if not np.all(np.isfinite(x)):
            return False
        if Fx is None:
            Fx = safe_F(x)
        if Fx is None:
            return False
        return norm(Fx) <= 1e-8

    # 1. Initial Evaluation
    x0 = np.asarray(x0, dtype=float)
    if not np.all(np.isfinite(x0)):
        x0 = np.array([0.0, 0.0])
        
    Fx0 = safe_F(x0)
    if Fx0 is not None and norm(Fx0) <= 1e-8:
        return x0

    # Powell's Method Implementation
    # This method is derivative-free and works on the potential Phi(x) = 0.5 ||F(x)||^2
    def powell_method(x_start, max_iter=100):
        x = np.array(x_start, dtype=float)
        # Initial directions: standard basis
        directions = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
        
        # Initialize with a small step to ensure we don't start exactly on a gradient
        # but actually Powell usually starts with unit vectors and minimizes along them.
        # We need to be careful with the initial step size. 
        # Standard Powell minimizes along each direction.
        
        f_curr = objective(x)
        if np.isinf(f_curr):
            return x, f_curr, 0

        calls = 1 # for initial f_curr if we count it, but x_start was already evaluated? 
                  # Actually, x_start was evaluated in the caller or here? 
                  # Let's assume x_start is fresh or we re-evaluate. 
                  # To be safe, we evaluate x_start here.
        
        best_x = x.copy()
        best_f = f_curr
        best_Fx = safe_F(x)
        
        for _ in range(max_iter):
            if time_left() < 0.5:
                break
                
            f_old = f_curr
            min_f = f_curr
            min_idx = 0
            
            # 1. Minimize along each direction
            for i in range(len(directions)):
                # Find minimum along direction i
                x_min, f_min, calls_i = minimize_1d(x, directions[i], f_curr)
                calls += calls_i
                if f_min < min_f:
                    min_f = f_min
                    min_idx = i
                x = x_min
                f_curr = f_min
                
            # Check for convergence
            if abs(f_old - f_curr) < 1e-14 * max(1.0, abs(f_old)):
                break

            # 2. Move to the new point
            x_next = x.copy()
            f_next = f_curr

            # 3. Move along the new direction (x_next - x_start_of_iteration)
            # The new direction is the vector from the start of the iteration to the end
            # Actually, standard Powell:
            # x_new = x + 2*(x_min - x_prev) - x_prev_dir_min ... this is getting complex.
            # Let's use the standard update:
            # New direction d_new = x_curr - x_start_iter
            # Check if moving along d_new improves further.
            
            x_start_iter = x_start.copy() # Wait, x_start changes every iteration in Powell?
            # No, in Powell's method, the set of directions is updated.
            # Let's stick to a robust implementation.
            
            # Standard Powell Step:
            # 1. Minimize along each direction d_i to get x_i.
            # 2. Let x_{n} be the point with min f.
            # 3. Compute new direction d_new = x_n - x_{n-1} (where x_{n-1} is the point before the last step? No.)
            #    Actually, d_new = x_n - x_{start_of_iteration} is not right.
            #    d_new is the vector from the initial point of the cycle to the final point of the cycle.
            
            # Let's restart the logic for clarity.
            break # The above loop structure was slightly off for standard Powell. 
                  # Let's implement a cleaner version.
            
        return best_x, best_f, calls

    # Cleaner 1D minimization along a direction
    def minimize_1d(x_base, direction, f_base):
        # Golden section search or simple parabolic? 
        # For robustness, let's use a simple adaptive step search.
        # Since we are in 2D, we can afford a few evaluations.
        # Let's try a simple line search: try steps 1, 0.5, 2, etc.
        # But we need to know the scale.
        # A better 1D minimizer for this context: 
        # Try a small step, if it decreases, try larger, else smaller.
        
        step = 0.1
        x_curr = x_base + step * direction
        f_curr = objective(x_curr)
        calls = 1
        
        if np.isinf(f_curr):
            return x_base, f_base, calls

        if f_curr < f_base:
            # Try to extend
            prev_x = x_base
            prev_f = f_base
            for _ in range(10):
                x_new = x_base + 2 * step * direction
                f_new = objective(x_new)
                calls += 1
                if f_new < f_curr:
                    prev_x = x_curr
                    prev_f = f_curr
                    x_curr = x_new
                    f_curr = f_new
                    step *= 2
                else:
                    break
            # We have a bracket or a local min. 
            # For simplicity and speed, return the best found.
            if f_new < f_curr: # This logic is getting messy. 
                pass
        else:
            # Try smaller step
            step = 0.01
            x_curr = x_base + step * direction
            f_curr = objective(x_curr)
            calls += 1
            if f_curr < f_base:
                 pass # Found a better point
            
        # This simple 1D search is weak. 
        # Let's use a proper 1D minimizer: Brent's method is overkill if we don't have scipy? 
        # We have scipy.
        
        return x_curr, f_curr, calls

    # Let's scrap the manual Powell and use scipy.optimize.minimize with Powell method?
    # The prompt says SciPy is available.
    # "Strategy to try: ... Powell's method ... avoids instability ... leverages 2D structure"
    # Using scipy.optimize.minimize(method='Powell') is the most robust way to implement this.
    
    def run_powell_scipy(x_start, max_iter=200):
        try:
            from scipy.optimize import minimize
            # Define a wrapper that handles None returns
            def obj_func(x):
                val = objective(x)
                if np.isinf(val):
                    return 1e10
                return val
            
            res = minimize(obj_func, x_start, method='Powell',
                           options={'maxiter': max_iter, 'xtol': 1e-10, 'ftol': 1e-12})
            
            if res.x is not None and np.all(np.isfinite(res.x)):
                Fx_res = safe_F(res.x)
                if Fx_res is not None:
                    return res.x, Fx_res
            return x_start, safe_F(x_start)
        except Exception:
            return x_start, safe_F(x_start)

    # Multi-start Strategy
    # Generate diverse starting points
    scale = max(1.0, norm(x0))
    rng = np.random.default_rng(42)
    
    start_points = [x0, np.array([0.0, 0.0])]
    
    # Add points on spheres
    for r in [scale * 0.1, scale * 0.5, scale * 1.0, scale * 2.0, scale * 5.0, scale * 10.0]:
        for _ in range(4):
            u = rng.normal(size=2)
            u /= norm(u)
            start_points.append(x0 + r * u)
            start_points.append(r * u)
            
    # Deduplicate
    unique_points = []
    for p in start_points:
        if not np.all(np.isfinite(p)):
            continue
        is_dup = False
        for up in unique_points:
            if norm(p - up) < 1e-6:
                is_dup = True
                break
        if not is_dup:
            unique_points.append(p)

    best_x = x0
    best_Fx = Fx0
    best_norm = norm(Fx0) if Fx0 is not None else np.inf

    for sp in unique_points:
        if time_left() < 1.0:
            break
        
        # Run Powell
        x_res, Fx_res = run_powell_scipy(sp)
        
        if Fx_res is not None:
            n_res = norm(Fx_res)
            if n_res < best_norm:
                best_norm = n_res
                best_x = x_res
                best_Fx = Fx_res
                if best_norm <= 1e-8:
                    return best_x

    return best_x

# EVOLVE-BLOCK-END
