import numpy as np
import scipy.optimize as opt
import time

N = 300

def construct():
    start_time = time.time()
    best_points = None
    best_radius = 0.0

    # Define objective and gradient
    def objective_and_grad(x_flat):
        x = x_flat.reshape(-1, 2)
        # Compute pairwise distances
        # x: (N, 2)
        # dist: (N, N)
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dist, np.inf)
        
        # Boundary distances
        radii = np.linalg.norm(x, axis=1)
        bound_dist = 2 * (1 - radii)
        
        # Find global minimum
        min_pair_dist = np.min(dist)
        min_bound_dist = np.min(bound_dist)
        
        min_val = min(min_pair_dist, min_bound_dist)
        
        # Gradient
        grad_flat = np.zeros_like(x_flat)
        x_grad = np.zeros_like(x)
        
        if min_pair_dist < min_bound_dist:
            # Find pair
            idx = np.unravel_index(np.argmin(dist), dist.shape)
            i, j = idx
            # dist[i,j] is the min
            d = dist[i, j]
            if d < 1e-10:
                # Degenerate, add small noise
                return -min_val, grad_flat
            v = (x[i] - x[j]) / d
            x_grad[i] -= v
            x_grad[j] += v
        else:
            # Find point
            i = np.argmin(bound_dist)
            r = radii[i]
            if r < 1e-10:
                # At center, gradient undefined, push out
                x_grad[i] = np.array([1.0, 0.0]) 
            else:
                x_grad[i] = 2 * x[i] / r
        
        grad_flat = x_grad.flatten()
        return -min_val, grad_flat

    def run_optimization(initial_points, max_iter=1000, timeout=20):
        nonlocal best_points, best_radius
        x_flat = initial_points.flatten()
        
        def wrapper(x):
            val, grad = objective_and_grad(x)
            return val, grad
            
        # L-BFGS-B
        try:
            res = opt.minimize(wrapper, x_flat, method='L-BFGS-B', jac=True, options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.success or res.fun < 0: # res.fun is -min_dist
                new_radius = -res.fun / 2.0
                if new_radius > best_radius:
                    best_radius = new_radius
                    best_points = res.x.reshape(-1, 2).copy()
        except Exception:
            pass

    # Generate initializations
    inits = []
    
    # 1. Hexagonal
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    # ... generate hex points ...
    # Need to ensure exactly N points.
    # If more, take closest to center.
    # If fewer, this init is invalid for direct use, but we can pad? No, must be N.
    # The seed logic handles this. We can reuse a similar logic or just use random if count != N.
    
    # Let's write a helper to get N points from a grid
    def get_hex_points(n):
        spacing = 2.0 / (np.sqrt(n) + 1.0)
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
            if len(points) >= n:
                return points[np.argsort(radius, kind="stable")[:n]]
            spacing *= 0.98
            
    inits.append(get_hex_points(N))
    
    # 2. Spiral
    theta = np.linspace(0, 100 * np.pi, N)
    r = np.linspace(0, 1, N) ** 0.5
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    inits.append(np.column_stack((x, y)))
    
    # 3. Random
    rng = np.random.default_rng(42)
    r = np.sqrt(rng.random(N))
    theta = rng.random(N) * 2 * np.pi
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    inits.append(np.column_stack((x, y)))
    
    # 4. Concentric Rings
    # Approximate number of points on ring k is proportional to r_k
    # r_k ~ sqrt(k)
    # Let's just distribute points evenly in area, then project to rings?
    # Simpler: use a known heuristic for ring packing
    # Or just use random with a different seed.
    
    # Run optimizations
    for i, init in enumerate(inits):
        if time.time() - start_time > 100:
            break
        run_optimization(init, timeout=20)
        
    # If no best found, return seed-like
    if best_points is None:
        return get_hex_points(N)
        
    return best_points
