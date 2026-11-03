import numpy as np
from scipy.optimize import minimize
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def _covering_radius(points):
        # Compute the maximum angular distance from any point on the sphere to the nearest point in 'points'
        # This is equivalent to the maximum circumradius of the convex hull facets.
        # We approximate this by sampling the sphere densely and finding the max distance.
        # For speed, we can also use the convex hull if available, but sampling is robust.
        # Let's use a moderate grid for speed during optimization.
        # A 30x30 grid on the sphere is ~900 points, fast enough for L-BFGS.
        # Actually, for better precision, let's use a slightly denser grid or the convex hull.
        # Convex hull is exact but might be slower or fail if points are coplanar (unlikely here).
        # Let's stick to a dense sampling for robustness and speed.
        # 60x30 grid -> 1800 points.
        n_theta = 60
        n_phi = 30
        theta = np.linspace(0, np.pi, n_theta, endpoint=False)
        phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
        T, P = np.meshgrid(theta, phi)
        x = np.sin(T) * np.cos(P)
        y = np.sin(T) * np.sin(P)
        z = np.cos(T)
        samples = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T
        
        # Calculate distances from samples to all points
        # points: (N, 3), samples: (M, 3)
        # dist_sq = |s - p|^2 = |s|^2 + |p|^2 - 2 s.p = 2 - 2 s.p
        # angular dist = arccos(s.p)
        # We want max arccos(min_p s.p)
        # Let dot = s.p (M, N)
        # min_dot = min_p dot (M,)
        # max_angle = max arccos(min_dot)
        # To avoid arccos for speed, we can minimize max(1 - min_dot) or similar, 
        # but arccos is monotonic, so maximizing angle is minimizing min_dot.
        # So we want to minimize the maximum of (1 - min_dot) ? No, we want to minimize the angle.
        # angle = arccos(cos_theta). Minimizing angle is equivalent to maximizing cos_theta.
        # So we want to maximize min_p(s.p) for all s, and then take the max over s?
        # No, we want the smallest covering radius, i.e., min over config of max over s of arccos(min_p s.p).
        # This is equivalent to min over config of max over s of (1 - min_p s.p) ? No.
        # Let's just compute the angles.
        
        dots = samples @ points.T  # (M, N)
        min_dots = np.min(dots, axis=1)  # (M,)
        # Clip to [-1, 1] for numerical stability
        min_dots = np.clip(min_dots, -1.0, 1.0)
        angles = np.arccos(min_dots)
        return np.max(angles)

    def _objective_flat(x_flat):
        # x_flat: (3N,)
        points = x_flat.reshape(N, 3)
        # Normalize points to unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        points = points / norms
        return _covering_radius(points)

    # Initial guess: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z]).ravel()

    best_points = x0.reshape(N, 3)
    best_val = _objective_flat(x0)

    # Try multiple restarts with small perturbations if time permits
    # L-BFGS-B is local, so restarts are crucial.
    
    rng = np.random.default_rng(42)
    
    for restart in range(10):
        if time.time() > deadline - 10.0:
            break
            
        if restart == 0:
            x_start = x0
        else:
            # Add random noise to the best points found so far
            noise = rng.normal(0, 0.1, size=(N, 3))
            x_start = (best_points + noise).ravel()
        
        try:
            res = minimize(_objective_flat, x_start, method='L-BFGS-B', 
                           options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 2000})
            
            if res.success or res.fun < best_val:
                x_opt = res.x
                points_opt = x_opt.reshape(N, 3)
                norms = np.linalg.norm(points_opt, axis=1, keepdims=True)
                norms = np.where(norms < 1e-10, 1.0, norms)
                points_opt = points_opt / norms
                val = _covering_radius(points_opt)
                
                if val < best_val:
                    best_val = val
                    best_points = points_opt
        except Exception:
            pass

    # Final normalization
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    best_points = best_points / norms
    
    return best_points
