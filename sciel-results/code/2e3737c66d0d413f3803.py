import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull

N = 13

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    time_limit = 148.0  # Leave 2s margin for safety

    # Precompute indices for all triples as numpy arrays for vectorized computation
    tri_i = []
    tri_j = []
    tri_k = []
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                tri_i.append(i)
                tri_j.append(j)
                tri_k.append(k)
    tri_i = np.array(tri_i, dtype=np.int32)
    tri_j = np.array(tri_j, dtype=np.int32)
    tri_k = np.array(tri_k, dtype=np.int32)
    n_triples = len(tri_i)
    
    def compute_score(points):
        """Compute min triangle area / convex hull area for given points."""
        try:
            hull = ConvexHull(points)
            area_hull = hull.volume
        except Exception:
            return 0.0, 0.0
            
        if area_hull < 1e-9:
            return 0.0, 0.0

        x = points[:, 0]
        y = points[:, 1]
        # Vectorized triangle area computation
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = np.abs(v1x * v2y - v1y * v2x)
        areas = 0.5 * cross
        min_area = np.min(areas)

        score = min_area / area_hull
        return score, area_hull

    def _maxmin_penalty(points, t_target, gamma=1.0):
        """
        Penalty-based objective: minimize penalty for triangles with area < t_target.
        Uses a smooth penalty: sum of (t_target - area)^2 for areas below target.
        Also includes a small term to maximize hull area for normalization benefit.
        """
        try:
            hull = ConvexHull(points)
            area_hull = hull.volume
        except Exception:
            return 1e6
            
        if area_hull < 1e-9:
            return 1e6

        x = points[:, 0]
        y = points[:, 1]
        
        # Vectorized triangle area computation
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = np.abs(v1x * v2y - v1y * v2x)
        areas = 0.5 * cross

        # Smooth penalty: only penalize triangles below target
        deficit = np.maximum(t_target - areas, 0.0)
        penalty = gamma * np.sum(deficit ** 2)
        
        # Add small regularization to encourage spread
        # Penalize points that are too close to each other
        dist_matrix = np.zeros((N, N))
        for i in range(N):
            for j in range(i+1, N):
                d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                dist_matrix[i,j] = d
                dist_matrix[j,i] = d
        
        # Soft repulsion for very close points
        close_penalty = 0.01 * np.sum(np.maximum(0.05 - dist_matrix, 0)**2)
        
        return penalty + close_penalty

    def _gradient_maxmin(points, t_target, gamma=1.0):
        """
        Compute gradient of the max-min penalty objective.
        """
        try:
            hull = ConvexHull(points)
            area_hull = hull.volume
        except Exception:
            return np.zeros(2*N)
            
        if area_hull < 1e-9:
            return np.zeros(2*N)

        x = points[:, 0].copy()
        y = points[:, 1].copy()
        
        # Compute all triangle areas
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = np.abs(v1x * v2y - v1y * v2x)
        areas = 0.5 * cross

        # Deficit for each triangle
        deficit = np.maximum(t_target - areas, 0.0)
        mask = deficit > 0

        grad = np.zeros(2*N)
        
        if np.any(mask):
            # For triangles below target, compute gradient of (t - area)^2
            # d(area)/d(points) can be computed analytically
            # area = 0.5 * |cross|
            # For non-degenerate triangles, d(area)/d(p_i) depends on which point
            
            for idx in np.where(mask)[0]:
                i, j, k = tri_i[idx], tri_j[idx], tri_k[idx]
                # Area = 0.5 * |(x_j-x_i)*(y_k-y_i) - (y_j-y_i)*(x_k-x_i)|
                # Let's compute gradient numerically for simplicity (small step)
                eps = 1e-6
                for dim in [0, 1]:
                    for p in [i, j, k]:
                        # Forward difference
                        old_val = points[p, dim]
                        points[p, dim] = old_val + eps
                        # Recompute this triangle's area
                        v1x_p = x[tri_j[idx]] - x[tri_i[idx]]
                        v1y_p = y[tri_j[idx]] - y[tri_i[idx]]
                        v2x_p = x[tri_k[idx]] - x[tri_i[idx]]
                        v2y_p = y[tri_k[idx]] - y[tri_i[idx]]
                        cross_p = abs(v1x_p * v2y_p - v1y_p * v2x_p)
                        area_p = 0.5 * cross_p
                        deficit_p = max(t_target - area_p, 0.0)
                        points[p, dim] = old_val
                        grad[p*2 + dim] += gamma * 2.0 * deficit * (deficit_p - deficit) / eps
        
        # Close-point repulsion gradient
        for i in range(N):
            for j in range(i+1, N):
                d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                if d < 0.05 and d > 1e-8:
                    # Gradient of 0.01 * (0.05 - d)^2
                    factor = 0.01 * 2.0 * (0.05 - d) * (-1.0) / d
                    grad[i*2] += factor * (x[i] - x[j])
                    grad[i*2+1] += factor * (y[i] - y[j])
                    grad[j*2] -= factor * (x[i] - x[j])
                    grad[j*2+1] -= factor * (y[i] - y[j])
        
        return grad

    def _fermat_spiral(n, radius=1.0):
        """Generate points using Fermat spiral for good spatial distribution."""
        angles = np.arange(n) * 2.39996322972865332  # Golden angle in radians
        r = np.sqrt(np.arange(1, n+1) / n) * radius
        xs = r * np.cos(angles)
        ys = r * np.sin(angles)
        return np.column_stack([xs, ys])

    def _regular_polygon(n, radius=1.0, center=(0.0, 0.0)):
        """Generate points of a regular n-gon."""
        if n == 1:
            return np.array([center])
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        xs = center[0] + radius * np.cos(angles)
        ys = center[1] + radius * np.sin(angles)
        return np.column_stack([xs, ys])

    def _simulated_annealing(points, t_target, gamma, max_iters=200, T_start=0.05, T_end=0.001):
        """
        Simulated annealing to escape local minima.
        Moves the worst-performing point in a random direction.
        """
        current = points.copy()
        current_score, _ = compute_score(current)
        best = current.copy()
        best_score = current_score
        
        rng = np.random.RandomState(42)
        
        for it in range(max_iters):
            T = T_start * (T_end / T_start) ** (it / max_iters)
            
            # Pick a point to move (bias toward points in small triangles)
            x = current[:, 0]
            y = current[:, 1]
            v1x = x[tri_j] - x[tri_i]
            v1y = y[tri_j] - y[tri_i]
            v2x = x[tri_k] - x[tri_i]
            v2y = y[tri_k] - y[tri_i]
            cross = np.abs(v1x * v2y - v1y * v2x)
            areas = 0.5 * cross
            
            # Find triangles with small areas
            small_idx = np.argsort(areas)[:10]
            # Pick one of the points from a small triangle
            tri_choice = rng.randint(len(small_idx))
            i, j, k = tri_i[small_idx[tri_choice]], tri_j[small_idx[tri_choice]], tri_k[small_idx[tri_choice]]
            point_idx = rng.choice([i, j, k])
            
            # Random move
            move_size = T * 0.5
            direction = rng.randn(2)
            direction /= (np.linalg.norm(direction) + 1e-8)
            
            new_points = current.copy()
            new_points[point_idx] += move_size * direction
            
            new_score, _ = compute_score(new_points)
            delta = new_score - current_score
            
            if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-8)):
                current = new_points
                if new_score > best_score:
                    best_score = new_score
                    best = new_points.copy()
        
        return best, best_score

    best_points = None
    best_score = -float('inf')
    
    # Strategy: Use max-min penalty with adaptive target t
    # Start with Fermat spiral and regular polygon inits
    
    inits = []
    inits.append(_fermat_spiral(N, radius=1.0))
    inits.append(_regular_polygon(N, radius=1.0))
    inits.append(_fermat_spiral(N, radius=1.0) + np.random.RandomState(1).randn(N, 2) * 0.05)
    inits.append(_fermat_spiral(N, radius=1.0) + np.random.RandomState(2).randn(N, 2) * 0.1)
    
    for init in inits:
        if time.time() > start_time + time_limit - 40:
            break
        
        s, hull_area = compute_score(init)
        if s > best_score:
            best_score = s
            best_points = init.copy()
        
        # Adaptive max-min penalty optimization
        current = init.copy()
        # Start with a low target and increase
        t_target = s * hull_area * 0.5  # Start below current min area
        
        for stage in range(4):
            if time.time() > start_time + time_limit - 35:
                break
            
            # Increase target gradually
            t_target *= 1.3
            gamma = 1.0 + stage * 0.5
            
            # Use L-BFGS-B with penalty objective
            try:
                def penalty_obj(x_flat):
                    pts = x_flat.reshape(-1, 2)
                    return _maxmin_penalty(pts, t_target, gamma)
                
                def penalty_grad(x_flat):
                    pts = x_flat.reshape(-1, 2).copy()
                    g = _gradient_maxmin(pts, t_target, gamma)
                    return g
                
                result = opt.minimize(
                    penalty_obj,
                    current.flatten(),
                    jac=penalty_grad,
                    method='L-BFGS-B',
                    options={'maxiter': 80, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
                )
                current = result.x.reshape(-1, 2)
                opt_score, _ = compute_score(current)
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = current.copy()
            except Exception:
                pass
        
        # Simulated annealing to escape local minima
        if time.time() < start_time + time_limit - 25:
            sa_points, sa_score = _simulated_annealing(
                best_points, 
                best_score * (compute_score(best_points)[1] if best_points is not None else 1.0),
                gamma=1.0,
                max_iters=100,
                T_start=0.03,
                T_end=0.001
            )
            if sa_score > best_score:
                best_score = sa_score
                best_points = sa_points.copy()
            
            # Polish after SA
            if time.time() < start_time + time_limit - 20:
                try:
                    t_target = best_score * compute_score(best_points)[1]
                    result = opt.minimize(
                        lambda x: _maxmin_penalty(x.reshape(-1,2), t_target, 2.0),
                        best_points.flatten(),
                        method='L-BFGS-B',
                        options={'maxiter': 100, 'ftol': 1e-13, 'gtol': 1e-11, 'maxls': 60}
                    )
                    polished = result.x.reshape(-1, 2)
                    pol_score, _ = compute_score(polished)
                    if pol_score > best_score:
                        best_score = pol_score
                        best_points = polished.copy()
                except Exception:
                    pass

    # Final Nelder-Mead polish on the best configuration
    if best_points is not None and time.time() < start_time + time_limit - 10:
        try:
            def hard_neg(x):
                pts = x.reshape(-1, 2)
                s, _ = compute_score(pts)
                return -s
            
            res = opt.minimize(
                hard_neg,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 300, 'adaptive': True, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            nm_points = res.x.reshape(-1, 2)
            nm_score, _ = compute_score(nm_points)
            if nm_score > best_score:
                best_score = nm_score
                best_points = nm_points.copy()
        except Exception:
            pass
        
        # One more SA round if time permits
        if time.time() < start_time + time_limit - 5:
            try:
                sa_points, sa_score = _simulated_annealing(
                    best_points,
                    best_score * compute_score(best_points)[1],
                    gamma=1.0,
                    max_iters=50,
                    T_start=0.02,
                    T_end=0.0005
                )
                if sa_score > best_score:
                    best_score = sa_score
                    best_points = sa_points.copy()
            except Exception:
                pass

    if best_points is None:
        best_points = _regular_polygon(13, radius=1.0)
    
    return best_points

# EVOLVE-BLOCK-END
