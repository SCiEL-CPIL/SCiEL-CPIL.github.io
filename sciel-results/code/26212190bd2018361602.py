"""Circle packing in unit square n=32."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Voronoi
import time
import sys

N = 32

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin (150s limit)

    def voronoi_radii(x, y):
        """Compute maximum radii using Voronoi diagram for interior circles,
        boundary distance for boundary circles."""
        points = np.column_stack((x, y))
        # Add boundary points to make Voronoi cells bounded
        # Add points far outside to bound the diagram
        pad = 2.0
        boundary_points = []
        # Corners and edge midpoints extended
        for px in [-pad, 0, 0.5, 1, 1+pad]:
            for py in [-pad, 0, 0.5, 1, 1+pad]:
                if not (0 <= px <= 1 and 0 <= py <= 1):
                    boundary_points.append([px, py])
        boundary_points = np.array(boundary_points)
        
        # Also add points just outside each edge
        for i in range(10):
            t = (i + 0.5) / 10
            boundary_points = np.vstack([boundary_points, [[t, -pad], [t, 1+pad], [-pad, t], [1+pad, t]]])
        
        all_points = np.vstack([points, boundary_points])
        
        try:
            vor = Voronoi(all_points)
        except Exception:
            # Fallback: use pairwise distances
            dist_boundary = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
            dx = x[:, np.newaxis] - x[np.newaxis, :]
            dy = y[:, np.newaxis] - y[np.newaxis, :]
            dist_matrix = np.sqrt(dx**2 + dy**2)
            np.fill_diagonal(dist_matrix, np.inf)
            min_dist = np.min(dist_matrix, axis=1)
            radii = np.minimum(dist_boundary, min_dist / 2.0)
            return np.maximum(radii, 0)
        
        radii = np.zeros(N)
        for i in range(N):
            region = vor.regions[vor.point_region[i]]
            if -1 in region or len(region) == 0:
                # Unbounded region - use boundary distance
                r = np.minimum(np.minimum(x[i], 1 - x[i]), np.minimum(y[i], 1 - y[i]))
                radii[i] = r
            else:
                # Compute inradius of the Voronoi cell
                vertices = vor.vertices[region]
                # The inradius is the minimum distance from point to any edge of the cell
                # More efficiently: find the closest vertex to the point? No, need closest edge.
                # Actually, for a Voronoi cell, the inradius is the distance to the nearest
                # edge, which equals the distance to the nearest other point divided by... 
                # Wait, no. The inradius of the Voronoi cell is the largest circle centered
                # at point i that fits in the cell. This is the minimum distance from point i
                # to any edge of the Voronoi cell.
                
                # Alternative: the inradius equals min over all other points j of dist(i,j)/2
                # only if the cell is determined by those nearest neighbors. Actually, the
                # inradius of the Voronoi cell is exactly min_j dist(i,j)/2 for the nearest
                # neighbors that form the cell, but it could also be limited by the boundary.
                
                # Let's just compute it properly: distance from point to each edge
                min_dist_to_edge = np.inf
                for k in range(len(vertices)):
                    v1 = vertices[k]
                    v2 = vertices[(k + 1) % len(vertices)]
                    # Distance from point to line segment v1-v2
                    d = _point_to_segment_dist(points[i], v1, v2)
                    min_dist_to_edge = min(min_dist_to_edge, d)
                
                # Also consider boundary
                r_boundary = np.minimum(np.minimum(x[i], 1 - x[i]), np.minimum(y[i], 1 - y[i]))
                radii[i] = min(min_dist_to_edge, r_boundary)
        
        return np.maximum(radii, 0)

    def _point_to_segment_dist(p, a, b):
        """Distance from point p to line segment ab."""
        ab = b - a
        ap = p - a
        ab_len_sq = np.dot(ab, ab)
        if ab_len_sq < 1e-20:
            return np.linalg.norm(ap)
        t = np.dot(ap, ab) / ab_len_sq
        t = np.clip(t, 0, 1)
        closest = a + t * ab
        return np.linalg.norm(p - closest)

    def compute_radii(x, y):
        """Compute the maximum radii for given centers using pairwise distances (fast)."""
        dist_boundary = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist_matrix = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist_matrix, np.inf)
        min_dist = np.min(dist_matrix, axis=1)
        radii = np.minimum(dist_boundary, min_dist / 2.0)
        radii = np.maximum(radii, 0)
        return radii

    def sum_radii(x, y):
        """Sum of radii (the value we want to maximize)."""
        return np.sum(compute_radii(x, y))

    def objective(params):
        """Objective to minimize (negative sum of radii)."""
        x = params[0::2]
        y = params[1::2]
        return -sum_radii(x, y)

    def smooth_objective(params, alpha=50.0):
        """Smooth objective using soft-min for gradient-based methods."""
        x = params[0::2]
        y = params[1::2]

        dist_boundary = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))

        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist_matrix = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist_matrix, 1e10)

        # Soft-min of pairwise distances
        scaled = -alpha * dist_matrix
        row_max = np.max(scaled, axis=1, keepdims=True)
        exp_scaled = np.exp(scaled - row_max)
        np.fill_diagonal(exp_scaled, 0.0)
        log_sum = np.log(np.sum(exp_scaled, axis=1)) + row_max[:, 0]
        min_dist_soft = -log_sum / alpha

        # Soft-min of boundary distances
        boundary_terms = np.stack([x, 1 - x, y, 1 - y], axis=1)
        scaled_b = -alpha * boundary_terms
        row_max_b = np.max(scaled_b, axis=1, keepdims=True)
        exp_b = np.exp(scaled_b - row_max_b)
        log_sum_b = np.log(np.sum(exp_b, axis=1)) + row_max_b[:, 0]
        min_boundary_soft = -log_sum_b / alpha

        # Soft-min of (boundary, dist/2)
        a = min_boundary_soft
        b = min_dist_soft / 2.0
        scaled_ab = -alpha * np.stack([a, b], axis=1)
        row_max_ab = np.max(scaled_ab, axis=1, keepdims=True)
        exp_ab = np.exp(scaled_ab - row_max_ab)
        log_sum_ab = np.log(np.sum(exp_ab, axis=1)) + row_max_ab[:, 0]
        radii_soft = -log_sum_ab / alpha

        radii_soft = np.maximum(radii_soft, 0)
        return -np.sum(radii_soft)

    def hex_init():
        """Hexagonal lattice initialization for better packing density."""
        x = np.zeros(N)
        y = np.zeros(N)
        cols = 6
        rows = 6
        dx = 1.0 / cols
        dy = 1.0 / (rows * np.sqrt(3) / 2)
        idx = 0
        for r in range(rows):
            for c in range(cols):
                if idx >= N:
                    break
                x[idx] = (c + (0.5 if r % 2 == 1 else 0)) * dx
                x[idx] = np.clip(x[idx], 0.01, 0.99)
                y[idx] = (r + 0.5) * dy
                y[idx] = np.clip(y[idx], 0.01, 0.99)
                idx += 1
        if idx < N:
            for k in range(idx, N):
                x[k] = 0.5
                y[k] = 0.5
        params = np.zeros(2 * N)
        params[0::2] = x
        params[1::2] = y
        return params

    def grid_init():
        """Square grid initialization."""
        side = int(np.ceil(np.sqrt(N)))
        x = np.zeros(N)
        y = np.zeros(N)
        idx = 0
        for i in range(side):
            for j in range(side):
                if idx >= N:
                    break
                x[idx] = (i + 0.5) / side
                y[idx] = (j + 0.5) / side
                idx += 1
        if idx < N:
            for k in range(idx, N):
                x[k] = 0.5
                y[k] = 0.5
        params = np.zeros(2 * N)
        params[0::2] = x
        params[1::2] = y
        return params

    def random_init(seed):
        """Random initialization."""
        rng = np.random.RandomState(seed)
        x = rng.uniform(0.05, 0.95, N)
        y = rng.uniform(0.05, 0.95, N)
        params = np.zeros(2 * N)
        params[0::2] = x
        params[1::2] = y
        return params

    def lbfgsb_refine(params, maxiter=3000):
        """Refine parameters using L-BFGS-B with smooth objective."""
        bounds = [(0.0, 1.0)] * (2 * N)
        try:
            res = minimize(
                smooth_objective, params, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            return res.x
        except Exception:
            return params

    def powell_refine(params, maxiter=5000):
        """Refine parameters using Powell's method."""
        bounds = [(0.0, 1.0)] * (2 * N)
        try:
            res = minimize(
                objective, params, method='Powell', bounds=bounds,
                options={'maxiter': maxiter, 'xtol': 1e-12, 'ftol': 1e-12}
            )
            return res.x
        except Exception:
            return params

    def coordinate_descent(x, y, max_iter=100):
        """Coordinate descent: move one circle at a time to improve sum of radii."""
        x = x.copy()
        y = y.copy()
        current_sum = sum_radii(x, y)
        
        step_sizes = [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
        
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
            improved = False
            for i in range(N):
                for step in step_sizes:
                    for dx in [-step, 0, step]:
                        for dy in [-step, 0, step]:
                            if dx == 0 and dy == 0:
                                continue
                            x_try = x.copy()
                            y_try = y.copy()
                            x_try[i] += dx
                            y_try[i] += dy
                            if x_try[i] < 0 or x_try[i] > 1 or y_try[i] < 0 or y_try[i] > 1:
                                continue
                            s_try = sum_radii(x_try, y_try)
                            if s_try > current_sum + 1e-15:
                                x, y = x_try, y_try
                                current_sum = s_try
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
        return x, y, current_sum

    def repulsion_force_optimize(x, y, n_iter=200):
        """Optimize using repulsion forces: each circle pushes others away."""
        x = x.copy()
        y = y.copy()
        current_sum = sum_radii(x, y)
        
        for it in range(n_iter):
            if time.time() > deadline - 15:
                break
            
            # Compute forces
            fx = np.zeros(N)
            fy = np.zeros(N)
            
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < 1e-10:
                        dx = 0.01
                        dy = 0.01
                        dist = np.sqrt(dx**2 + dy**2)
                    
                    # Repulsion is stronger when circles are close
                    force = 1.0 / (dist**2 + 0.01)
                    fx[i] += dx / dist * force
                    fy[i] += dy / dist * force
                    fx[j] -= dx / dist * force
                    fy[j] -= dy / dist * force
                
                # Boundary forces
                if x[i] < 0.1:
                    fx[i] += 0.5
                if x[i] > 0.9:
                    fx[i] -= 0.5
                if y[i] < 0.1:
                    fy[i] += 0.5
                if y[i] > 0.9:
                    fy[i] -= 0.5
            
            # Move
            lr = 0.01 * (1 - it / n_iter) + 0.001
            x += lr * fx
            y += lr * fy
            x = np.clip(x, 0.0, 1.0)
            y = np.clip(y, 0.0, 1.0)
            
            new_sum = sum_radii(x, y)
            if new_sum < current_sum - 0.01:
                # Revert if too bad
                x -= lr * fx
                y -= lr * fy
                x = np.clip(x, 0.0, 1.0)
                y = np.clip(y, 0.0, 1.0)
            else:
                current_sum = new_sum
        
        return x, y, current_sum

    # Phase 1: Generate diverse initializations
    inits = [
        hex_init(),
        grid_init(),
        random_init(42),
        random_init(123),
        random_init(789),
        random_init(321),
    ]
    
    # Add jittered hex
    for seed in [10, 20]:
        rng = np.random.RandomState(seed)
        p = hex_init()
        p += rng.normal(0, 0.03, size=p.shape)
        p = np.clip(p, 0.01, 0.99)
        inits.append(p)
    
    # Phase 2: Optimize each initialization
    best_params = None
    best_obj = np.inf
    
    for i, init in enumerate(inits):
        if time.time() > deadline - 40:
            break
        
        # L-BFGS-B with smooth objective
        p1 = lbfgsb_refine(init, maxiter=2000)
        obj1 = objective(p1)
        
        # Powell refinement
        p2 = powell_refine(p1, maxiter=3000)
        obj2 = objective(p2)
        
        # Repulsion force optimization
        x_rep = p2[0::2].copy()
        y_rep = p2[1::2].copy()
        x_rep, y_rep, _ = repulsion_force_optimize(x_rep, y_rep, n_iter=100)
        p3 = np.zeros(2 * N)
        p3[0::2] = x_rep
        p3[1::2] = y_rep
        obj3 = objective(p3)
        
        # Choose best
        if obj3 < obj2:
            p_best = p3
            obj_best = obj3
        else:
            p_best = p2
            obj_best = obj2
        
        if obj_best < best_obj:
            best_params = p_best
            best_obj = obj_best
    
    # Phase 3: Coordinate descent refinement
    x_best = best_params[0::2].copy()
    y_best = best_params[1::2].copy()
    x_best, y_best, sum_best = coordinate_descent(x_best, y_best, max_iter=100)
    
    # Phase 4: Final high-precision polish
    if time.time() < deadline - 10:
        final_params = np.zeros(2 * N)
        final_params[0::2] = x_best
        final_params[1::2] = y_best
        final_params = powell_refine(final_params, maxiter=8000)
        final_obj = objective(final_params)
        if final_obj < best_obj:
            best_params = final_params
            best_obj = final_obj
    
    # Final construction
    x = best_params[0::2]
    y = best_params[1::2]
    radii = compute_radii(x, y)
    
    # Shrink radii slightly for floating point safety
    radii *= 0.999999
    
    result = np.column_stack((x, y, radii))
    return result

# EVOLVE-BLOCK-END
