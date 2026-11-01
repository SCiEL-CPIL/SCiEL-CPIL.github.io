import numpy as np
from scipy.optimize import minimize
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Objective function: minimize negative minimum distance
    def objective(x):
        # x is flattened array of shape (2*N,)
        c = x.reshape(N, 2)
        
        # Calculate pairwise distances
        # Use broadcasting to get all pairwise distances
        # diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        # dists = np.linalg.norm(diff, axis=2)
        # np.fill_diagonal(dists, np.inf)
        # min_pair_dist = np.min(dists)
        
        # More efficient:
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        min_pair_dist = np.sqrt(np.min(dists_sq))
        
        # Distance to boundary
        # Square is [-0.5, 0.5]
        # dist to left: 0.5 - x
        # dist to right: 0.5 + x
        # dist to bottom: 0.5 - y
        # dist to top: 0.5 + y
        d_left = 0.5 - c[:, 0]
        d_right = 0.5 + c[:, 0]
        d_bottom = 0.5 - c[:, 1]
        d_top = 0.5 + c[:, 1]
        
        min_boundary_dist = np.min([np.min(d_left), np.min(d_right), np.min(d_bottom), np.min(d_top)])
        
        r = min(min_pair_dist / 2.0, min_boundary_dist)
        
        return -r

    # Gradient function
    def gradient(x):
        c = x.reshape(N, 2)
        
        # Pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        dists = np.sqrt(dists_sq)
        
        min_pair_dist = np.min(dists)
        
        # Boundary distances
        d_left = 0.5 - c[:, 0]
        d_right = 0.5 + c[:, 0]
        d_bottom = 0.5 - c[:, 1]
        d_top = 0.5 + c[:, 1]
        
        min_boundary_dist = np.min([np.min(d_left), np.min(d_right), np.min(d_bottom), np.min(d_top)])
        
        r = min(min_pair_dist / 2.0, min_boundary_dist)
        
        grad = np.zeros_like(x)
        
        # Determine which constraint is active
        # If min_pair_dist / 2 < min_boundary_dist, pair constraint is active
        # Else boundary constraint is active
        
        if min_pair_dist / 2.0 < min_boundary_dist:
            # Find indices of min pair
            # dists is N x N
            min_idx = np.unravel_index(np.argmin(dists), dists.shape)
            i, j = min_idx
            if i == j:
                # Should not happen due to inf diagonal
                return grad
            
            # Gradient of distance d_ij w.r.t c_i is (c_i - c_j) / d_ij
            # We are minimizing -r, so we want to maximize r.
            # grad of r w.r.t c_i is (c_i - c_j) / d_ij
            # grad of -r is -(c_i - c_j) / d_ij
            # But we want to move away from each other.
            # To increase distance, c_i should move in direction (c_i - c_j).
            # So grad of objective (-r) should point in direction of decreasing r.
            # Wait. minimize f(x). f(x) = -r.
            # We want to increase r. So we want to move in direction of -grad(f) = grad(r).
            # L-BFGS-B minimizes f. So it moves in -grad(f).
            # So we should return grad(f) = -grad(r).
            
            vec = c[i] - c[j]
            dist = dists[i, j]
            if dist > 0:
                g_i = vec / dist
                g_j = -vec / dist
                
                # grad(r) w.r.t c_i is g_i
                # grad(-r) w.r.t c_i is -g_i
                grad[i*2:i*2+2] = -g_i
                grad[j*2:j*2+2] = -g_j
        else:
            # Boundary constraint active
            # Find which boundary and which point
            min_d = min_boundary_dist
            # Check which boundary
            # We need to find the point and side that gives min_d
            # This is slightly more complex.
            
            # Let's just use finite differences or a simpler gradient if this is too complex.
            # Actually, we can check mins for each side.
            
            # d_left min
            idx_left = np.argmin(d_left)
            if np.isclose(d_left[idx_left], min_d):
                # Point idx_left is closest to left
                # dist = 0.5 - x
                # grad(dist) = (-1, 0)
                # grad(-r) = -(-1, 0) = (1, 0)
                grad[idx_left*2:idx_left*2+2] = [1.0, 0.0]
            
            idx_right = np.argmin(d_right)
            if np.isclose(d_right[idx_right], min_d):
                # dist = 0.5 + x
                # grad(dist) = (1, 0)
                # grad(-r) = -(1, 0) = (-1, 0)
                grad[idx_right*2:idx_right*2+2] = [-1.0, 0.0]
                
            idx_bottom = np.argmin(d_bottom)
            if np.isclose(d_bottom[idx_bottom], min_d):
                # dist = 0.5 - y
                # grad(dist) = (0, -1)
                # grad(-r) = (0, 1)
                grad[idx_bottom*2:idx_bottom*2+2] = [0.0, 1.0]
                
            idx_top = np.argmin(d_top)
            if np.isclose(d_top[idx_top], min_d):
                # dist = 0.5 + y
                # grad(dist) = (0, 1)
                # grad(-r) = (0, -1)
                grad[idx_top*2:idx_top*2+2] = [0.0, -1.0]
        
        return grad

    # Hexagonal grid initialization
    def hex_grid_init():
        # Approximate number of rows/cols
        # Density approx 0.9
        # Area 1. 50 circles.
        # r ~ 0.075. s = 2r ~ 0.15.
        # Width 1. 1/0.15 ~ 6.6 cols.
        # Height 1. 1/(0.15*sqrt(3)/2) ~ 1/0.13 ~ 7.7 rows.
        
        s = 0.16
        points = []
        y = -0.5 + s * np.sqrt(3) / 2
        row = 0
        while y <= 0.5:
            x_start = -0.5 + s/2 if row % 2 == 0 else 0
            # Adjust x_start to center?
            # Let's just generate and filter
            x = -0.5
            while x <= 0.5:
                if row % 2 == 1:
                    x += s/2
                points.append([x, y])
                x += s
            y += s * np.sqrt(3) / 2
            row += 1
        
        points = np.array(points)
        
        # Filter points inside square
        mask = (points[:, 0] >= -0.5) & (points[:, 0] <= 0.5) & (points[:, 1] >= -0.5) & (points[:, 1] <= 0.5)
        points = points[mask]
        
        # If we have more than N, random sample
        if len(points) > N:
            idx = np.random.choice(len(points), N, replace=False)
            points = points[idx]
        # If we have less, add random points
        elif len(points) < N:
            extra = np.random.uniform(-0.4, 0.4, size=(N - len(points), 2))
            points = np.vstack([points, extra])
            
        return points

    best_x = None
    best_score = -np.inf
    
    # Try a few restarts
    for i in range(10):
        if time.time() > deadline:
            break
            
        x0 = hex_grid_init().flatten()
        
        # Bounds
        bounds = [(-0.5, 0.5)] * (2 * N)
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', jac=gradient, bounds=bounds, options={'maxiter': 200, 'ftol': 1e-10})
            if res.success:
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_x = res.x
        except Exception:
            pass
            
    if best_x is None:
        # Fallback to grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
        
    return best_x.reshape(N, 2)
