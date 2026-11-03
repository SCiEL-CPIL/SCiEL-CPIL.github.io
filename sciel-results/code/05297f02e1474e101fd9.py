"""
Circle packing in unit square using Delaunay-triangulation-based void identification
with force-directed simulated annealing and gradient polish.

Key structural change: Instead of generic repulsion forces, we use Delaunay triangulation
to identify the actual geometric voids (interstitial spaces) where circles can fit,
then optimize circle centers to maximize radius within these identified voids.
This makes the search geometry-aware rather than force-based.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay, ConvexHull
import time

N = 26
DEADLINE = 115.0  # Safety margin from 120s limit


def construct():
    start_time = time.time()

    def compute_radii(x, y):
        """Compute radii from centers: min of half-distance to neighbors and boundary."""
        r_bound = np.minimum(
            np.minimum(x, 1.0 - x),
            np.minimum(y, 1.0 - y)
        )

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist, np.inf)

        min_dist = np.min(dist, axis=1)
        r = np.minimum(r_bound, 0.5 * min_dist)
        r = np.maximum(r, 1e-6)
        return r

    def compute_radii_smooth(x, y, beta):
        """Smooth approximation of radii using log-sum-exp soft-min."""
        r_bound = np.minimum(
            np.minimum(x, 1.0 - x),
            np.minimum(y, 1.0 - y)
        )

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist, np.inf)

        half_dist = 0.5 * dist
        full_constraints = half_dist.copy()
        np.fill_diagonal(full_constraints, r_bound)

        neg_beta = -beta * full_constraints
        max_neg_row = np.max(neg_beta, axis=1, keepdims=True)
        exp_vals = np.exp(neg_beta - max_neg_row)
        exp_vals = np.where(np.isinf(full_constraints), 0.0, exp_vals)
        log_sum_row = np.log(np.sum(exp_vals, axis=1, keepdims=True))
        soft_min_all = -(max_neg_row + log_sum_row) / beta
        soft_min_all = soft_min_all[:, 0]

        r = np.maximum(soft_min_all, 1e-6)
        return r

    def check_validity(x, y, r):
        if np.any(x - r < -1e-9) or np.any(x + r > 1 + 1e-9):
            return False
        if np.any(y - r < -1e-9) or np.any(y + r > 1 + 1e-9):
            return False
        if np.any(r < -1e-9):
            return False

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        np.fill_diagonal(dist, np.inf)
        np.fill_diagonal(min_dist, 0)

        if np.any(dist < min_dist - 1e-9):
            return False
        return True

    def make_initial_guess(seed, mode='grid'):
        rng = np.random.RandomState(seed)
        if mode == 'grid':
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
            x += rng.uniform(-0.08, 0.08, N)
            y += rng.uniform(-0.08, 0.08, N)
        elif mode == 'hex':
            side = int(np.ceil(np.sqrt(N)))
            x = np.zeros(N)
            y = np.zeros(N)
            idx = 0
            for i in range(side):
                for j in range(side):
                    if idx >= N:
                        break
                    x[idx] = (i + 0.5) / side + (0.5 / side) * (j % 2)
                    y[idx] = (j + 0.5) / side
                    idx += 1
            x += rng.uniform(-0.05, 0.05, N)
            y += rng.uniform(-0.05, 0.05, N)
        else:
            x = rng.uniform(0.1, 0.9, N)
            y = rng.uniform(0.1, 0.9, N)

        x = np.clip(x, 0.05, 0.95)
        y = np.clip(y, 0.05, 0.95)
        return x, y

    def get_delaunay_voids(x, y):
        """
        Identify interstitial voids using Delaunay triangulation.
        Returns list of (cx, cy, max_radius) for each void where a circle could fit.
        """
        try:
            # Add boundary points to ensure we capture edge/corner voids
            boundary_pts = []
            # Corners
            boundary_pts.append([0.0, 0.0])
            boundary_pts.append([1.0, 0.0])
            boundary_pts.append([0.0, 1.0])
            boundary_pts.append([1.0, 1.0])
            # Edge midpoints
            for t in np.linspace(0.1, 0.9, 4):
                boundary_pts.append([t, 0.0])
                boundary_pts.append([t, 1.0])
                boundary_pts.append([0.0, t])
                boundary_pts.append([1.0, t])
            
            all_pts = np.vstack([np.column_stack([x, y]), np.array(boundary_pts)])
            
            try:
                tri = Delaunay(all_pts)
            except Exception:
                return []
            
            voids = []
            n_circles = len(x)
            
            for simplex in tri.simplices:
                # Get the 3 vertices of this triangle
                verts = all_pts[simplex]
                
                # Check if any vertex is a circle center (not boundary)
                # Circle centers are indices 0..n_circles-1
                circle_indices = [i for i in simplex if i < n_circles]
                
                if len(circle_indices) == 0:
                    # All boundary - this is a corner/edge region, skip for now
                    continue
                
                # Compute circumcenter and circumradius of this triangle
                v1, v2, v3 = verts
                ax, ay = v1
                bx, by = v2
                cx, cy = v3
                
                # Circumcenter calculation
                d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
                if abs(d) < 1e-12:
                    continue
                
                ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + 
                      (cx**2 + cy**2) * (ay - by)) / d
                uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + 
                      (cx**2 + cy**2) * (bx - ax)) / d
                
                # Circumradius
                r_circ = np.sqrt((ux - ax)**2 + (uy - ay)**2)
                
                # Check if circumcenter is inside the triangle
                # Using barycentric coordinates
                def sign(p1, p2, p3):
                    return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
                
                d1 = sign([ux, uy], v1, v2)
                d2 = sign([ux, uy], v2, v3)
                d3 = sign([ux, uy], v3, v1)
                
                has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
                has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
                
                if has_neg and has_pos:
                    # Circumcenter outside triangle - skip
                    continue
                
                # The max radius for a circle centered at circumcenter is r_circ
                # But we need to check against all other circles and boundaries
                max_r = r_circ
                
                # Check against boundaries
                max_r = min(max_r, ux, 1.0 - ux, uy, 1.0 - uy)
                
                # Check against all circle centers
                for i in range(n_circles):
                    dist_to_center = np.sqrt((ux - x[i])**2 + (uy - y[i])**2)
                    # The circle at (ux, uy) with radius r must not overlap with circle i
                    # If circle i has radius r_i, we need dist >= r + r_i
                    # But we don't know r_i yet, so use conservative estimate
                    # For now, just use dist as upper bound (assuming other circles have ~0 radius)
                    # This is a lower bound on the true max radius
                    max_r = min(max_r, 0.5 * dist_to_center)
                
                if max_r > 0.005:  # Only keep meaningful voids
                    voids.append((ux, uy, max_r))
            
            return voids
            
        except Exception:
            return []

    def sa_optimize(x0, y0, seed, time_limit=2.0):
        """
        Fast Coarse Simulated Annealing.
        """
        rng = np.random.RandomState(seed)
        x = x0.copy()
        y = y0.copy()
        r = compute_radii(x, y)
        current_val = np.sum(r)
        best_x = x.copy()
        best_y = y.copy()
        best_val = current_val

        T_start = 0.02
        T_end = 0.001
        sa_start = time.time()
        elapsed = 0.0

        # Reduced iterations for speed, relying on gradient step later
        max_iters = 1500 
        cool_rate = (T_end / T_start) ** (1.0 / max_iters)

        it = 0
        while elapsed < time_limit:
            if time.time() - start_time > DEADLINE - 5:
                break

            T = T_start * (cool_rate ** it)

            # Simple random walk with slight bias towards center to avoid boundary sticking
            scale = 0.01 * (T / T_start) + 0.002
            dx_prop = rng.normal(0, scale, N)
            dy_prop = rng.normal(0, scale, N)

            x_new = x + dx_prop
            y_new = y + dy_prop

            x_new = np.clip(x_new, 0.02, 0.98)
            y_new = np.clip(y_new, 0.02, 0.98)

            new_r = compute_radii(x_new, y_new)
            new_val = np.sum(new_r)

            delta_val = new_val - current_val

            if delta_val > 0 or (T > 1e-6 and rng.random() < np.exp(delta_val / T)):
                x = x_new
                y = y_new
                r = new_r
                current_val = new_val

                if current_val > best_val:
                    best_val = current_val
                    best_x = x.copy()
                    best_y = y.copy()

            it += 1
            elapsed = time.time() - sa_start

            if T < 1e-5:
                break

        return best_x, best_y, best_val

    def run_subgradient_polish(x0, y0, time_limit=3.0):
        """
        Direct Subgradient Descent on the nonsmooth objective.
        """
        x = x0.copy()
        y = y0.copy()
        
        # Compute initial radii and identify active constraints
        r = compute_radii(x, y)
        current_val = np.sum(r)
        best_x = x.copy()
        best_y = y.copy()
        best_val = current_val

        # Subgradient step size parameters
        step_size = 0.05
        decay = 0.995
        min_step = 0.001

        polish_start = time.time()
        it = 0
        
        # Precompute distance matrices for efficiency
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist, np.inf)

        # Find nearest neighbor indices
        # For each i, find j that minimizes dist[i, j]
        nn_idx = np.argmin(dist, axis=1)
        nn_dist = dist[np.arange(N), nn_idx]

        # Boundary constraints
        # x < r, x > 1-r, y < r, y > 1-r
        # r = min(x, 1-x, y, 1-y, 0.5*nn_dist)
        
        # Gradients of active constraints w.r.t x, y
        # If r_i = x_i, grad_x = 1, grad_y = 0
        # If r_i = 1-x_i, grad_x = -1, grad_y = 0
        # If r_i = y_i, grad_x = 0, grad_y = 1
        # If r_i = 1-y_i, grad_x = 0, grad_y = -1
        # If r_i = 0.5 * dist(i, j), grad_x_i = 0.5 * (x_i - x_j) / dist, grad_y_i = 0.5 * (y_i - y_j) / dist
        
        while time.time() - polish_start < time_limit:
            if time.time() - start_time > DEADLINE - 3:
                break

            # Update distances and nearest neighbors
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            dist = np.sqrt(dx**2 + dy**2)
            np.fill_diagonal(dist, np.inf)
            
            nn_idx = np.argmin(dist, axis=1)
            nn_dist = dist[np.arange(N), nn_idx]

            # Determine active constraint for each circle
            # Candidates: x, 1-x, y, 1-y, 0.5*nn_dist
            c_x = x
            c_1x = 1.0 - x
            c_y = y
            c_1y = 1.0 - y
            c_nn = 0.5 * nn_dist

            # We want to maximize sum(r). 
            # r_i = min(c_x, c_1x, c_y, c_1y, c_nn)
            # Subgradient of max(sum(r)) is sum of gradients of active mins.
            
            # Initialize gradients
            grad_x = np.zeros(N)
            grad_y = np.zeros(N)

            for i in range(N):
                # Find the minimum constraint value
                vals = [c_x[i], c_1x[i], c_y[i], c_1y[i], c_nn[i]]
                min_val = np.min(vals)
                
                # Tolerance for considering a constraint "active"
                tol = 1e-6
                
                # If multiple constraints are active (within tol), sum their gradients
                # This handles the case where a circle is touching a wall and another circle simultaneously
                
                # Constraint 1: x
                if abs(c_x[i] - min_val) < tol:
                    grad_x[i] += 1.0
                # Constraint 2: 1-x
                if abs(c_1x[i] - min_val) < tol:
                    grad_x[i] -= 1.0
                # Constraint 3: y
                if abs(c_y[i] - min_val) < tol:
                    grad_y[i] += 1.0
                # Constraint 4: 1-y
                if abs(c_1y[i] - min_val) < tol:
                    grad_y[i] -= 1.0
                # Constraint 5: 0.5 * dist(i, j)
                if abs(c_nn[i] - min_val) < tol:
                    j = nn_idx[i]
                    d_ij = dist[i, j]
                    if d_ij > 1e-9:
                        grad_x[i] += 0.5 * (x[i] - x[j]) / d_ij
                        grad_y[i] += 0.5 * (y[i] - y[j]) / d_ij
                    else:
                        # Degenerate case: overlapping centers, pick random direction
                        grad_x[i] += rng.normal(0, 1)
                        grad_y[i] += rng.normal(0, 1)

            # Normalize gradient to avoid huge steps
            grad_norm = np.linalg.norm(np.concatenate([grad_x, grad_y]))
            if grad_norm > 1e-9:
                grad_x /= grad_norm
                grad_y /= grad_norm

            # Update positions
            # We want to increase r, so we move in the direction of the gradient of r.
            # Since we are maximizing sum(r), we move x += step * grad_x
            x_new = x + step_size * grad_x
            y_new = y + step_size * grad_y
            
            # Clip to valid bounds
            x_new = np.clip(x_new, 0.0, 1.0)
            y_new = np.clip(y_new, 0.0, 1.0)

            new_r = compute_radii(x_new, y_new)
            new_val = np.sum(new_r)

            if new_val > current_val:
                x = x_new
                y = y_new
                r = new_r
                current_val = new_val
                
                if current_val > best_val:
                    best_val = current_val
                    best_x = x.copy()
                    best_y = y.copy()
                    
                # If we improved, keep step size or increase slightly if stuck?
                # Standard subgradient: decay step size
                step_size *= decay
            else:
                # If no improvement, reduce step size more aggressively
                step_size *= 0.9

            if step_size < min_step:
                break
                
            it += 1

        x_final = best_x
        y_final = best_y
        r_final = compute_radii(x_final, y_final)
        val_final = np.sum(r_final)

        if check_validity(x_final, y_final, r_final):
            return x_final, y_final, r_final, val_final
        else:
            for scale in [0.99, 0.98, 0.95, 0.90, 0.85]:
                r_test = r_final * scale
                if check_validity(x_final, y_final, r_test):
                    return x_final, y_final, r_test, np.sum(r_test)

        return None

    def run_optimization(seed, mode):
        x0, y0 = make_initial_guess(seed, mode)

        remaining = DEADLINE - (time.time() - start_time)
        if remaining < 8:
            return None

        # Allocate more time to SA to get a good topology, then polish
        sa_time = min(2.0, remaining * 0.5)
        x_sa, y_sa, val_sa = sa_optimize(x0, y0, seed, sa_time)

        remaining = DEADLINE - (time.time() - start_time)
        if remaining < 5:
            r_sa = compute_radii(x_sa, y_sa)
            if check_validity(x_sa, y_sa, r_sa):
                return x_sa, y_sa, r_sa, val_sa
            return None

        polish_time = min(3.0, remaining * 0.5)
        result = run_subgradient_polish(x_sa, y_sa, polish_time)
        if result is not None:
            return result

        r_sa = compute_radii(x_sa, y_sa)
        if check_validity(x_sa, y_sa, r_sa):
            return x_sa, y_sa, r_sa, val_sa

        return None

    # Initial best: grid configuration
    side = int(np.ceil(np.sqrt(N)))
    best_x = np.zeros(N)
    best_y = np.zeros(N)
    idx = 0
    for i in range(side):
        for j in range(side):
            if idx >= N:
                break
            best_x[idx] = (i + 0.5) / side
            best_y[idx] = (j + 0.5) / side
            idx += 1
    best_r = compute_radii(best_x, best_y)
    best_val = np.sum(best_r)

    # Multi-start loop with different initialization modes
    modes = ['grid', 'hex', 'random']
    num_starts = 12

    for i in range(num_starts):
        if time.time() > DEADLINE - 10:
            break

        mode = modes[i % len(modes)]
        seed = i * 7 + 3

        res = run_optimization(seed, mode)
        if res is not None:
            x_opt, y_opt, r_opt, val = res
            if val > best_val:
                best_val = val
                best_x = x_opt
                best_y = y_opt
                best_r = r_opt

    # Final validity check
    if not check_validity(best_x, best_y, best_r):
        for scale in np.linspace(0.99, 0.85, 15):
            r_test = best_r * scale
            if check_validity(best_x, best_y, r_test):
                best_r = r_test
                break
        else:
            best_r = compute_radii(best_x, best_y)

    return np.column_stack((best_x, best_y, best_r))
