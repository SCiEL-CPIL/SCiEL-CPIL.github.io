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
        Force-directed Simulated Annealing with Delaunay-aware guidance.
        """
        rng = np.random.RandomState(seed)
        x = x0.copy()
        y = y0.copy()
        r = compute_radii(x, y)
        current_val = np.sum(r)
        best_x = x.copy()
        best_y = y.copy()
        best_val = current_val

        T_start = 0.04
        T_end = 0.0003
        sa_start = time.time()
        elapsed = 0.0

        max_iters = 3000
        cool_rate = (T_end / T_start) ** (1.0 / max_iters)

        it = 0
        while elapsed < time_limit:
            if time.time() - start_time > DEADLINE - 5:
                break

            T = T_start * (cool_rate ** it)

            # Compute forces (vectorized)
            fx = np.zeros(N)
            fy = np.zeros(N)

            # Boundary repulsion
            boundary_k = 5.0
            fx += boundary_k * np.clip(0.05 - x, 0, None)
            fx -= boundary_k * np.clip(x - 0.95, 0, None)
            fy += boundary_k * np.clip(0.05 - y, 0, None)
            fy -= boundary_k * np.clip(y - 0.95, 0, None)

            # Pairwise repulsion
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            d = np.sqrt(dx**2 + dy**2)
            np.fill_diagonal(d, 1e10)

            repulse_k = 1.5
            f_mag = repulse_k / (d**2 + 0.005)
            np.fill_diagonal(f_mag, 0)

            fx += np.sum(-f_mag * dx / d, axis=1)
            fy += np.sum(-f_mag * dy / d, axis=1)

            # Every 200 iterations, use Delaunay voids to guide moves
            if it % 200 == 0 and it > 0:
                voids = get_delaunay_voids(x, y)
                if len(voids) > 0:
                    # For each circle, find the nearest void and bias movement toward it
                    for i in range(N):
                        # Find nearest void
                        void_dists = [np.sqrt((x[i] - vx)**2 + (y[i] - vy)**2) 
                                     for vx, vy, _ in voids]
                        nearest_idx = np.argmin(void_dists)
                        vx, vy, vr = voids[nearest_idx]
                        
                        # Bias force toward the void center
                        bias_strength = 0.3 * (1.0 - void_dists[nearest_idx])
                        fx[i] += bias_strength * (vx - x[i])
                        fy[i] += bias_strength * (vy - y[i])

            # Propose move
            scale = 0.015 * (T / T_start) + 0.003
            dx_prop = fx * scale * 0.5 + rng.normal(0, scale, N)
            dy_prop = fy * scale * 0.5 + rng.normal(0, scale, N)

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

    def run_gradient_polish(x0, y0, time_limit=3.0):
        """
        Progressive beta annealing gradient polish.
        """
        params0 = np.concatenate([x0, y0])
        bounds = [(0.02, 0.98)] * (2 * N)

        beta_schedule = [10.0, 25.0, 50.0, 100.0, 200.0]

        best_params = params0.copy()
        best_val = -np.sum(compute_radii(x0, y0))

        polish_start = time.time()
        for beta in beta_schedule:
            if time.time() - start_time > DEADLINE - 3:
                break

            def objective(params):
                x = params[:N]
                y = params[N:2*N]
                r = compute_radii_smooth(x, y, beta)
                return -np.sum(r)

            try:
                res = minimize(objective, best_params, method='L-BFGS-B',
                               bounds=bounds,
                               options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 20})

                x_opt = res.x[:N]
                y_opt = res.x[N:2*N]
                r_opt = compute_radii(x_opt, y_opt)
                val = np.sum(r_opt)

                if val > -best_val:
                    best_val = -val
                    best_params = res.x.copy()
            except Exception:
                pass

            if time.time() - polish_start > time_limit:
                break

        x_final = best_params[:N]
        y_final = best_params[N:2*N]
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

        sa_time = min(1.5, remaining * 0.3)
        x_sa, y_sa, val_sa = sa_optimize(x0, y0, seed, sa_time)

        remaining = DEADLINE - (time.time() - start_time)
        if remaining < 5:
            r_sa = compute_radii(x_sa, y_sa)
            if check_validity(x_sa, y_sa, r_sa):
                return x_sa, y_sa, r_sa, val_sa
            return None

        polish_time = min(2.5, remaining * 0.4)
        result = run_gradient_polish(x_sa, y_sa, polish_time)
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
