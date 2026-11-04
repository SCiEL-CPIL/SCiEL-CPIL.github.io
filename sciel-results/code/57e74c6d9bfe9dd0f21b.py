"""Circle packing in unit square - Voronoi-based dual optimization."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Voronoi

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def is_valid(x, tol=1e-8):
        xs, ys, rs = x[:, 0], x[:, 1], x[:, 2]
        if np.any(xs - rs < -tol) or np.any(xs + rs > 1 + tol):
            return False
        if np.any(ys - rs < -tol) or np.any(ys + rs > 1 + tol):
            return False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(xs[i] - xs[j], ys[i] - ys[j])
                if d < rs[i] + rs[j] - tol:
                    return False
        return True

    def compute_radii(centers):
        """Exact max radii for fixed centers."""
        centers = np.asarray(centers, dtype=float)
        n = len(centers)
        radii = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                           np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(dist, np.inf)
        half_dist = dist / 2.0
        min_pair = np.min(half_dist, axis=1)
        radii = np.minimum(radii, min_pair)
        return radii

    def compute_radii_fast(centers):
        """Exact max radii - optimized for repeated calls."""
        n = len(centers)
        radii = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                           np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
        np.fill_diagonal(dist_sq, np.inf)
        min_pair_sq = np.min(dist_sq, axis=1)
        min_pair = np.sqrt(min_pair_sq) / 2.0
        radii = np.minimum(radii, min_pair)
        return radii

    def voronoi_inradii(centers):
        """
        Compute inradii using Voronoi tessellation.
        For each site, the inradius of its Voronoi cell gives the max radius
        that circle can have without overlapping neighbors (given the cell
        is clipped to the unit square).
        """
        centers = np.asarray(centers, dtype=float)
        n = len(centers)
        
        # Add boundary points to clip Voronoi cells to the square
        # Use a large number of boundary points for accurate clipping
        n_bnd = 200
        bnd_pts = []
        for t in np.linspace(0, 1, n_bnd, endpoint=False):
            bnd_pts.append([t, 0.0])
            bnd_pts.append([t, 1.0])
            bnd_pts.append([0.0, t])
            bnd_pts.append([1.0, t])
        bnd_pts = np.array(bnd_pts)
        
        all_pts = np.vstack([centers, bnd_pts])
        
        try:
            vor = Voronoi(all_pts)
        except Exception:
            # Fallback to direct computation
            return compute_radii(centers)
        
        radii = np.zeros(n)
        for i in range(n):
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]
            if len(region) == 0 or -1 in region:
                # Unbounded region (shouldn't happen with boundary points)
                radii[i] = compute_radii(centers)[i]
                continue
            
            vertices = vor.vertices[region]
            if len(vertices) < 3:
                radii[i] = compute_radii(centers)[i]
                continue
            
            # Compute inradius of the polygon (distance from center to each edge)
            min_dist = np.inf
            for k in range(len(vertices)):
                v1 = vertices[k]
                v2 = vertices[(k + 1) % len(vertices)]
                # Distance from point to line segment
                d = point_to_segment_dist(centers[i], v1, v2)
                if d < min_dist:
                    min_dist = d
            
            radii[i] = min_dist
        
        return radii

    def point_to_segment_dist(p, a, b):
        """Distance from point p to segment ab."""
        ab = b - a
        ap = p - a
        if np.dot(ab, ab) < 1e-15:
            return np.linalg.norm(ap)
        t = np.dot(ap, ab) / np.dot(ab, ab)
        t = np.clip(t, 0, 1)
        closest = a + t * ab
        return np.linalg.norm(p - closest)

    def voronoi_objective(x_flat):
        """Negative sum of Voronoi inradii."""
        c = x_flat.reshape(-1, 2)
        r = voronoi_inradii(c)
        return -np.sum(r)

    def voronoi_gradient(x_flat):
        """
        Numerical gradient of the Voronoi inradius objective.
        Uses central differences on the smooth inradius function.
        """
        c = x_flat.reshape(-1, 2)
        n = len(c)
        grad = np.zeros_like(x_flat)
        eps = 1e-6
        
        for i in range(n):
            for d in range(2):
                c_plus = c.copy()
                c_plus[i, d] += eps
                c_minus = c.copy()
                c_minus[i, d] -= eps
                f_plus = voronoi_objective(c_plus.flatten())
                f_minus = voronoi_objective(c_minus.flatten())
                grad[2 * i + d] = (f_plus - f_minus) / (2 * eps)
        
        return grad

    def voronoi_optimize(centers, max_iter=200, tol=1e-10):
        """
        Optimize center positions using Voronoi inradius as objective.
        Uses L-BFGS-B with numerical gradients.
        """
        x0 = centers.flatten()
        bounds = [(0.0, 1.0)] * (2 * N)
        
        try:
            res = minimize(voronoi_objective, x0, method='L-BFGS-B',
                           bounds=bounds,
                           options={'maxiter': max_iter, 'gtol': tol, 'ftol': 1e-14,
                                    'maxls': 50})
            c = res.x.reshape(-1, 2)
            c = np.clip(c, 0.0, 1.0)
            return c
        except Exception:
            return centers

    def smooth_radii(centers, beta=100.0):
        """Soft-min approximation of radii using log-sum-exp."""
        centers = np.asarray(centers, dtype=float)
        n = len(centers)
        db = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                        np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(dist, np.inf)
        half_dist = dist / 2.0

        radii = np.zeros(n)
        for i in range(n):
            constraints = np.concatenate(([db[i]], half_dist[i, :]))
            constraints = constraints[np.isfinite(constraints)]
            if len(constraints) == 0:
                radii[i] = 0.5
            else:
                m = np.min(constraints)
                s = np.sum(np.exp(-beta * (constraints - m)))
                radii[i] = m - np.log(s) / beta
        return radii

    def smooth_obj(x_flat):
        c = x_flat.reshape(-1, 2)
        r = smooth_radii(c)
        return -np.sum(r)

    def smooth_grad(x_flat):
        beta = 100.0
        c = x_flat.reshape(-1, 2)
        n = len(c)
        grad = np.zeros_like(x_flat)

        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(dist, np.inf)

        for i in range(n):
            bvals = np.array([c[i, 0], 1.0 - c[i, 0], c[i, 1], 1.0 - c[i, 1]])
            pvals = dist[i, :].copy()
            pvals[i] = np.inf
            pvals = pvals / 2.0

            all_vals = np.concatenate([bvals, pvals])
            m = np.min(all_vals)
            exps = np.exp(-beta * (all_vals - m))
            s = np.sum(exps)
            weights = -exps / s

            grad[2 * i] += weights[0] * 1.0 + weights[1] * (-1.0)
            grad[2 * i + 1] += weights[2] * 1.0 + weights[3] * (-1.0)

            for j in range(n):
                if j == i:
                    continue
                d = dist[i, j]
                if d < 1e-12:
                    continue
                w = weights[4 + j]
                grad[2 * i] += w * (c[i, 0] - c[j, 0]) / (2.0 * d)
                grad[2 * i + 1] += w * (c[i, 1] - c[j, 1]) / (2.0 * d)
                grad[2 * j] += w * (c[j, 0] - c[i, 0]) / (2.0 * d)
                grad[2 * j + 1] += w * (c[j, 1] - c[i, 1]) / (2.0 * d)

        return -grad

    def gradient_polish(centers, max_iter=500):
        """Polish using L-BFGS-B on smooth surrogate."""
        x0 = centers.flatten()
        bounds = [(0.0, 1.0)] * (2 * N)

        def obj(x):
            return smooth_obj(x)

        def grad(x):
            return smooth_grad(x)

        try:
            res = minimize(obj, x0, jac=grad, method='L-BFGS-B',
                           bounds=bounds,
                           options={'maxiter': max_iter, 'gtol': 1e-10, 'ftol': 1e-14})
            c = res.x.reshape(-1, 2)
            c = np.clip(c, 0.0, 1.0)
            return c
        except Exception:
            return centers

    def best_fit_local_search(centers, max_sweeps=50, tol=1e-10):
        """Best-fit: for each circle, find position that maximizes its radius."""
        centers = np.array(centers, dtype=float)
        n = len(centers)

        for sweep in range(max_sweeps):
            if time.time() > deadline - 10:
                break
            improved = False
            for i in range(n):
                others = np.delete(centers, i, axis=0)
                cur_r = min(
                    centers[i, 0], 1.0 - centers[i, 0],
                    centers[i, 1], 1.0 - centers[i, 1]
                )
                for oc in others:
                    d = np.hypot(centers[i, 0] - oc[0], centers[i, 1] - oc[1])
                    r = d / 2.0
                    if r < cur_r:
                        cur_r = r

                best_r = cur_r
                best_pos = centers[i].copy()
                search_range = max(cur_r * 0.5, 0.02)
                n_grid = 7
                for dx in np.linspace(-search_range, search_range, n_grid):
                    for dy in np.linspace(-search_range, search_range, n_grid):
                        nx = centers[i, 0] + dx
                        ny = centers[i, 1] + dy
                        if nx < 0 or nx > 1 or ny < 0 or ny > 1:
                            continue
                        r = min(nx, 1.0 - nx, ny, 1.0 - ny)
                        for oc in others:
                            d = np.hypot(nx - oc[0], ny - oc[1])
                            rr = d / 2.0
                            if rr < r:
                                r = rr
                        if r > best_r + tol:
                            best_r = r
                            best_pos = np.array([nx, ny])

                if np.linalg.norm(best_pos - centers[i]) > tol:
                    centers[i] = best_pos
                    improved = True

            if not improved:
                break

        return centers

    def simulated_annealing(centers, T0=0.1, T_end=0.001, n_iter=3000, rng=None):
        """Simulated annealing."""
        if rng is None:
            rng = np.random.RandomState(0)
        centers = np.array(centers, dtype=float)
        n = len(centers)
        current_score = np.sum(compute_radii_fast(centers))
        best_centers = centers.copy()
        best_score = current_score

        for it in range(n_iter):
            T = T0 * (T_end / T0) ** (it / max(n_iter - 1, 1))
            if time.time() > deadline - 20:
                break

            i = rng.randint(n)
            move_size = 0.02 + 0.05 * (T / T0)
            new_pos = centers[i].copy()
            new_pos[0] += rng.normal(0, move_size)
            new_pos[1] += rng.normal(0, move_size)
            new_pos = np.clip(new_pos, 0.01, 0.99)

            old_pos = centers[i].copy()
            centers[i] = new_pos
            new_score = np.sum(compute_radii_fast(centers))

            delta = new_score - current_score
            if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_centers = centers.copy()
            else:
                centers[i] = old_pos

        return best_centers, best_score

    def high_precision_polish(centers, max_iter=200):
        """Very fine best-fit local search."""
        centers = np.array(centers, dtype=float)
        n = len(centers)

        for sweep in range(max_iter):
            if time.time() > deadline - 5:
                break
            improved = False
            for i in range(n):
                others = np.delete(centers, i, axis=0)
                cur_r = min(
                    centers[i, 0], 1.0 - centers[i, 0],
                    centers[i, 1], 1.0 - centers[i, 1]
                )
                for oc in others:
                    d = np.hypot(centers[i, 0] - oc[0], centers[i, 1] - oc[1])
                    r = d / 2.0
                    if r < cur_r:
                        cur_r = r

                # Multi-scale refinement
                best_r = cur_r
                best_pos = centers[i].copy()
                for scale in [max(cur_r * 0.3, 0.01), max(cur_r * 0.1, 0.003), max(cur_r * 0.03, 0.0005)]:
                    n_g = 11
                    for dx in np.linspace(-scale, scale, n_g):
                        for dy in np.linspace(-scale, scale, n_g):
                            nx = centers[i, 0] + dx
                            ny = centers[i, 1] + dy
                            if nx < 0 or nx > 1 or ny < 0 or ny > 1:
                                continue
                            r = min(nx, 1.0 - nx, ny, 1.0 - ny)
                            for oc in others:
                                d = np.hypot(nx - oc[0], ny - oc[1])
                                rr = d / 2.0
                                if rr < r:
                                    r = rr
                            if r > best_r + 1e-14:
                                best_r = r
                                best_pos = np.array([nx, ny])

                if np.linalg.norm(best_pos - centers[i]) > 1e-12:
                    centers[i] = best_pos
                    improved = True

            if not improved:
                break

        return centers

    def grid_init():
        side = int(np.ceil(np.sqrt(N)))
        centers = []
        for i in range(side):
            for j in range(side):
                if len(centers) < N:
                    centers.append([(i + 0.5) / side, (j + 0.5) / side])
        return np.array(centers[:N])

    def random_init(seed):
        rng = np.random.RandomState(seed)
        return rng.uniform(0.05, 0.95, size=(N, 2))

    def hexagonal_init():
        """Hexagonal packing initialization."""
        centers = []
        # Approximate hexagonal grid
        r = 0.19
        dy = 2 * r
        dx = 2 * r * np.sqrt(3) / 2
        for row in range(6):
            for col in range(6):
                if len(centers) >= N:
                    break
                x = 0.1 + col * dx + (row % 2) * dx / 2
                y = 0.1 + row * dy * 0.866
                if x < 0.95 and y < 0.95:
                    centers.append([x, y])
        # Fill remaining with random
        rng = np.random.RandomState(42)
        while len(centers) < N:
            centers.append([rng.uniform(0.05, 0.95), rng.uniform(0.05, 0.95)])
        return np.array(centers[:N])

    best_centers = None
    best_score = -np.inf
    best_result = None

    def evaluate_and_update(centers):
        nonlocal best_centers, best_score, best_result
        radii = compute_radii(centers)
        result = np.column_stack([centers, radii])
        if is_valid(result, tol=1e-6):
            score = np.sum(radii)
            if score > best_score:
                best_score = score
                best_centers = centers.copy()
                best_result = result
                return True
        return False

    # Phase 1: Multiple diverse initializations with SA + Voronoi optimization
    inits = [
        ('grid', grid_init()),
        ('hex', hexagonal_init()),
        ('rand0', random_init(0)),
        ('rand1', random_init(1)),
        ('rand7', random_init(7)),
        ('rand13', random_init(13)),
        ('rand23', random_init(23)),
        ('rand37', random_init(37)),
        ('rand53', random_init(53)),
        ('rand71', random_init(71)),
        ('rand89', random_init(89)),
        ('rand42', random_init(42)),
        ('rand99', random_init(99)),
        ('rand123', random_init(123)),
        ('rand200', random_init(200)),
        ('rand300', random_init(300)),
        ('rand500', random_init(500)),
        ('rand777', random_init(777)),
        ('rand1234', random_init(1234)),
        ('rand2024', random_init(2024)),
    ]

    for name, init_centers in inits:
        if time.time() > deadline - 50:
            break

        # Quick best-fit
        centers = best_fit_local_search(init_centers, max_sweeps=10)

        # SA
        rng = np.random.RandomState(hash(name) % 10000)
        centers_sa, _ = simulated_annealing(centers, T0=0.08, T_end=0.0005,
                                             n_iter=2500, rng=rng)

        # Best-fit polish
        centers_bf = best_fit_local_search(centers_sa, max_sweeps=20)
        evaluate_and_update(centers_bf)

    # Phase 2: Voronoi-based optimization on best solutions
    if best_centers is not None:
        if time.time() < deadline - 40:
            centers_vor = voronoi_optimize(best_centers, max_iter=300)
            centers_vor = best_fit_local_search(centers_vor, max_sweeps=20)
            evaluate_and_update(centers_vor)

    # Phase 3: Gradient polish on best
    if best_centers is not None:
        if time.time() < deadline - 25:
            centers_grad = gradient_polish(best_centers, max_iter=400)
            centers_grad = best_fit_local_search(centers_grad, max_sweeps=20)
            evaluate_and_update(centers_grad)

    # Phase 4: Combined Voronoi + gradient cycles
    if best_centers is not None:
        if time.time() < deadline - 20:
            for _ in range(3):
                if time.time() > deadline - 10:
                    break
                centers_v = voronoi_optimize(best_centers, max_iter=200)
                centers_g = gradient_polish(centers_v, max_iter=300)
                centers_g = best_fit_local_search(centers_g, max_sweeps=15)
                if not evaluate_and_update(centers_g):
                    break

    # Phase 5: High-precision polish
    if best_centers is not None:
        if time.time() < deadline - 10:
            centers_hp = high_precision_polish(best_centers, max_iter=150)
            evaluate_and_update(centers_hp)

        # More Voronoi + high precision
        if time.time() < deadline - 5:
            for _ in range(3):
                if time.time() > deadline - 3:
                    break
                centers_v = voronoi_optimize(best_centers, max_iter=150)
                centers_hp = high_precision_polish(centers_v, max_iter=100)
                if not evaluate_and_update(centers_hp):
                    break

    # Final high-precision pass
    if best_centers is not None and time.time() < deadline - 2:
        centers_final = high_precision_polish(best_centers, max_iter=50)
        evaluate_and_update(centers_final)

    # Fallback
    if best_result is None:
        centers = grid_init()
        radii = compute_radii(centers)
        best_result = np.column_stack([centers, radii])

    # Final safety check
    if not is_valid(best_result, tol=1e-6):
        centers = grid_init()
        radii = compute_radii(centers)
        best_result = np.column_stack([centers, radii])

    return best_result

# EVOLVE-BLOCK-END
