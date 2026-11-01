"""Program for covering the sphere with 130 equal caps using smooth soft-objective
optimization with analytical gradients and an exact convex-hull polish."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 130


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before the 150 s hard limit

    def fibonacci_spiral(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def generate_grid(n_lat, n_lon):
        lat = np.linspace(-np.pi / 2, np.pi / 2, n_lat)
        lon = np.linspace(0, 2 * np.pi, n_lon, endpoint=False)
        LA, LO = np.meshgrid(lat, lon, indexing='ij')
        z = np.sin(LA)
        x = np.cos(LA) * np.cos(LO)
        y = np.cos(LA) * np.sin(LO)
        return np.column_stack([x.ravel(), y.ravel(), z.ravel()])

    # Coarse -> medium -> fine -> very-fine grids.
    grid_coarse = generate_grid(40, 80)
    grid_med = generate_grid(70, 140)
    grid_fine = generate_grid(110, 220)
    grid_vf = generate_grid(150, 300)

    def covering_radius(pts, grid):
        dots = np.clip(grid @ pts.T, -1.0, 1.0)
        ang = np.arccos(dots)
        return float(np.max(np.min(ang, axis=1)))

    def exact_covering_radius(pts):
        """Compute exact covering radius from convex hull facets."""
        try:
            hull = ConvexHull(pts)
        except Exception:
            return np.pi
        max_radius = 0.0
        for simplex in hull.simplices:
            verts = pts[simplex]
            a, b, c = verts[0], verts[1], verts[2]
            normal = np.cross(b - a, c - a)
            norm_val = np.linalg.norm(normal)
            if norm_val < 1e-12:
                continue
            normal = normal / norm_val
            centroid = np.mean(verts, axis=0)
            if np.dot(normal, centroid) < 0:
                normal = -normal
            dots = np.clip(np.dot(verts, normal), -1.0, 1.0)
            angles = np.arccos(dots)
            facet_radius = np.max(angles)
            max_radius = max(max_radius, facet_radius)
        return max_radius

    def soft_covering(pts, beta, grid, t=None):
        """Smooth approximation of max_y min_i angle(y, p_i).

        Uses epigraph formulation: minimize t + (1/beta)*log(sum_g exp(beta*(smin_g - t)))
        where smin_g is the soft-min over config points at grid point g.
        If t is None, use the closed-form solution t = m2 + log(sum(exp(beta*(smin-m2))))/beta.
        """
        dots = np.clip(grid @ pts.T, -1.0, 1.0)
        ang = np.arccos(dots)
        # soft-min over config points (axis 1)
        m1 = np.max(ang, axis=1, keepdims=True)
        w1 = np.exp(beta * (ang - m1))
        w1 /= np.sum(w1, axis=1, keepdims=True)
        smin = np.sum(w1 * ang, axis=1)
        # If t is provided, use it; otherwise compute closed-form
        if t is None:
            m2 = np.max(smin)
            w2 = np.exp(beta * (smin - m2))
            w2 /= np.sum(w2)
            t = m2 + np.log(np.sum(w2)) / beta
        else:
            m2 = np.max(smin)
            w2 = np.exp(beta * (smin - m2))
            w2 /= np.sum(w2)
        val = t + np.log(np.sum(np.exp(beta * (smin - t)))) / beta
        return val, w1, w2, ang, dots, t

    def soft_covering_grad(pts, beta, grid, t=None):
        """Analytical gradient of the soft covering radius w.r.t. pts (N,3).

        d angle / d p_j = -g / sqrt(1 - d^2) + (d / sqrt(1 - d^2)) * p_j  (tangent).
        """
        val, w1, w2, ang, dots, t_val = soft_covering(pts, beta, grid, t)
        denom = np.sqrt(np.maximum(1.0 - dots * dots, 1e-16))
        # per (grid, config) coefficient = w2[grid] * w1[grid, config]
        coeff = w2[:, None] * w1
        alpha = coeff / denom
        # term1_j = - sum_g alpha[g,j] * g_g
        term1 = -(alpha.T @ grid)
        # term2_j = (sum_g alpha[g,j] * dots[g,j]) * p_j
        beta_s = np.sum(alpha * dots, axis=0)
        term2 = beta_s[:, None] * pts
        return val, term1 + term2

    def lloyd_step(pts, grid, alpha=0.5, temp=2.0):
        dots = np.clip(grid @ pts.T, -1.0, 1.0)
        nearest = np.argmax(dots, axis=1)
        new_pts = pts.copy()
        for i in range(N):
            mask = (nearest == i)
            if not np.any(mask):
                continue
            cell = grid[mask]
            d = np.clip(np.dot(pts[i], cell.T), -1.0, 1.0)
            a = np.arccos(d)
            w = np.exp(-a * temp)
            w /= np.sum(w)
            c = np.sum(cell * w[:, None], axis=0)
            nrm = np.linalg.norm(c)
            if nrm > 1e-10:
                new_pts[i] = c / nrm
        cur = (1 - alpha) * pts + alpha * new_pts
        cur /= np.linalg.norm(cur, axis=1, keepdims=True)
        return cur

    def gap_fill_step(pts, grid, alpha=0.5):
        dots = np.clip(grid @ pts.T, -1.0, 1.0)
        ang = np.arccos(dots)
        min_d = np.min(ang, axis=1)
        nidx = np.argmin(ang, axis=1)
        wi = int(np.argmax(min_d))
        wp = grid[wi]
        ci = int(nidx[wi])
        p = pts[ci]
        v = wp - p
        v_tan = v - np.dot(v, p) * p
        np_ = p + alpha * v_tan
        np_ /= np.linalg.norm(np_)
        pts = pts.copy()
        pts[ci] = np_
        return pts

    def optimize_smooth(pts, beta, grid, max_iter=60, tol=1e-10, gtol=1e-7):
        if time.time() > deadline - 12.0:
            return pts
        x0 = pts.flatten()

        def obj(x):
            if time.time() > deadline - 6.0:
                return 1e10
            return soft_covering(x.reshape(N, 3), beta, grid)[0]

        def objg(x):
            if time.time() > deadline - 6.0:
                return 1e10, np.zeros_like(x)
            v, g = soft_covering_grad(x.reshape(N, 3), beta, grid)
            return v, g.flatten()

        try:
            res = minimize(objg, x0, method='L-BFGS-B', jac=True,
                           options={'maxiter': max_iter, 'ftol': tol,
                                    'gtol': gtol, 'maxls': 40})
            if res.fun < obj(x0):
                p = res.x.reshape(N, 3)
                p /= np.linalg.norm(p, axis=1, keepdims=True)
                return p
        except Exception:
            pass
        return pts

    def facet_circumcenter(a, b, c):
        """Circumcenter (unit normal direction) of a spherical triangle."""
        n = np.cross(b - a, c - a)
        nv = np.linalg.norm(n)
        if nv < 1e-12:
            return None
        n = n / nv
        if np.dot(n, (a + b + c) / 3.0) < 0:
            n = -n
        return n

    def facet_radius(n, verts):
        d = np.clip(np.dot(verts, n), -1.0, 1.0)
        return float(np.max(np.arccos(d)))

    def exact_polish(pts, max_iters=200):
        """Nudge points toward circumcenters of their worst facets (exact hull)."""
        pts = pts.copy()
        step_size = 0.1
        for it in range(max_iters):
            if time.time() > deadline - 3.0:
                break
            try:
                hull = ConvexHull(pts)
            except Exception:
                break
            # per-point worst facet radius
            worst = np.zeros(N)
            worst_facet = np.full(N, -1)
            for si, simp in enumerate(hull.simplices):
                verts = pts[simp]
                n = facet_circumcenter(verts[0], verts[1], verts[2])
                if n is None:
                    continue
                r = facet_radius(n, verts)
                for j in simp:
                    if r > worst[j]:
                        worst[j] = r
                        worst_facet[j] = si
            cur_max = float(np.max(worst))
            # only move points that are on a (near-)worst facet
            for j in range(N):
                if worst[j] < cur_max - 1e-4:
                    continue
                si = int(worst_facet[j])
                if si < 0:
                    continue
                simp = hull.simplices[si]
                verts = pts[simp]
                n = facet_circumcenter(verts[0], verts[1], verts[2])
                if n is None:
                    continue
                p = pts[j]
                v = n - p
                v_tan = v - np.dot(v, p) * p
                # Adaptive step: smaller near convergence
                actual_step = step_size * min(1.0, (cur_max - 0.195) * 10)
                np_ = p + actual_step * v_tan
                np_ /= np.linalg.norm(np_)
                pts[j] = np_
            # Decay step size
            step_size *= 0.95
        return pts

    def worst_facet_refinement(pts, max_iters=150):
        """Directly attack the worst facet by moving its vertices toward the circumcenter."""
        pts = pts.copy()
        step_size = 0.12
        for it in range(max_iters):
            if time.time() > deadline - 2.0:
                break
            try:
                hull = ConvexHull(pts)
            except Exception:
                break
            # Find the single worst facet
            worst_si = -1
            worst_r = -1.0
            worst_cc = None
            for si, simp in enumerate(hull.simplices):
                verts = pts[simp]
                n = facet_circumcenter(verts[0], verts[1], verts[2])
                if n is None:
                    continue
                r = facet_radius(n, verts)
                if r > worst_r:
                    worst_r = r
                    worst_si = si
                    worst_cc = n
            if worst_si < 0:
                break
            # Move the three vertices of the worst facet toward the circumcenter
            simp = hull.simplices[worst_si]
            for j in simp:
                p = pts[j]
                v = worst_cc - p
                v_tan = v - np.dot(v, p) * p
                actual_step = step_size * min(1.0, (worst_r - 0.195) * 12)
                np_ = p + actual_step * v_tan
                np_ /= np.linalg.norm(np_)
                pts[j] = np_
            step_size *= 0.94
        return pts

    best_pts = None
    best_rad = np.inf
    n_restarts = 16

    for r in range(n_restarts):
        if time.time() > deadline - 40.0:
            break

        if r == 0:
            pts = fibonacci_spiral(N)
        elif r == 1:
            rng = np.random.RandomState(42)
            pts = rng.randn(N, 3)
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        elif r == 2:
            rng = np.random.RandomState(123)
            pts = rng.randn(N, 3)
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        elif r == 3:
            rng = np.random.RandomState(789)
            pts = rng.randn(N, 3)
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        elif r >= 4 and best_pts is not None:
            rng = np.random.RandomState(1000 + r)
            scale = 0.05 * (1.0 - r / (n_restarts + 1.0))
            pts = best_pts + scale * rng.randn(N, 3)
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        else:
            pts = fibonacci_spiral(N)

        # Stage 1: coarse Lloyd
        for i in range(20):
            if time.time() > deadline - 32.0:
                break
            pts = lloyd_step(pts, grid_coarse, alpha=0.5, temp=2.0)

        # Stage 2: medium Lloyd + gap fill
        for i in range(12):
            if time.time() > deadline - 27.0:
                break
            pts = lloyd_step(pts, grid_med, alpha=0.4, temp=3.0)
            if i % 4 == 0:
                pts = gap_fill_step(pts, grid_med, alpha=0.2)

        # Stage 3: smooth optimization (medium)
        pts = optimize_smooth(pts, beta=50, grid=grid_med, max_iter=45)

        # Stage 4: fine Lloyd + gap fill
        for i in range(12):
            if time.time() > deadline - 20.0:
                break
            pts = lloyd_step(pts, grid_fine, alpha=0.3, temp=4.0)
            if i % 3 == 0:
                pts = gap_fill_step(pts, grid_fine, alpha=0.1)

        # Stage 5: smooth optimization (fine)
        pts = optimize_smooth(pts, beta=100, grid=grid_fine, max_iter=35)

        # Stage 6: very-fine smooth optimization
        if time.time() < deadline - 15.0:
            pts = optimize_smooth(pts, beta=150, grid=grid_vf, max_iter=25)

        # Stage 7: exact hull polish
        pts = exact_polish(pts)

        # Stage 8: worst-facet refinement (from inspiration)
        if time.time() < deadline - 3.0:
            pts = worst_facet_refinement(pts)

        rad = covering_radius(pts, grid_vf)
        if rad < best_rad:
            best_rad = rad
            best_pts = pts.copy()

    if best_pts is None:
        best_pts = fibonacci_spiral(N)
    return best_pts
