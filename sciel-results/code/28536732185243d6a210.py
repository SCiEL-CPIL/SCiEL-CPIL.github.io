"""Rewritten program: sphere covering with 130 caps using facet-aware local search."""
import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin

    def fibonacci_spiral(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def normalize(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1.0
        return pts / norms

    def compute_covering_radius(pts):
        """Compute covering radius as max circumradius over convex hull faces."""
        try:
            hull = ConvexHull(pts)
            faces = hull.simplices
            a = pts[faces[:, 0]]
            b = pts[faces[:, 1]]
            c = pts[faces[:, 2]]

            # Circumcenter of triangle on unit sphere
            # The circumcenter direction is proportional to cross(a,b)+cross(b,c)+cross(c,a)
            ab = np.cross(a, b)
            bc = np.cross(b, c)
            ca = np.cross(c, a)
            c_dir = ab + bc + ca
            norms_c = np.linalg.norm(c_dir, axis=1)

            radii = np.zeros(len(faces))
            valid = norms_c > 1e-10

            if np.any(valid):
                cc = c_dir[valid] / norms_c[valid, np.newaxis]
                cos_r = np.clip(np.sum(cc * a[valid], axis=1), -1.0, 1.0)
                r = np.arccos(cos_r)
                # If circumcenter is on the far side, use the supplement
                r[r > np.pi / 2] = np.pi - r[r > np.pi / 2]
                radii[valid] = r

            if np.any(~valid):
                a_deg = a[~valid]
                b_deg = b[~valid]
                c_deg = c[~valid]
                d_ab = np.arccos(np.clip(np.sum(a_deg * b_deg, axis=1), -1, 1))
                d_bc = np.arccos(np.clip(np.sum(b_deg * c_deg, axis=1), -1, 1))
                d_ca = np.arccos(np.clip(np.sum(c_deg * a_deg, axis=1), -1, 1))
                radii[~valid] = np.maximum(np.maximum(d_ab, d_bc), d_ca)

            return float(np.max(radii))
        except Exception:
            # Fallback: grid-based estimate
            n_lat = 40
            n_lon = 80
            lat = np.linspace(-np.pi / 2, np.pi / 2, n_lat)
            lon = np.linspace(0, 2 * np.pi, n_lon, endpoint=False)
            LA, LO = np.meshgrid(lat, lon, indexing='ij')
            z = np.sin(LA)
            x = np.cos(LA) * np.cos(LO)
            y = np.cos(LA) * np.sin(LO)
            grid_pts = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
            dot = np.dot(grid_pts, pts.T)
            dot = np.clip(dot, -1.0, 1.0)
            ang_dists = np.arccos(dot)
            min_dists = np.min(ang_dists, axis=1)
            return float(np.max(min_dists))

    def get_facet_data(pts):
        """Return (radii, circumcenters, faces) for all hull facets."""
        try:
            hull = ConvexHull(pts)
            faces = hull.simplices
            a = pts[faces[:, 0]]
            b = pts[faces[:, 1]]
            c = pts[faces[:, 2]]

            ab = np.cross(a, b)
            bc = np.cross(b, c)
            ca = np.cross(c, a)
            c_dir = ab + bc + ca
            norms_c = np.linalg.norm(c_dir, axis=1)

            radii = np.zeros(len(faces))
            circumcenters = np.zeros((len(faces), 3))
            valid = norms_c > 1e-10

            if np.any(valid):
                cc = c_dir[valid] / norms_c[valid, np.newaxis]
                cos_r = np.clip(np.sum(cc * a[valid], axis=1), -1.0, 1.0)
                r = np.arccos(cos_r)
                r[r > np.pi / 2] = np.pi - r[r > np.pi / 2]
                radii[valid] = r
                circumcenters[valid] = cc

            if np.any(~valid):
                a_deg = a[~valid]
                b_deg = b[~valid]
                c_deg = c[~valid]
                d_ab = np.arccos(np.clip(np.sum(a_deg * b_deg, axis=1), -1, 1))
                d_bc = np.arccos(np.clip(np.sum(b_deg * c_deg, axis=1), -1, 1))
                d_ca = np.arccos(np.clip(np.sum(c_deg * a_deg, axis=1), -1, 1))
                radii[~valid] = np.maximum(np.maximum(d_ab, d_bc), d_ca)
                circumcenters[~valid] = (a_deg + b_deg + c_deg) / 3.0
                norms_cc = np.linalg.norm(circumcenters[~valid], axis=1, keepdims=True)
                norms_cc[norms_cc < 1e-15] = 1.0
                circumcenters[~valid] /= norms_cc

            return radii, circumcenters, faces
        except Exception:
            return None, None, None

    def repulsion_forces(pts):
        """Compute repulsion forces: each point pushed away from others, stronger when close."""
        dot = np.dot(pts, pts.T)
        dot = np.clip(dot, -1.0, 1.0)
        np.fill_diagonal(dot, 1.0)
        dist_sq = 2.0 - 2.0 * dot
        dist_sq[dist_sq < 1e-12] = 1e-12
        # Force magnitude ~ 1/d^3 for strong local repulsion
        w = 1.0 / (dist_sq ** 1.5)
        np.fill_diagonal(w, 0.0)
        forces = pts * np.sum(w, axis=1, keepdims=True) - w @ pts
        return forces

    def facet_push_forces(pts, radii, circumcenters, faces, beta=15.0):
        """
        Compute forces that push vertices of large facets toward the facet circumcenter.
        Uses softmax weighting so the largest facets dominate.
        """
        n_points = len(pts)
        n_faces = len(faces)
        if n_faces == 0:
            return np.zeros((n_points, 3))

        forces = np.zeros((n_points, 3))

        # Softmax weights emphasizing large radii
        max_r = np.max(radii)
        weights = np.exp(beta * (radii - max_r))
        total_w = np.sum(weights)
        if total_w < 1e-15:
            weights = np.ones(n_faces) / n_faces
        else:
            weights /= total_w

        # Build sparse-like accumulation
        # For each face, add force to its 3 vertices
        for fi in range(n_faces):
            w = weights[fi]
            if w < 1e-10:
                continue
            cc = circumcenters[fi]
            for vi in faces[fi]:
                p = pts[vi]
                # Tangential component of (cc - p)
                radial_comp = np.dot(cc - p, p) * p
                tangential = (cc - p) - radial_comp
                forces[vi] += w * tangential

        return forces

    def local_search(pts, time_budget):
        """
        Iterative local search:
        - Compute facet data
        - Apply repulsion + facet push forces
        - Normalize
        - Track best
        """
        best_pts = pts.copy()
        best_rad = compute_covering_radius(pts)

        alpha_rep = 0.02
        alpha_facet = 0.05
        decay = 0.997
        min_alpha = 1e-6
        stall = 0
        max_stall = 40

        while time.time() < start_time + time_budget:
            if alpha_rep < min_alpha and alpha_facet < min_alpha:
                break

            radii, ccs, faces = get_facet_data(pts)
            if radii is None:
                break

            # Repulsion
            f_rep = repulsion_forces(pts)
            # Facet push
            f_facet = facet_push_forces(pts, radii, ccs, faces, beta=15.0)

            pts = pts + alpha_rep * f_rep + alpha_facet * f_facet
            pts = normalize(pts)

            alpha_rep *= decay
            alpha_facet *= decay

            # Evaluate every few iterations
            if (int((start_time + time_budget) - time.time()) % 3 == 0) or alpha_rep < 0.005:
                rad = compute_covering_radius(pts)
                if rad < best_rad:
                    best_rad = rad
                    best_pts = pts.copy()
                    stall = 0
                else:
                    stall += 1
                    if stall > max_stall:
                        # Reduce step sizes more aggressively
                        alpha_rep *= 0.8
                        alpha_facet *= 0.8
                        stall = 0

        return best_pts, best_rad

    def polish(pts, time_budget):
        """Fine-tune with very small steps, alternating repulsion and facet push."""
        best_pts = pts.copy()
        best_rad = compute_covering_radius(pts)

        alpha = 0.002
        decay = 0.995
        min_alpha = 1e-7

        while time.time() < start_time + time_budget and alpha > min_alpha:
            radii, ccs, faces = get_facet_data(pts)
            if radii is None:
                break

            f_rep = repulsion_forces(pts)
            f_facet = facet_push_forces(pts, radii, ccs, faces, beta=25.0)

            pts = pts + alpha * (0.3 * f_rep + f_facet)
            pts = normalize(pts)
            alpha *= decay

            rad = compute_covering_radius(pts)
            if rad < best_rad:
                best_rad = rad
                best_pts = pts.copy()

        return best_pts, best_rad

    best_points = None
    best_rad = np.inf

    n_restarts = 8
    restart_time = (deadline - start_time) / n_restarts

    for restart in range(n_restarts):
        if time.time() > deadline - 15.0:
            break

        if restart == 0:
            pts = fibonacci_spiral(N).astype(np.float64)
        elif restart == 1:
            pts = np.random.randn(N, 3)
            pts = normalize(pts)
        else:
            if best_points is not None:
                pts = best_points + 0.08 * np.random.randn(N, 3)
                pts = normalize(pts)
            else:
                pts = fibonacci_spiral(N).astype(np.float64) + 0.05 * np.random.randn(N, 3)
                pts = normalize(pts)

        # Local search phase
        ls_budget = min(restart_time - 5.0, 15.0)
        pts_ls, rad_ls = local_search(pts, ls_budget)

        if rad_ls < best_rad:
            best_rad = rad_ls
            best_points = pts_ls.copy()

        # Polish phase
        if time.time() < deadline - 10.0:
            pts_pol, rad_pol = polish(pts_ls, min(8.0, deadline - start_time - 10.0))
            if rad_pol < best_rad:
                best_rad = rad_pol
                best_points = pts_pol.copy()

    # Final polish on best
    if time.time() < deadline - 5.0 and best_points is not None:
        pts_final, rad_final = polish(best_points, min(10.0, deadline - start_time - 5.0))
        if rad_final < best_rad:
            best_rad = rad_final
            best_points = pts_final.copy()

    if best_points is None:
        best_points = fibonacci_spiral(N).astype(np.float64)

    best_points = normalize(best_points)
    return best_points

# EVOLVE-BLOCK-END
