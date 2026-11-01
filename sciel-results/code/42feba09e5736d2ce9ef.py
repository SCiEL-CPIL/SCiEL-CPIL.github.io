"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for safety

    def fibonacci_spiral(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def compute_covering_radius(pts):
        """Compute covering radius as max circumradius over convex hull faces."""
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
            norms = np.linalg.norm(c_dir, axis=1)
            
            radii = np.zeros(len(faces))
            valid = norms > 1e-10
            
            if np.any(valid):
                c_dir_valid = c_dir[valid] / norms[valid, np.newaxis]
                cos_r = np.clip(np.sum(c_dir_valid * a[valid], axis=1), -1.0, 1.0)
                r = np.arccos(cos_r)
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
            lat = np.linspace(-np.pi/2, np.pi/2, 50)
            lon = np.linspace(0, 2*np.pi, 100, endpoint=False)
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

    def get_face_data(pts):
        """Return face radii, faces, circumcenters, and per-vertex max face radius."""
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
            norms = np.linalg.norm(c_dir, axis=1)
            
            radii = np.zeros(len(faces))
            circumcenters = np.zeros_like(a)
            valid = norms > 1e-10
            
            if np.any(valid):
                c_dir_valid = c_dir[valid] / norms[valid, np.newaxis]
                cos_r = np.clip(np.sum(c_dir_valid * a[valid], axis=1), -1.0, 1.0)
                r = np.arccos(cos_r)
                r[r > np.pi / 2] = np.pi - r[r > np.pi / 2]
                radii[valid] = r
                circumcenters[valid] = c_dir_valid
            
            if np.any(~valid):
                a_deg = a[~valid]
                b_deg = b[~valid]
                c_deg = c[~valid]
                d_ab = np.arccos(np.clip(np.sum(a_deg * b_deg, axis=1), -1, 1))
                d_bc = np.arccos(np.clip(np.sum(b_deg * c_deg, axis=1), -1, 1))
                d_ca = np.arccos(np.clip(np.sum(c_deg * a_deg, axis=1), -1, 1))
                radii[~valid] = np.maximum(np.maximum(d_ab, d_bc), d_ca)
                # For degenerate faces, use average as pseudo-circumcenter
                circumcenters[~valid] = (a_deg + b_deg + c_deg) / 3.0
                cc_norms = np.linalg.norm(circumcenters[~valid], axis=1, keepdims=True)
                circumcenters[~valid] = circumcenters[~valid] / np.maximum(cc_norms, 1e-10)
            
            # Per-vertex max face radius
            vertex_max_rad = np.zeros(len(pts))
            for fi in range(len(faces)):
                for vi in faces[fi]:
                    if radii[fi] > vertex_max_rad[vi]:
                        vertex_max_rad[vi] = radii[fi]
            
            return radii, faces, circumcenters, vertex_max_rad
        except Exception:
            return None, None, None, None

    def gradient_descent_covering(points, max_iter=300, deadline=None, lr_init=0.02, lr_decay=0.995):
        """Gradient descent on covering radius using soft-max over face circumradii."""
        if deadline is None:
            deadline = time.time() + 100
        
        lr = lr_init
        best_local = points.copy()
        best_local_rad = compute_covering_radius(points)
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            face_radii, faces, circumcenters, vertex_max_rad = get_face_data(points)
            if face_radii is None:
                break
            
            # Soft-max weighting: emphasize large faces
            # Use temperature scaling
            temp = 0.05
            shifted = face_radii - np.max(face_radii)
            weights = np.exp(shifted / temp)
            weights /= np.sum(weights)
            
            # Compute gradient: for each face, move vertices toward circumcenter
            # Gradient of circumradius w.r.t. vertex position is proportional to
            # (vertex - circumcenter) projected onto tangent plane
            n_pts = len(points)
            grad = np.zeros_like(points)
            
            for fi in range(len(faces)):
                if weights[fi] < 1e-10:
                    continue
                cc = circumcenters[fi]
                r = face_radii[fi]
                # For a spherical triangle with circumcenter cc and radius r,
                # the gradient of r w.r.t. vertex v is roughly:
                # (v - cc * dot(v,cc)) / |v - cc*dot(v,cc)| * (1/r)
                # Simplified: move vertex toward cc in tangent direction
                for vi in faces[fi]:
                    v = points[vi]
                    # Tangent component toward circumcenter
                    dot_vc = np.dot(v, cc)
                    tangent = cc - dot_vc * v
                    tangent_norm = np.linalg.norm(tangent)
                    if tangent_norm > 1e-10:
                        # Gradient direction: moving v toward cc reduces r
                        # So gradient of r is opposite: away from cc
                        # We want to minimize r, so move in direction -grad_r
                        # grad_r ~ (v - cc_projected) / r, so -grad_r ~ (cc_projected - v) / r
                        # Which is -tangent / tangent_norm * (1/r)
                        # Actually simpler: just move toward cc
                        grad[vi] += weights[fi] * tangent / (tangent_norm * max(r, 0.1))
            
            # Add small repulsion to prevent clustering
            dot = np.dot(points, points.T)
            dot = np.clip(dot, -1.0, 1.0)
            np.fill_diagonal(dot, 1.0)
            dist_sq = 2.0 - 2.0 * dot
            dist_sq[dist_sq < 1e-10] = 1e-10
            w_rep = 1.0 / (dist_sq ** 2.0)
            np.fill_diagonal(w_rep, 0.0)
            sum_w_rep = np.sum(w_rep, axis=1)
            sum_p_w_rep = w_rep @ points
            rep_forces = points * sum_w_rep[:, np.newaxis] - sum_p_w_rep
            grad += 0.1 * rep_forces
            
            # Update
            points = points - lr * grad
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms
            lr *= lr_decay
            
            # Track best
            if it % 3 == 0:
                rad = compute_covering_radius(points)
                if rad < best_local_rad:
                    best_local_rad = rad
                    best_local = points.copy()
            
            if lr < 1e-8:
                break
        
        return best_local, best_local_rad

    def repulsion_optimize(points, max_iter=400, deadline=None):
        """Repulsion-based optimization."""
        if deadline is None:
            deadline = time.time() + 100
        
        alpha = 0.03
        decay = 0.996
        best_local = None
        best_local_rad = np.inf
        stall_count = 0
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Standard repulsion
            dot = np.dot(points, points.T)
            dot = np.clip(dot, -1.0, 1.0)
            np.fill_diagonal(dot, 1.0)
            dist_sq = 2.0 - 2.0 * dot
            dist_sq[dist_sq < 1e-10] = 1e-10
            w = 1.0 / (dist_sq ** 1.5)
            np.fill_diagonal(w, 0.0)
            
            sum_w = np.sum(w, axis=1)
            sum_p_w = w @ points
            forces = points * sum_w[:, np.newaxis] - sum_p_w
            points = points + alpha * forces
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms
            alpha *= decay
            
            # Check improvement every few iterations
            if it % 5 == 0:
                rad = compute_covering_radius(points)
                if rad < best_local_rad:
                    best_local_rad = rad
                    best_local = points.copy()
                    stall_count = 0
                else:
                    stall_count += 1
                    if stall_count > 15:
                        alpha *= 0.9
                        stall_count = 0
            
            if alpha < 1e-7:
                break
        
        return best_local if best_local is not None else points, best_local_rad

    def polish(points, n_iter=50, alpha=0.003):
        """Fine polishing with very small steps."""
        for _ in range(n_iter):
            dot = np.dot(points, points.T)
            dot = np.clip(dot, -1.0, 1.0)
            np.fill_diagonal(dot, 1.0)
            dist_sq = 2.0 - 2.0 * dot
            dist_sq[dist_sq < 1e-10] = 1e-10
            w = 1.0 / (dist_sq ** 1.5)
            np.fill_diagonal(w, 0.0)
            sum_w = np.sum(w, axis=1)
            sum_p_w = w @ points
            forces = points * sum_w[:, np.newaxis] - sum_p_w
            points = points + alpha * forces
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms
        return points

    best_points = None
    best_rad = np.inf

    # Start with Fibonacci spiral
    points = fibonacci_spiral(N).astype(np.float64)
    
    # Phase 1: Repulsion optimization from Fibonacci start
    if time.time() < deadline - 80:
        points, rad = repulsion_optimize(points, max_iter=300, deadline=deadline - 60)
        if rad < best_rad:
            best_rad = rad
            best_points = points.copy()
    
    # Phase 2: Gradient descent on covering radius
    if time.time() < deadline - 50 and best_points is not None:
        points, rad = gradient_descent_covering(best_points.copy(), max_iter=200, deadline=deadline - 30)
        if rad < best_rad:
            best_rad = rad
            best_points = points.copy()
    
    # Phase 3: Another round of repulsion from improved start
    if time.time() < deadline - 30 and best_points is not None:
        points, rad = repulsion_optimize(best_points.copy(), max_iter=200, deadline=deadline - 20)
        if rad < best_rad:
            best_rad = rad
            best_points = points.copy()
    
    # Phase 4: Final gradient descent
    if time.time() < deadline - 15 and best_points is not None:
        points, rad = gradient_descent_covering(best_points.copy(), max_iter=150, deadline=deadline - 8)
        if rad < best_rad:
            best_rad = rad
            best_points = points.copy()
    
    # Phase 5: Final polish
    if time.time() < deadline - 5 and best_points is not None:
        points = polish(best_points.copy(), n_iter=30, alpha=0.002)
        rad = compute_covering_radius(points)
        if rad < best_rad:
            best_rad = rad
            best_points = points.copy()
    
    if best_points is None:
        best_points = fibonacci_spiral(N).astype(np.float64)
    
    return best_points

# EVOLVE-BLOCK-END
