"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Leave 15s margin for safety
    
    def fibonacci_spiral(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    def compute_covering_radius(pts):
        """Compute covering radius as max circumradius over convex hull faces (vectorized)."""
        try:
            hull = ConvexHull(pts)
            faces = hull.simplices  # shape (F, 3)
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
    
    def compute_face_radii_and_circumcenters(pts):
        """Return arrays of face circumradii and circumcenters for gradient-aware optimization."""
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
            circumcenters = np.zeros((len(faces), 3))
            valid = norms > 1e-10
            
            if np.any(valid):
                cc = c_dir[valid] / norms[valid, np.newaxis]
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
                # Approximate circumcenter for degenerate faces
                circumcenters[~valid] = (a_deg + b_deg + c_deg) / 3.0
                norms_cc = np.linalg.norm(circumcenters[~valid], axis=1, keepdims=True)
                circumcenters[~valid] /= norms_cc
            
            return radii, circumcenters, faces
        except Exception:
            return None, None, None
    
    def facet_gradient_step(points, alpha, radii, circumcenters, faces, beta=20.0):
        """
        Move each vertex toward the circumcenters of facets it belongs to,
        weighted by a soft-max over facet radii to smooth the max.
        Vectorized for speed.
        """
        n_points = len(points)
        n_faces = len(faces)
        forces = np.zeros((n_points, 3))
        
        if n_faces == 0:
            return points
        
        # Soft-max weights: w_i = exp(beta * r_i) / sum(exp(beta * r_j))
        # This makes facets with larger radii get more weight
        max_r = np.max(radii)
        weights = np.exp(beta * (radii - max_r))
        weights /= np.sum(weights)
        
        # Vectorized: for each face, compute force on each of its 3 vertices
        # faces shape: (F, 3), points shape: (N, 3), circumcenters shape: (F, 3)
        # For vertex v in face f: force = w_f * (cc_f - p_v) - w_f * dot(cc_f - p_v, p_v) * p_v
        
        # Get the 3 vertices of each face
        v0 = faces[:, 0]
        v1 = faces[:, 1]
        v2 = faces[:, 2]
        
        p0 = points[v0]  # (F, 3)
        p1 = points[v1]
        p2 = points[v2]
        
        cc = circumcenters  # (F, 3)
        w = weights[:, np.newaxis]  # (F, 1)
        
        # Force for vertex 0 of each face
        f0 = w * (cc - p0)
        radial0 = np.sum(f0 * p0, axis=1, keepdims=True)
        f0 = f0 - radial0 * p0
        
        # Force for vertex 1 of each face
        f1 = w * (cc - p1)
        radial1 = np.sum(f1 * p1, axis=1, keepdims=True)
        f1 = f1 - radial1 * p1
        
        # Force for vertex 2 of each face
        f2 = w * (cc - p2)
        radial2 = np.sum(f2 * p2, axis=1, keepdims=True)
        f2 = f2 - radial2 * p2
        
        # Accumulate forces using np.add.at for handling duplicate vertices
        np.add.at(forces, v0, f0)
        np.add.at(forces, v1, f1)
        np.add.at(forces, v2, f2)
        
        points = points + alpha * forces
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        return points
    
    def repulsion_step(points, alpha):
        """Standard repulsion step to prevent clustering. Uses a softer kernel for stability."""
        dot = np.dot(points, points.T)
        dot = np.clip(dot, -1.0, 1.0)
        np.fill_diagonal(dot, 1.0)
        dist_sq = 2.0 - 2.0 * dot
        dist_sq[dist_sq < 1e-10] = 1e-10
        # Softer kernel: 1/dist_sq instead of 1/dist_sq^1.5 for better stability
        w = 1.0 / dist_sq
        np.fill_diagonal(w, 0.0)
        sum_w = np.sum(w, axis=1)
        sum_p_w = w @ points
        forces = points * sum_w[:, np.newaxis] - sum_p_w
        points = points + alpha * forces
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        return points
    
    def polish_step(points, alpha, n_iter=50):
        """High-precision polish with very small steps and adaptive beta."""
        for it in range(n_iter):
            radii, ccs, faces = compute_face_radii_and_circumcenters(points)
            if radii is not None:
                # Adaptive beta: increase as spread decreases to sharpen focus on worst facets
                r_spread = np.max(radii) - np.min(radii)
                beta = 20.0 + 50.0 * np.exp(-r_spread * 20.0)
                points = facet_gradient_step(points, alpha, radii, ccs, faces, beta=beta)
            # Small repulsion to maintain separation
            points = repulsion_step(points, alpha * 0.05)
        return points
    
    best_points = None
    best_rad = np.inf
    
    n_restarts = 6
    
    for restart in range(n_restarts):
        if time.time() > deadline - 20.0:
            break
        
        if restart == 0:
            points = fibonacci_spiral(N).astype(np.float64)
        elif restart == 1:
            points = np.random.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        else:
            points = best_points + 0.10 * np.random.randn(N, 3)
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms
        
        alpha = 0.03
        decay = 0.997
        max_iter = 800
        stall_count = 0
        last_rad = np.inf
        
        for it in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            
            # Compute facet data
            radii, ccs, faces = compute_face_radii_and_circumcenters(points)
            
            if radii is not None:
                # Adaptive beta for main loop
                r_spread = np.max(radii) - np.min(radii)
                beta = 15.0 + 30.0 * np.exp(-r_spread * 15.0)
                # Facet gradient step
                points = facet_gradient_step(points, alpha, radii, ccs, faces, beta=beta)
                # Small repulsion to prevent collapse
                points = repulsion_step(points, alpha * 0.03)
            
            alpha *= decay
            
            if it % 3 == 0:
                rad = compute_covering_radius(points)
                if rad < best_rad:
                    best_rad = rad
                    best_points = points.copy()
                    stall_count = 0
                else:
                    stall_count += 1
                    if stall_count > 40:
                        alpha *= 0.85
                        stall_count = 0
            
            if alpha < 1e-8:
                break
        
        # Polish phase
        if time.time() < deadline - 10.0 and best_points is not None:
            polished = polish_step(best_points.copy(), 0.002, 80)
            rad_polished = compute_covering_radius(polished)
            if rad_polished < best_rad:
                best_rad = rad_polished
                best_points = polished
    
    # Final global polish with multiple stages
    if time.time() < deadline - 5.0 and best_points is not None:
        final = polish_step(best_points.copy(), 0.0015, 100)
        rad_final = compute_covering_radius(final)
        if rad_final < best_rad:
            best_rad = rad_final
            best_points = final
        
        if time.time() < deadline - 3.0 and best_points is not None:
            final2 = polish_step(best_points.copy(), 0.0008, 120)
            rad_final2 = compute_covering_radius(final2)
            if rad_final2 < best_rad:
                best_rad = rad_final2
                best_points = final2
    
    if best_points is None:
        best_points = fibonacci_spiral(N).astype(np.float64)
    
    # Ensure all points are on the unit sphere
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    best_points = best_points / norms
    
    return best_points

# EVOLVE-BLOCK-END
