import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130

def _fibonacci_spiral(n, seed=0.0):
    """Generate n points on the unit sphere using Fibonacci spiral with optional offset."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + seed
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _compute_facet_circumradii(pts):
    """
    Compute circumradius for each facet of the convex hull.
    Returns (radii, simplices, circumcenters) or (None, None, None) on failure.
    """
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
        centers = np.zeros((len(faces), 3))
        valid = norms > 1e-10
        
        if np.any(valid):
            c_dir_valid = c_dir[valid] / norms[valid, np.newaxis]
            centers[valid] = c_dir_valid
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
            s = a_deg + b_deg + c_deg
            sn = np.linalg.norm(s, axis=1, keepdims=True)
            sn[sn < 1e-10] = 1e-10
            centers[~valid] = s / sn
        
        return radii, faces, centers
    except Exception:
        return None, None, None

def _covering_radius(pts):
    """Compute the covering radius (max facet circumradius) in radians."""
    radii, _, _ = _compute_facet_circumradii(pts)
    if radii is None:
        return np.pi
    return float(np.max(radii))

def _repulsion_optimize(pts, alpha_init=0.02, decay=0.995, max_iter=500, deadline=None):
    """Optimize points using repulsion force to spread them evenly."""
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        dot = np.dot(pts, pts.T)
        dot = np.clip(dot, -1.0, 1.0)
        np.fill_diagonal(dot, 1.0)
        dist_sq = 2.0 - 2.0 * dot
        dist_sq[dist_sq < 1e-10] = 1e-10
        
        inv_d3 = 1.0 / (dist_sq * np.sqrt(dist_sq))
        np.fill_diagonal(inv_d3, 0.0)
        
        sum_inv_d3 = np.sum(inv_d3, axis=1, keepdims=True)
        sum_pts_inv_d3 = inv_d3 @ pts
        
        grad = pts * sum_inv_d3 - sum_pts_inv_d3
        
        radial = np.sum(grad * pts, axis=1, keepdims=True)
        grad = grad - radial * pts
        
        grad_norm = np.linalg.norm(grad, axis=1, keepdims=True)
        grad_norm[grad_norm < 1e-10] = 1.0
        grad = grad / grad_norm
        
        pts = pts + alpha_init * grad
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        
        alpha_init *= decay
    
    return pts

def _facet_optimize(pts, lr_init=0.05, decay=0.995, max_iter=2000, deadline=None):
    """
    Optimize by moving each point toward the circumcenter of its worst facet.
    """
    best_pts = pts.copy()
    best_radius = _covering_radius(pts)
    
    # Precompute facet-to-point mapping for efficiency
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        radii, faces, centers = _compute_facet_circumradii(pts)
        if radii is None:
            break
        
        n_faces = len(radii)
        
        # For each point, find the worst facet containing it
        grad = np.zeros_like(pts)
        
        for j in range(N):
            mask = np.any(faces == j, axis=1)
            if not np.any(mask):
                continue
            
            facet_indices = np.where(mask)[0]
            worst_idx = facet_indices[np.argmax(radii[facet_indices])]
            worst_r = radii[worst_idx]
            
            if worst_r < 1e-10:
                continue
            
            center = centers[worst_idx]
            tangent = center - np.dot(center, pts[j]) * pts[j]
            tn = np.linalg.norm(tangent)
            if tn > 1e-10:
                tangent = tangent / tn
                grad[j] = tangent * min(worst_r, 1.0)
        
        grad_norm = np.linalg.norm(grad, axis=1, keepdims=True)
        grad_norm[grad_norm < 1e-10] = 1.0
        grad = grad / grad_norm
        
        pts = pts + lr_init * grad
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        
        current_radius = _covering_radius(pts)
        if current_radius < best_radius:
            best_radius = current_radius
            best_pts = pts.copy()
        
        lr_init *= decay
    
    return best_pts, best_radius

def _penalty_qp_optimize(pts, target_radius, penalty_weight, max_iter=200, deadline=None):
    """
    Penalty method: maximize sum of squared pairwise distances
    subject to min pairwise distance >= target_radius.
    Uses gradient ascent on the penalty objective.
    """
    best_pts = pts.copy()
    best_obj = -np.inf
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute pairwise distances
        dot = np.dot(pts, pts.T)
        dot = np.clip(dot, -1.0, 1.0)
        np.fill_diagonal(dot, 1.0)
        dist_sq = 2.0 - 2.0 * dot
        dist_sq[dist_sq < 1e-10] = 1e-10
        dist = np.sqrt(dist_sq)
        
        # Objective: sum of squared distances - penalty * sum of (target - dist)^2 for violations
        sum_sq_dist = np.sum(dist_sq)
        violations = np.maximum(0.0, target_radius - dist)
        penalty = penalty_weight * np.sum(violations ** 2)
        obj = sum_sq_dist - penalty
        
        if obj > best_obj:
            best_obj = obj
            best_pts = pts.copy()
        
        # Gradient of sum of squared distances:
        # d/d(pts_i) sum_j |pts_i - pts_j|^2 = 2 * N * pts_i - 2 * sum_j pts_j
        # But on the sphere, we project to tangent space.
        # Actually, for maximizing sum of squared distances, the gradient in R^3 is:
        # grad_i = 2 * (N * pts_i - sum_j pts_j)
        # Project to tangent plane.
        sum_pts = np.sum(pts, axis=0, keepdims=True)
        grad = 2.0 * (N * pts - sum_pts)
        radial = np.sum(grad * pts, axis=1, keepdims=True)
        grad = grad - radial * pts
        
        # Gradient of penalty term:
        # penalty = w * sum_j max(0, target - d_ij)^2 for i != j
        # d/d(pts_i) (target - d_ij)^2 = -2 * (target - d_ij) * d/d(pts_i) d_ij
        # d/d(pts_i) d_ij = (pts_i - pts_j) / d_ij
        # So grad_penalty_i = -2w * sum_j (target - d_ij) * (pts_i - pts_j) / d_ij for d_ij < target
        grad_penalty = np.zeros_like(pts)
        for j in range(N):
            viol_mask = dist[j] < target_radius
            if not np.any(viol_mask):
                continue
            viol_indices = np.where(viol_mask)[0]
            # For each violating pair (j, k), add to grad_penalty[j]
            for k in viol_indices:
                if k == j:
                    continue
                diff = target_radius - dist[j, k]
                if diff > 0:
                    direction = (pts[j] - pts[k]) / dist[j, k]
                    grad_penalty[j] += 2.0 * penalty_weight * diff * direction
        
        # Project penalty gradient to tangent space
        radial_p = np.sum(grad_penalty * pts, axis=1, keepdims=True)
        grad_penalty = grad_penalty - radial_p * pts
        
        total_grad = grad - grad_penalty
        
        # Normalize and take step
        grad_norm = np.linalg.norm(total_grad, axis=1, keepdims=True)
        grad_norm[grad_norm < 1e-10] = 1.0
        total_grad = total_grad / grad_norm
        
        step = 0.01
        pts = pts + step * total_grad
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    
    return best_pts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_pts = None
    best_radius = np.pi
    
    n_restarts = 6
    
    for restart in range(n_restarts):
        if time.time() > deadline - 25.0:
            break
        
        if restart == 0:
            pts = _fibonacci_spiral(N, seed=0.0)
        elif restart == 1:
            pts = _fibonacci_spiral(N, seed=0.5)
        elif restart == 2:
            pts = _fibonacci_spiral(N, seed=1.0)
        elif restart == 3:
            rng = np.random.RandomState(42 + restart)
            pts = rng.randn(N, 3)
            pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        elif restart == 4:
            rng = np.random.RandomState(100 + restart)
            pts = rng.randn(N, 3)
            pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        else:
            if best_pts is not None:
                rng = np.random.RandomState(200 + restart)
                pts = best_pts + 0.05 * rng.randn(N, 3)
                pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
            else:
                pts = _fibonacci_spiral(N, seed=float(restart))
        
        # Phase 1: Repulsion
        rep_deadline = deadline - 50.0
        if time.time() < rep_deadline:
            pts = _repulsion_optimize(pts, alpha_init=0.02, decay=0.995, 
                                       max_iter=300, deadline=rep_deadline)
        
        # Phase 2: Facet-based optimization
        facet_deadline = deadline - 15.0
        if time.time() < facet_deadline:
            pts, radius = _facet_optimize(pts, lr_init=0.05, decay=0.995,
                                           max_iter=1500, deadline=facet_deadline)
            
            if radius < best_radius:
                best_radius = radius
                best_pts = pts.copy()
        
        # Phase 3: Penalty QP polish
        if best_pts is not None and time.time() < deadline - 8.0:
            target = best_radius * 0.95
            pts_qp, _ = _penalty_qp_optimize(best_pts, target, penalty_weight=10.0,
                                              max_iter=100, deadline=deadline - 8.0)
            radius_qp = _covering_radius(pts_qp)
            if radius_qp < best_radius:
                best_radius = radius_qp
                best_pts = pts_qp.copy()
        
        # Phase 4: Fine-tuning
        if best_pts is not None and time.time() < deadline - 5.0:
            fine_deadline = deadline - 5.0
            pts_fine, radius_fine = _facet_optimize(best_pts, lr_init=0.01, decay=0.998,
                                                      max_iter=800, deadline=fine_deadline)
            if radius_fine < best_radius:
                best_radius = radius_fine
                best_pts = pts_fine.copy()
    
    if best_pts is None:
        best_pts = _fibonacci_spiral(N, seed=0.0)
    
    best_pts = best_pts / np.linalg.norm(best_pts, axis=1, keepdims=True)
    
    final_radius = _covering_radius(best_pts)
    if final_radius < best_radius:
        best_radius = final_radius
    
    return best_pts
