"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms


def _covering_radius_deg(pts):
    """Compute covering radius in degrees via convex hull facets."""
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 180.0
    facets = hull.simplices
    if len(facets) == 0:
        return 180.0
    
    # Vectorized computation of circumcenters
    a = pts[facets[:, 0]]
    b = pts[facets[:, 1]]
    c = pts[facets[:, 2]]
    
    # Normal vectors (not yet normalized)
    n = np.cross(b - a, c - a)
    nn = np.linalg.norm(n, axis=1, keepdims=True)
    nn = np.where(nn < 1e-10, 1.0, nn)
    n = n / nn
    
    # Ensure normals point outward (same side as centroid)
    centroids = (a + b + c) / 3.0
    dot_centroid = np.einsum('ij,ij->i', n, centroids)
    n = np.where(dot_centroid[:, None] < 0, -n, n)
    
    # Circumcenter is n (unit vector)
    # Cosine of circumradius = dot(C, a)
    cos_R = np.einsum('ij,ij->i', n, a)
    cos_R = np.clip(cos_R, -1.0, 1.0)
    R = np.arccos(cos_R)
    
    return np.degrees(np.max(R))


def _grad_covering_radius(pts):
    """
    Compute gradient of the covering radius.
    The covering radius is determined by the largest facet circumradius.
    For the facet with max circumradius, the gradient w.r.t. its three vertices
    pushes them toward the circumcenter (to reduce the circumradius).
    """
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.zeros_like(pts)
    facets = hull.simplices
    if len(facets) == 0:
        return np.zeros_like(pts)
    
    a = pts[facets[:, 0]]
    b = pts[facets[:, 1]]
    c = pts[facets[:, 2]]
    
    n = np.cross(b - a, c - a)
    nn = np.linalg.norm(n, axis=1, keepdims=True)
    nn = np.where(nn < 1e-10, 1.0, nn)
    n = n / nn
    
    centroids = (a + b + c) / 3.0
    dot_centroid = np.einsum('ij,ij->i', n, centroids)
    n = np.where(dot_centroid[:, None] < 0, -n, n)
    
    cos_R = np.einsum('ij,ij->i', n, a)
    cos_R = np.clip(cos_R, -1.0, 1.0)
    R = np.arccos(cos_R)
    
    # Find the facet(s) with maximum circumradius
    max_R = np.max(R)
    # Weight each facet by how close it is to the max (smooth max)
    beta = 50.0
    weights = np.exp(beta * (R - max_R))
    weights = weights / np.sum(weights)
    
    # For each facet, the gradient of R w.r.t. vertex a_i:
    # R = arccos(n · a), where n depends on a, b, c
    # This is complex. Instead, use a simpler approach:
    # The circumcenter C = n. To reduce R = arccos(C·a), we want to increase C·a.
    # d(C·a)/da = C (but C also depends on a, so this is approximate)
    # Project C onto the tangent plane at a: tang = C - (C·a)*a
    # Move a in direction tang to increase C·a
    
    grad = np.zeros_like(pts)
    for k in range(len(facets)):
        w = weights[k]
        if w < 1e-6:
            continue
        C = n[k]
        idx_a = facets[k, 0]
        idx_b = facets[k, 1]
        idx_c = facets[k, 2]
        
        for idx in [idx_a, idx_b, idx_c]:
            p = pts[idx]
            dot_cp = np.dot(C, p)
            dot_cp = np.clip(dot_cp, -1.0, 1.0)
            # Tangent component of C at p
            tang = C - dot_cp * p
            tn = np.linalg.norm(tang)
            if tn > 1e-10:
                tang = tang / tn
                # Moving p in direction tang increases C·p, thus decreases R
                # But we want to minimize R, so move in direction tang
                grad[idx] += w * tang
    
    return grad


def _fibonacci_spiral(n, jitter=0.0, rng=None):
    """Generate Fibonacci spiral points on sphere with optional jitter."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    if jitter > 0 and rng is not None:
        noise = rng.normal(0, jitter, size=pts.shape)
        pts = pts + noise
        pts = _normalize(pts)
    return pts


def _optimize(pts, t0, deadline):
    """Run gradient descent with adaptive step size and occasional random perturbations."""
    best_pts = pts.copy()
    best_score = _covering_radius_deg(pts)
    lr = 0.05
    rng = np.random.default_rng(hash(tuple(pts.ravel().round(6))) % 2**32)
    
    no_improve = 0
    max_no_improve = 50
    
    for iteration in range(50000):
        if time.time() > deadline:
            break
        
        grad = _grad_covering_radius(pts)
        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-10:
            # Try a random perturbation
            noise = rng.normal(0, 0.01, size=pts.shape)
            new_pts = _normalize(pts + noise)
            new_score = _covering_radius_deg(new_pts)
            if new_score < best_score:
                best_score = new_score
                best_pts = new_pts.copy()
                pts = new_pts
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > max_no_improve:
                    break
            continue
        
        grad = grad / gnorm
        
        # Update
        new_pts = pts + lr * grad
        new_pts = _normalize(new_pts)
        
        new_score = _covering_radius_deg(new_pts)
        if new_score < best_score - 1e-10:
            best_score = new_score
            best_pts = new_pts.copy()
            pts = new_pts
            no_improve = 0
            lr = min(lr * 1.02, 0.1)
        else:
            no_improve += 1
            lr *= 0.7
            if no_improve > max_no_improve:
                # Try random restart from current best
                noise = rng.normal(0, 0.05, size=pts.shape)
                pts = _normalize(best_pts + noise)
                lr = 0.05
                no_improve = 0
    
    return best_pts, best_score


def construct():
    t0 = time.time()
    deadline = 135.0  # safety margin (150s limit)
    
    # Multiple starts with different jitter levels
    jitter_levels = [0.0, 0.01, 0.02, 0.05]
    
    best_pts = None
    best_score = 180.0
    
    for jitter in jitter_levels:
        if time.time() > deadline:
            break
        
        pts = _fibonacci_spiral(N, jitter=jitter, rng=np.random.default_rng(42))
        pts_opt, score = _optimize(pts, t0, deadline)
        
        if score < best_score:
            best_score = score
            best_pts = pts_opt
    
    if best_pts is None:
        # Fallback: Fibonacci spiral
        best_pts = _fibonacci_spiral(N)
    
    return best_pts
# EVOLVE-BLOCK-END
