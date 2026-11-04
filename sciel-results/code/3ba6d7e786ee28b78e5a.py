"""Rewritten program for covering the sphere with 130 equal caps."""
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import math

N = 130
SEED = 42

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _covering_radius_deg(pts):
    """Compute covering radius in degrees using convex hull facets."""
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 180.0
    
    # For each facet, find the circumcenter (pole) and its angular distance to the facet vertices
    max_rad = 0.0
    for simplex in hull.simplices:
        facet_pts = pts[simplex]
        # Find the point on the sphere closest to the origin that is equidistant to all facet vertices
        # The pole is the normalized centroid of the facet in the tangent plane sense
        # Better: find the circumcenter of the spherical triangle
        
        # Method: The pole is the direction that maximizes the minimum angular distance to facet vertices
        # It's the intersection of the sphere with the line from origin through the circumcenter of the planar triangle
        # For a spherical triangle, the covering point is the spherical circumcenter
        
        # Compute the circumcenter of the planar triangle formed by the 3 points
        a, b, c = facet_pts[0], facet_pts[1], facet_pts[2]
        
        # Solve for the circumcenter in 3D
        # The circumcenter lies on the plane perpendicular to each edge through its midpoint
        # Or: find point p such that |p-a|^2 = |p-b|^2 = |p-c|^2
        # This gives linear equations: 2(b-a)·p = |b|^2 - |a|^2, etc.
        
        # Since points are on unit sphere, |a|=|b|=|c|=1, so RHS = 0
        # 2(b-a)·p = 0, 2(c-a)·p = 0
        # p is perpendicular to both (b-a) and (c-a), so p is along the cross product
        v1 = b - a
        v2 = c - a
        pole = np.cross(v1, v2)
        pole_norm = np.linalg.norm(pole)
        if pole_norm < 1e-12:
            continue
        pole = pole / pole_norm
        
        # Check that this pole is on the correct side (outside the convex hull)
        # The pole should be in the direction away from the origin relative to the facet
        # Actually, for a convex hull of points on the sphere, the facet's outward normal
        # is the direction from the origin through the facet's centroid (roughly)
        centroid = np.mean(facet_pts, axis=0)
        if np.dot(pole, centroid) < 0:
            pole = -pole
        
        # The covering radius for this facet is the angular distance from pole to each vertex
        # All should be equal (it's the circumcenter)
        cos_dist = np.dot(facet_pts, pole)
        cos_dist = np.clip(cos_dist, -1.0, 1.0)
        dists = np.arccos(cos_dist)
        facet_rad = np.max(dists)
        max_rad = max(max_rad, facet_rad)
    
    return np.degrees(max_rad)

def _logsumexp_surrogate(pts, beta):
    """
    Smooth surrogate for the covering radius using log-sum-exp.
    We maximize the minimum separation, which is equivalent to minimizing the covering radius.
    
    Actually, let's use a different approach: minimize the max over all pairs of the
    angular distance, but that's the Tammes problem. For covering, we need the dual.
    
    Let's use the epigraph formulation with a smooth approximation.
    """
    pts = _normalize(pts)
    # Compute all pairwise angular distances
    cos_dist = np.clip(pts @ pts.T, -1.0, 1.0)
    dists = np.arccos(cos_dist)
    
    # For covering radius, we want to minimize the maximum distance from any sphere point
    # to its nearest site. This is related to the Delaunay triangulation.
    # 
    # A practical smooth surrogate: minimize the sum of exp(beta * d_ij) over all pairs
    # This encourages uniform spacing.
    
    # Exclude self-distances
    mask = ~np.eye(N, dtype=bool)
    d = dists[mask]
    
    # Log-sum-exp for numerical stability
    d_max = np.max(d)
    log_sum = np.log(np.sum(np.exp(beta * (d - d_max)))) + beta * d_max
    return log_sum

def _tangent_space_gradient(pts, beta):
    """
    Compute gradient of the log-sum-exp surrogate in tangent space.
    For each point i, the gradient is the sum over j of exp(beta*d_ij) * (grad d_ij / |grad d_ij|)
    projected onto the tangent space at point i.
    """
    pts = _normalize(pts)
    cos_dist = np.clip(pts @ pts.T, -1.0, 1.0)
    dists = np.arccos(cos_dist)
    
    # We want to minimize sum_j exp(beta * d_ij) for each i
    # grad d_ij / d theta_i = (cos_dist_ij) / sin(d_ij) * (direction)
    # Actually, for small perturbations, the gradient of d_ij with respect to point i
    # in tangent space is: -cos(d_ij)/sin(d_ij) * (p_j - (p_j·p_i)p_i)/|p_j - (p_j·p_i)p_i|
    
    grads = np.zeros_like(pts)
    for i in range(N):
        pi = pts[i]
        for j in range(N):
            if i == j:
                continue
            d = dists[i, j]
            if d < 1e-10 or d > np.pi - 1e-10:
                continue
            w = np.exp(beta * d)
            # Tangent vector from i to j
            proj = np.dot(pts[j], pi) * pi
            tangent = pts[j] - proj
            t_norm = np.linalg.norm(tangent)
            if t_norm < 1e-12:
                continue
            tangent = tangent / t_norm
            # Gradient of d_ij w.r.t. pi is -cos(d)/sin(d) * tangent
            # But we want to minimize, so we move in the direction that increases d
            # Actually, to maximize separation, we want to move away from neighbors
            # grad of -d_ij (to maximize) is cos(d)/sin(d) * tangent
            coeff = np.cos(d) / np.sin(d)
            grads[i] += w * coeff * tangent
    
    return grads

def _project_to_sphere(pts):
    return _normalize(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120s limit, leave margin
    
    np.random.seed(SEED)
    
    # Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
    pts = _normalize(np.column_stack([r * np.cos(phi), r * np.sin(phi), z]))
    
    best_pts = pts.copy()
    best_cr = _covering_radius_deg(pts)
    
    # Phase 1: L-BFGS-B optimization using log-sum-exp surrogate
    beta = 50.0  # Sharpness of the surrogate
    
    def objective(x):
        p = x.reshape(N, 3)
        p = _normalize(p)
        # Use log-sum-exp of pairwise distances
        cos_d = np.clip(p @ p.T, -1.0, 1.0)
        d = np.arccos(cos_d)
        mask = ~np.eye(N, dtype=bool)
        dd = d[mask]
        d_max = np.max(dd)
        return np.log(np.sum(np.exp(beta * (dd - d_max)))) + beta * d_max
    
    # Run L-BFGS-B
    try:
        result = minimize(
            objective,
            pts.flatten(),
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        if result.success or result.fun < objective(pts.flatten()):
            pts = _normalize(result.x.reshape(N, 3))
            cr = _covering_radius_deg(pts)
            if cr < best_cr:
                best_pts = pts.copy()
                best_cr = cr
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 30:
        return best_pts
    
    # Phase 2: Local refinement focusing on worst facets
    # Iteratively find the worst facet and move the nearby points to improve it
    for iteration in range(200):
        if time.time() > deadline - 10:
            break
        
        pts = _normalize(pts)
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        # Find the worst facet
        worst_idx = -1
        worst_rad = 0
        for fi, simplex in enumerate(hull.simplices):
            facet_pts = pts[simplex]
            a, b, c = facet_pts[0], facet_pts[1], facet_pts[2]
            v1 = b - a
            v2 = c - a
            pole = np.cross(v1, v2)
            pole_norm = np.linalg.norm(pole)
            if pole_norm < 1e-12:
                continue
            pole = pole / pole_norm
            centroid = np.mean(facet_pts, axis=0)
            if np.dot(pole, centroid) < 0:
                pole = -pole
            cos_dist = np.clip(np.dot(facet_pts, pole), -1.0, 1.0)
            dists = np.arccos(cos_dist)
            facet_rad = np.max(dists)
            if facet_rad > worst_rad:
                worst_rad = facet_rad
                worst_idx = fi
        
        if worst_idx < 0:
            break
        
        # Get the vertices of the worst facet
        simplex = hull.simplices[worst_idx]
        facet_indices = simplex
        
        # Compute the pole
        a, b, c = pts[facet_indices[0]], pts[facet_indices[1]], pts[facet_indices[2]]
        v1 = b - a
        v2 = c - a
        pole = np.cross(v1, v2)
        pole_norm = np.linalg.norm(pole)
        if pole_norm < 1e-12:
            break
        pole = pole / pole_norm
        centroid = np.mean(pts[facet_indices], axis=0)
        if np.dot(pole, centroid) < 0:
            pole = -pole
        
        # Move the facet vertices slightly toward the pole direction in tangent space
        # to reduce the circumradius
        step = 0.01
        for idx in facet_indices:
            pi = pts[idx]
            # Tangent direction toward pole
            proj = np.dot(pole, pi) * pi
            tangent = pole - proj
            t_norm = np.linalg.norm(tangent)
            if t_norm < 1e-12:
                continue
            tangent = tangent / t_norm
            pts[idx] = pi + step * tangent
        
        pts = _normalize(pts)
        cr = _covering_radius_deg(pts)
        if cr < best_cr:
            best_pts = pts.copy()
            best_cr = cr
    
    # Phase 3: Final high-precision polish
    if time.time() < deadline - 5:
        try:
            beta_final = 100.0
            def objective_final(x):
                p = x.reshape(N, 3)
                p = _normalize(p)
                cos_d = np.clip(p @ p.T, -1.0, 1.0)
                d = np.arccos(cos_d)
                mask = ~np.eye(N, dtype=bool)
                dd = d[mask]
                d_max = np.max(dd)
                return np.log(np.sum(np.exp(beta_final * (dd - d_max)))) + beta_final * d_max
            
            result = minimize(
                objective_final,
                best_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if result.fun < objective_final(best_pts.flatten()):
                p = _normalize(result.x.reshape(N, 3))
                cr = _covering_radius_deg(p)
                if cr < best_cr:
                    best_pts = p
        except Exception:
            pass
    
    return _normalize(best_pts)
