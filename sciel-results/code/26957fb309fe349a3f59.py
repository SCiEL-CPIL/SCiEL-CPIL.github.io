"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist

N = 50
_SEED = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.spatial import ConvexHull
import time


def _normalize(points):
    """Normalize rows to unit length, handling zero vectors."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return points / norms


def _covering_radius(pts):
    """
    Compute the covering radius (in radians) using convex hull facets.
    For each triangular facet, the circumcenter is the normalized normal
    to the facet plane. The angular radius is arccos(|n·v|).
    """
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        # Fallback: brute force on a fine grid
        n_theta = 30
        n_phi = 60
        theta = np.linspace(0, np.pi, n_theta)
        phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
        T, P = np.meshgrid(theta, phi, indexing='ij')
        Y = np.column_stack([
            np.sin(T) * np.cos(P).ravel(),
            np.sin(T) * np.sin(P).ravel(),
            np.cos(T).ravel()
        ])
        Y = _normalize(Y)
        dots = Y @ pts.T
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)
        min_dists = np.min(dists, axis=1)
        return np.max(min_dists)
    
    max_rad = 0.0
    for simplex in hull.simplices:
        v1, v2, v3 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        n = np.cross(v1 - v2, v1 - v3)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-12:
            continue
        n = n / norm_n
        dot_val = abs(n @ v1)
        dot_val = min(dot_val, 1.0)
        rad = np.arccos(dot_val)
        if rad > max_rad:
            max_rad = rad
    
    return max_rad


def _covering_radius_degrees(pts):
    """Covering radius in degrees."""
    return np.degrees(_covering_radius(pts))


def _soft_covering_radius(pts, beta=50.0):
    """
    Smooth approximation of the covering radius using log-sum-exp over a grid.
    """
    pts = _normalize(pts)
    n_theta = 25
    n_phi = 50
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    T, P = np.meshgrid(theta, phi, indexing='ij')
    Y = np.column_stack([
        np.sin(T) * np.cos(P).ravel(),
        np.sin(T) * np.sin(P).ravel(),
        np.cos(T).ravel()
    ])
    Y = _normalize(Y)
    dots = Y @ pts.T
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)
    min_dists = np.min(dists, axis=1)
    max_d = np.max(min_dists)
    soft_max = max_d + np.log(np.sum(np.exp(beta * (min_dists - max_d)))) / beta
    return soft_max


def _fibonacci_spiral():
    """Generate initial points using Fibonacci spiral distribution."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _repulsion_energy(pts, power=20):
    """
    Sum of inverse powers of pairwise distances (repulsion energy).
    Maximizing this (or minimizing its negative) spreads points apart.
    """
    pts = _normalize(pts)
    dots = pts @ pts.T
    np.fill_diagonal(dots, 0)
    dots = np.clip(dots, -1.0, 0.999999)
    dists = np.arccos(dots)
    # Avoid division by zero
    dists = np.maximum(dists, 1e-10)
    energy = np.sum(1.0 / (dists ** power))
    return energy


def _repulsion_gradient(pts, power=20):
    """
    Gradient of repulsion energy w.r.t. points (tangent space).
    """
    pts = _normalize(pts)
    dots = pts @ pts.T
    np.fill_diagonal(dots, 0)
    dots = np.clip(dots, -1.0, 0.999999)
    dists = np.arccos(dots)
    dists = np.maximum(dists, 1e-10)
    
    # d(1/d^power)/d(d) = -power / d^(power+1)
    # d(d)/d(pts) = -pts_j / d  (for j != i)
    # So d(energy)/d(pts_i) = sum_j -power/d_ij^(power+1) * (-pts_j / d_ij)
    #                        = sum_j power * pts_j / d_ij^(power+2)
    
    coeff = power / (dists ** (power + 2))
    np.fill_diagonal(coeff, 0)
    
    grad = coeff @ pts  # (N, 3)
    
    # Project to tangent space
    radial = (grad * pts).sum(axis=1, keepdims=True)
    grad_tangent = grad - radial * pts
    return grad_tangent


def _polish_facet_shrink(pts, max_iters=500, tol=1e-10):
    """
    Iteratively shrink the largest facet by moving points toward the
    circumcenter of that facet.
    """
    pts = _normalize(pts)
    for it in range(max_iters):
        rad = _covering_radius(pts)
        if rad < tol:
            break
        
        # Find the facet with the largest circumradius
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        max_facet_rad = 0.0
        max_facet = None
        for simplex in hull.simplices:
            v1, v2, v3 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
            n = np.cross(v1 - v2, v1 - v3)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-12:
                continue
            n = n / norm_n
            dot_val = abs(n @ v1)
            dot_val = min(dot_val, 1.0)
            rad_f = np.arccos(dot_val)
            if rad_f > max_facet_rad:
                max_facet_rad = rad_f
                max_facet = simplex
        
        if max_facet is None:
            break
        
        # The circumcenter direction is the normalized normal
        v1, v2, v3 = pts[max_facet[0]], pts[max_facet[1]], pts[max_facet[2]]
        n = np.cross(v1 - v2, v1 - v3)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-12:
            break
        n = n / norm_n
        
        # Choose the sign that points outward (away from origin, toward the "gap")
        # The circumcenter should be on the side of the facet that is the "empty cap"
        # We want to move points toward the circumcenter to reduce the facet size
        # The circumcenter is at ±n; we want the one that is farthest from all points
        # Actually, we want to move the vertices of the facet toward the circumcenter
        
        # Move each vertex of the facet toward the circumcenter
        # Use a small step size
        step = 0.01
        for idx in max_facet:
            # Move point idx toward n (the circumcenter direction)
            # Use spherical interpolation (slerp)
            p = pts[idx]
            dot = p @ n
            dot = np.clip(dot, -1.0, 1.0)
            angle = np.arccos(dot)
            if angle < 1e-10:
                continue
            # Slerp by a fraction of the angle
            new_angle = angle * (1.0 - step)
            # New point: sin(new_angle)/sin(angle) * p + (cos(new_angle) - dot*sin(new_angle)/sin(angle)) * n
            sin_a = np.sin(angle)
            sin_na = np.sin(new_angle)
            cos_na = np.cos(new_angle)
            if sin_a < 1e-10:
                continue
            coeff_p = sin_na / sin_a
            coeff_n = (cos_na - dot * sin_na / sin_a)
            new_p = coeff_p * p + coeff_n * n
            new_p = new_p / np.linalg.norm(new_p)
            pts[idx] = new_p
        
        pts = _normalize(pts)
    
    return pts


def _optimize_with_beta_cascade(pts, betas=[15, 50, 150, 300, 500], max_iter_per_stage=2000):
    """
    Multi-stage beta cascade optimization.
    Start with low beta (broad landscape) and increase to high beta (precise).
    """
    pts = _normalize(pts)
    for beta in betas:
        def obj(x):
            p = _normalize(x.reshape(N, 3))
            return _soft_covering_radius(p, beta=beta)
        
        try:
            res = minimize(
                obj,
                pts.ravel(),
                method='Nelder-Mead',
                options={
                    'maxiter': max_iter_per_stage,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            pts = _normalize(res.x.reshape(N, 3))
        except Exception:
            pass
    
    return pts


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial point: Fibonacci spiral
    best_pts = _fibonacci_spiral()
    best_radius = _covering_radius(best_pts)
    
    np.random.seed(_SEED)
    
    # Phase 1: Repulsion-based optimization to spread points
    # Maximize repulsion energy (minimize negative)
    for restart in range(3):
        if time.time() > deadline - 60.0:
            break
        
        if restart == 0:
            x0 = best_pts.copy()
        else:
            x0 = np.random.randn(N, 3)
            x0 = _normalize(x0)
        
        # Use Nelder-Mead to maximize repulsion (minimize negative)
        def obj_repulsion(x):
            p = _normalize(x.reshape(N, 3))
            return -_repulsion_energy(p, power=10)
        
        try:
            res = minimize(
                obj_repulsion,
                x0.ravel(),
                method='Nelder-Mead',
                options={
                    'maxiter': 3000,
                    'xatol': 1e-6,
                    'fatol': 1e-8,
                    'adaptive': True
                }
            )
            x_opt = _normalize(res.x.reshape(N, 3))
            rad = _covering_radius(x_opt)
            if rad < best_radius:
                best_radius = rad
                best_pts = x_opt.copy()
        except Exception:
            continue
    
    # Phase 2: Beta cascade optimization
    if time.time() < deadline - 40.0:
        polished = _optimize_with_beta_cascade(best_pts, betas=[15, 50, 150], max_iter_per_stage=1500)
        rad = _covering_radius(polished)
        if rad < best_radius:
            best_radius = rad
            best_pts = polished.copy()
    
    # Phase 3: Facet shrink polishing
    if time.time() < deadline - 20.0:
        polished = _polish_facet_shrink(best_pts, max_iters=300)
        rad = _covering_radius(polished)
        if rad < best_radius:
            best_radius = rad
            best_pts = polished.copy()
    
    # Phase 4: Final high-beta polish
    if time.time() < deadline - 10.0:
        polished = _optimize_with_beta_cascade(best_pts, betas=[300, 500], max_iter_per_stage=1000)
        rad = _covering_radius(polished)
        if rad < best_radius:
            best_radius = rad
            best_pts = polished.copy()
    
    # Phase 5: Final facet shrink
    if time.time() < deadline - 5.0:
        polished = _polish_facet_shrink(best_pts, max_iters=200)
        rad = _covering_radius(polished)
        if rad < best_radius:
            best_radius = rad
            best_pts = polished.copy()
    
    return best_pts
# EVOLVE-BLOCK-END
