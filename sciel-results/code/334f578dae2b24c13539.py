"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull

_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0  # safety margin before 150s kill

def _to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms

def _covering_radius(pts):
    """
    Compute the covering radius (in radians) of a set of points on the sphere.
    Uses convex hull facets: for each facet, the circumcenter on the sphere
    gives a candidate for the largest empty circle center.
    """
    pts = _to_sphere(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return _covering_radius_fallback(pts)
    
    max_radius = 0.0
    for simplex in hull.simplices:
        a = pts[simplex[0]]
        b = pts[simplex[1]]
        c = pts[simplex[2]]
        
        v1 = a - b
        v2 = a - c
        cross = np.cross(v1, v2)
        norm_cross = np.linalg.norm(cross)
        
        if norm_cross < 1e-12:
            continue
        
        x = cross / norm_cross
        
        dot_a = np.dot(x, a)
        dot_a = np.clip(dot_a, -1.0, 1.0)
        r1 = np.arccos(dot_a)
        
        dot_a_neg = -dot_a
        dot_a_neg = np.clip(dot_a_neg, -1.0, 1.0)
        r2 = np.arccos(dot_a_neg)
        
        candidate = max(r1, r2)
        if candidate > max_radius:
            max_radius = candidate
    
    # Check antipodes of each point
    for i in range(len(pts)):
        neg_pt = -pts[i]
        dots = pts @ neg_pt
        dots = np.clip(dots, -1.0, 1.0)
        min_dist = np.arccos(np.min(dots))
        if min_dist > max_radius:
            max_radius = min_dist
    
    return max_radius

def _covering_radius_fallback(pts):
    """Fallback covering radius computation via moderate sampling."""
    n_theta = 120
    n_phi = 240
    thetas = np.linspace(0, np.pi, n_theta)
    phis = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    
    max_radius = 0.0
    for theta in thetas:
        st = np.sin(theta)
        ct = np.cos(theta)
        y_batch = np.column_stack([
            st * np.cos(phis),
            st * np.sin(phis),
            np.full_like(phis, ct)
        ])
        dots = pts @ y_batch.T
        dots = np.clip(dots, -1.0, 1.0)
        min_dists = np.arccos(np.min(dots, axis=0))
        max_radius = max(max_radius, np.max(min_dists))
    return max_radius

def _objective(pts):
    """Compute the objective (covering radius in radians)."""
    return _covering_radius(pts)

def _psd_project(G, n=50):
    """Project a symmetric matrix onto the cone of PSD matrices with unit diagonal."""
    eigvals, eigvecs = np.linalg.eigh(G)
    eigvals = np.maximum(eigvals, 0.0)
    G_new = eigvecs @ np.diag(eigvals) @ eigvecs.T
    # Symmetrize
    G_new = (G_new + G_new.T) / 2.0
    # Enforce unit diagonal
    np.fill_diagonal(G_new, 1.0)
    return G_new

def _gram_optimize(G, max_time=30.0):
    """
    Optimize the Gram matrix to maximize the minimum off-diagonal entry.
    Uses smooth max-approximation gradient ascent with PSD projection.
    """
    n = G.shape[0]
    G = G.copy()
    # Ensure initial validity
    G = _psd_project(G, n)
    
    # Compute initial min off-diagonal
    G_min_diag = G.copy()
    np.fill_diagonal(G_min_diag, 2.0)  # Large value to ignore diagonal
    current_min = np.min(G_min_diag)
    
    best_G = G.copy()
    best_min = current_min
    
    start = time.time()
    step_size = 0.01
    beta = 50.0  # Smoothness parameter for soft-min
    no_improve = 0
    
    while time.time() - start < max_time:
        # Compute soft-min weights: emphasize the smallest entries
        G_off = G.copy()
        np.fill_diagonal(G_off, 2.0)
        
        # Shift and scale for numerical stability in softmax
        G_shifted = G_off - current_min
        weights = np.exp(beta * G_shifted)
        
        # Normalize weights to sum to 1 for each pair (symmetric)
        # We want a gradient that pushes up the minimum entries
        # grad = sum_j w_ij * (E_ii - E_ij - E_ji + E_jj) ... but diag is fixed
        # Simpler: gradient of soft-min is -w (we want to maximize, so we add)
        # Actually, we want to increase the minimum. 
        # f(G) = min_ij G_ij. 
        # grad f ≈ -w (where w is the weight on the min).
        # Since we want to MAXIMIZE, we do G += step * grad_max.
        # grad_max = w (normalized indicator of the min).
        
        # Let's use the gradient of the smooth min:
        # m(G) = -1/beta * log(sum(exp(-beta * G_off)))
        # grad m = (exp(-beta * G_off) / sum(exp(-beta * G_off)))
        # We want to maximize m, so we ascend: G += step * grad_m
        
        exp_neg = np.exp(-beta * G_off)
        sum_exp = np.sum(exp_neg)
        if sum_exp == 0:
            break
        grad = exp_neg / sum_exp
        
        # Ensure diagonal is zero in gradient (we don't change diagonal directly in update, 
        # but projection fixes it. However, to be safe, zero out diag in grad)
        np.fill_diagonal(grad, 0.0)
        
        # Update
        G_new = G + step_size * grad
        G_new = _psd_project(G_new, n)
        
        # Check improvement
        G_new_min_diag = G_new.copy()
        np.fill_diagonal(G_new_min_diag, 2.0)
        new_min = np.min(G_new_min_diag)
        
        if new_min > current_min:
            G = G_new
            current_min = new_min
            if current_min > best_min:
                best_min = current_min
                best_G = G.copy()
            no_improve = 0
            # Slowly increase beta to sharpen focus on the true minimum
            beta = min(beta * 1.01, 200.0)
            step_size = min(step_size * 1.005, 0.05)
        else:
            no_improve += 1
            step_size *= 0.95
            if no_improve > 100:
                break
        
        if time.time() - start > max_time:
            break
    
    return best_G, best_min

def _eigh_to_points(G):
    """Convert Gram matrix to points on the sphere (if possible)."""
    eigvals, eigvecs = np.linalg.eigh(G)
    # Take the top 3 eigenvectors for embedding in R3
    # Sort by eigenvalue descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    # If the 4th eigenvalue is significantly non-zero, the matrix is not rank 3.
    # We project onto the top 3.
    V = eigvecs[:, :3]
    D = np.diag(np.sqrt(np.maximum(eigvals[:3], 0)))
    
    # Points are rows of V @ D
    pts = V @ D
    
    # Normalize to unit sphere
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    pts = pts / norms
    
    return pts

def _local_polish(pts, max_time=10.0):
    """
    Local polish on point configuration using convex hull gradient.
    Moves points away from the center of the worst facet.
    """
    pts = _to_sphere(pts.copy())
    current_obj = _objective(pts)
    best_pts = pts.copy()
    best_obj = current_obj
    
    start = time.time()
    step_size = 0.01
    no_improve = 0
    
    while time.time() - start < max_time:
        # Find worst facet
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        max_r = 0
        worst_normal = None
        worst_idx = 0
        
        for i, simplex in enumerate(hull.simplices):
            a = pts[simplex[0]]
            b = pts[simplex[1]]
            c = pts[simplex[2]]
            
            v1 = a - b
            v2 = a - c
            cross = np.cross(v1, v2)
            norm_cross = np.linalg.norm(cross)
            
            if norm_cross < 1e-12:
                continue
            
            x = cross / norm_cross
            
            dot_a = np.dot(x, a)
            dot_a = np.clip(dot_a, -1.0, 1.0)
            r1 = np.arccos(dot_a)
            
            dot_a_neg = -dot_a
            dot_a_neg = np.clip(dot_a_neg, -1.0, 1.0)
            r2 = np.arccos(dot_a_neg)
            
            candidate = max(r1, r2)
            if candidate > max_r:
                max_r = candidate
                worst_normal = x
                worst_idx = i
        
        if worst_normal is None:
            break
        
        # Force: move all points away from the worst facet center?
        # Or just move the points forming the facet?
        # Actually, the covering radius is determined by the empty cap centered at worst_normal.
        # To reduce it, we should move points *towards* worst_normal? No, that would shrink the cap?
        # Wait. The cap is centered at worst_normal. The points defining the facet are on the boundary.
        # To reduce the radius (distance from worst_normal to points), we need to move points CLOSER to worst_normal.
        # But moving points closer to worst_normal might create a larger cap elsewhere.
        
        # Let's try moving the 3 points of the worst facet towards the normal.
        simplex = hull.simplices[worst_idx]
        forces = np.zeros_like(pts)
        for idx in simplex:
            # Move point idx towards worst_normal
            # Tangential component of (worst_normal - pts[idx])
            diff = worst_normal - pts[idx]
            tangential = diff - np.dot(diff, pts[idx]) * pts[idx]
            forces[idx] = tangential
        
        # Also add a small repulsive force from all other points to maintain spread?
        # Let's just do the facet move first.
        
        new_pts = pts + step_size * forces
        new_pts = _to_sphere(new_pts)
        
        new_obj = _objective(new_pts)
        
        if new_obj < current_obj:
            pts = new_pts
            current_obj = new_obj
            if current_obj < best_obj:
                best_obj = current_obj
                best_pts = pts.copy()
            no_improve = 0
            step_size = min(step_size * 1.01, 0.05)
        else:
            no_improve += 1
            step_size *= 0.95
            if no_improve > 50:
                break
    
    return best_pts, best_obj

def construct():
    """
    Construct 50 points on the unit sphere minimizing the covering radius.
    """
    deadline = _DEADLINE
    
    # Phase 1: Initialize Gram Matrix
    # Start with a random PSD matrix with unit diagonal
    # Or start from a known good configuration (Fibonacci) and compute its Gram matrix
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts_init = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts_init = _to_sphere(pts_init)
    
    G_init = pts_init @ pts_init.T
    G_init = (G_init + G_init.T) / 2.0
    np.fill_diagonal(G_init, 1.0)
    
    best_pts = pts_init.copy()
    best_obj = _objective(pts_init)
    
    remaining = deadline - time.time()
    
    # Phase 2: Gram Matrix Optimization
    # We will run multiple restarts with different initializations or perturbations
    np.random.seed(42)
    
    for restart in range(5):
        if time.time() > deadline - 20:
            break
        
        # Perturb the initial Gram matrix
        # Add small noise to off-diagonals
        noise = np.random.randn(N, N) * 0.05
        noise = (noise + noise.T) / 2.0
        np.fill_diagonal(noise, 0.0)
        
        G_perturbed = G_init + noise
        G_perturbed = _psd_project(G_perturbed, N)
        
        # Optimize
        G_opt, _ = _gram_optimize(G_perturbed, max_time=min(remaining - 20, 20))
        
        # Convert to points
        pts_from_gram = _eigh_to_points(G_opt)
        
        # Local polish
        if time.time() < deadline - 10:
            pts_polished, obj_polished = _local_polish(pts_from_gram, max_time=min(remaining - 10, 10))
            if obj_polished < best_obj:
                best_obj = obj_polished
                best_pts = pts_polished.copy()
        
        # Also check the unpolished points
        obj_raw = _objective(pts_from_gram)
        if obj_raw < best_obj:
            best_obj = obj_raw
            best_pts = pts_from_gram.copy()
            
        remaining = deadline - time.time()
    
    # Phase 3: Final Polish on Best
    if time.time() < deadline - 5:
        pts_final, obj_final = _local_polish(best_pts, max_time=min(remaining - 5, 10))
        if obj_final < best_obj:
            best_obj = obj_final
            best_pts = pts_final.copy()
    
    # Ensure all points are on the unit sphere
    best_pts = _to_sphere(best_pts)
    
    # Validate: check for coincident points
    dots = best_pts @ best_pts.T
    np.fill_diagonal(dots, 0)
    if np.max(np.abs(dots)) > 0.9999:
        best_pts = _to_sphere(best_pts + np.random.randn(N, 3) * 0.01)
        best_pts = _to_sphere(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
