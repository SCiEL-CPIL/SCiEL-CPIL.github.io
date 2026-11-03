"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _spherical_circumradius(v1, v2, v3):
    """
    Compute the spherical circumradius of the triangle formed by v1, v2, v3 on the unit sphere.
    Returns the angle in radians.
    """
    d1 = v1 - v2
    d2 = v2 - v3
    c = np.cross(d1, d2)
    norm_c = np.linalg.norm(c)
    if norm_c < 1e-10:
        return np.pi / 2
    c = c / norm_c
    
    dot = np.dot(c, v1)
    if dot < 0:
        c = -c
        dot = -dot
    
    dot = np.clip(dot, -1.0, 1.0)
    return np.arccos(dot)

def _covering_radius(pts):
    """Compute covering radius via convex hull facets using precise spherical circumradius."""
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
        max_circumradius = 0.0
        for simplex_idx in hull.simplices:
            if len(simplex_idx) == 3:
                simplex = pts[simplex_idx]
                cr = _spherical_circumradius(simplex[0], simplex[1], simplex[2])
                max_circumradius = max(max_circumradius, cr)
        return np.degrees(max_circumradius)
    except Exception:
        rng = np.random.RandomState(42)
        samples = rng.randn(5000, 3)
        samples /= np.linalg.norm(samples, axis=1, keepdims=True)
        dots = np.clip(np.dot(samples, pts.T), -1, 1)
        angles = np.arccos(dots)
        min_angles = np.min(angles, axis=1)
        return np.degrees(np.max(min_angles))

def _fibonacci_points(n, jitter=0.0):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    if jitter > 0:
        rng = np.random.RandomState(42)
        noise = rng.randn(n, 3) * jitter
        pts = pts + noise
    return _normalize(pts)

def _random_points(n, seed):
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    return _normalize(pts)

def _icosahedron_based_points(n, seed):
    rng = np.random.RandomState(seed)
    t = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ])
    ico = _normalize(ico)
    all_pts = np.vstack([ico, _random_points(n - len(ico), seed)])
    selected = [0]
    for _ in range(n - 1):
        sel_pts = all_pts[selected]
        dots = np.clip(np.dot(all_pts, sel_pts.T), -1, 1)
        angles = np.arccos(dots)
        min_dists = np.min(angles, axis=1)
        selected.append(np.argmax(min_dists))
    return _normalize(all_pts[selected])

def _local_refine(pts, time_limit):
    """Refine points using Nelder-Mead on covering radius."""
    pts = _normalize(pts).copy()
    best = pts.copy()
    best_cr = _covering_radius(best)
    start_time = time.time()
    
    def objective_flat(x):
        p = _normalize(x.reshape(-1, 3))
        return _covering_radius(p)
    
    for restart in range(5):
        if time.time() - start_time > time_limit - 5:
            break
        x0 = best.flatten()
        try:
            res = minimize(objective_flat, x0, method='Nelder-Mead', 
                          options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.fun < best_cr:
                best_cr = res.fun
                best = _normalize(res.x.reshape(-1, 3))
        except Exception:
            pass
        if time.time() - start_time > time_limit - 5:
            break
    
    return best, best_cr

def _gram_matrix_refine(pts, time_limit):
    """
    Refine using Gram matrix optimization:
    Maximize minimum off-diagonal entry (minimize max angular distance)
    via projected gradient ascent on PSD cone with unit diagonal.
    """
    pts = _normalize(pts).copy()
    best = pts.copy()
    best_cr = _covering_radius(best)
    start_time = time.time()
    
    # Compute initial Gram matrix
    G = pts @ pts.T
    np.fill_diagonal(G, 1.0)
    
    def _project_psd_unit_diag(G):
        """Project onto PSD cone with unit diagonal."""
        # Eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(G)
        # Clamp negative eigenvalues
        eigvals = np.maximum(eigvals, 0)
        G_new = (eigvecs * eigvals) @ eigvecs.T
        # Enforce unit diagonal
        np.fill_diagonal(G_new, 1.0)
        # Symmetrize
        G_new = (G_new + G_new.T) / 2
        return G_new
    
    def _reconstruct_from_gram(G):
        """Reconstruct points from Gram matrix via Cholesky or eigendecomposition."""
        # Add small regularization for numerical stability
        G_reg = G + 1e-8 * np.eye(N)
        try:
            L = np.linalg.cholesky(G_reg)
            pts_new = L.T  # Each row is a point
            return _normalize(pts_new)
        except np.linalg.LinAlgError:
            # Fallback to eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(G_reg)
            eigvals = np.maximum(eigvals, 0)
            pts_new = eigvecs * np.sqrt(eigvals)
            return _normalize(pts_new)
    
    # Gradient ascent: maximize min off-diagonal entry
    # Use smooth approximation: -logsumexp(-beta * off_diag)
    beta = 50.0
    lr = 0.01
    n_iter = 200
    
    for i in range(n_iter):
        if time.time() - start_time > time_limit - 5:
            break
        
        G = _project_psd_unit_diag(G)
        
        # Get off-diagonal entries
        mask = ~np.eye(N, dtype=bool)
        off_diag = G[mask]
        
        # Smooth max approximation: we want to maximize the minimum off-diagonal
        # This is equivalent to minimizing the covering radius (roughly)
        # Use -logsumexp(-beta * x) as smooth approximation of min(x)
        neg_beta_x = -beta * off_diag
        max_val = np.max(neg_beta_x)
        logsumexp = max_val + np.log(np.sum(np.exp(neg_beta_x - max_val)))
        obj = -logsumexp / beta
        
        # Gradient with respect to off-diagonal entries
        weights = np.exp(neg_beta_x - max_val)
        weights /= np.sum(weights)
        
        # Gradient of obj w.r.t. G (off-diagonal)
        grad_off = -weights
        
        # Reconstruct full gradient
        grad = np.zeros_like(G)
        grad[mask] = grad_off
        grad = grad + grad.T
        np.fill_diagonal(grad, 0)
        
        # Project gradient onto feasible set (zero diagonal already handled)
        G += lr * grad
        G = (G + G.T) / 2
        G = _project_psd_unit_diag(G)
        
        # Reconstruct points and check
        pts_new = _reconstruct_from_gram(G)
        cr = _covering_radius(pts_new)
        
        if cr < best_cr:
            best_cr = cr
            best = pts_new.copy()
            # Update G from new points
            G = best @ best.T
            np.fill_diagonal(G, 1.0)
    
    return best, best_cr

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s safety margin for 150s limit
    
    # Multiple diverse initializations
    initializations = []
    initializations.append(('fib', _fibonacci_points(N)))
    initializations.append(('fib_jitter', _fibonacci_points(N, jitter=0.05)))
    for seed in range(5):
        initializations.append((f'rand_{seed}', _random_points(N, seed)))
    for seed in range(3):
        initializations.append((f'ico_{seed}', _icosahedron_based_points(N, seed)))
    
    best_pts = None
    best_cr = np.inf
    
    # Evaluate each initialization
    for name, pts in initializations:
        if time.time() > deadline - 20:
            break
        cr = _covering_radius(pts)
        if cr < best_cr:
            best_cr = cr
            best_pts = pts.copy()
    
    # Refine the best initialization with Nelder-Mead
    if best_pts is not None and time.time() < deadline - 10:
        refined_pts, refined_cr = _local_refine(best_pts, deadline - time.time() - 5)
        if refined_cr < best_cr:
            best_cr = refined_cr
            best_pts = refined_pts
    
    # If we still have time, try Gram matrix refinement on the best point
    if best_pts is not None and time.time() < deadline - 15:
        gram_pts, gram_cr = _gram_matrix_refine(best_pts, deadline - time.time() - 5)
        if gram_cr < best_cr:
            best_cr = gram_cr
            best_pts = gram_pts
    
    # If we still have time, try refining other good initializations
    if time.time() < deadline - 15:
        for name, pts in initializations:
            if time.time() > deadline - 10:
                break
            cr = _covering_radius(pts)
            if cr < best_cr + 1.0:  # Only refine if close to best
                if time.time() < deadline - 10:
                    refined_pts, refined_cr = _local_refine(pts, deadline - time.time() - 5)
                    if refined_cr < best_cr:
                        best_cr = refined_cr
                        best_pts = refined_pts
    
    if best_pts is None:
        best_pts = _fibonacci_points(N)
    
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
