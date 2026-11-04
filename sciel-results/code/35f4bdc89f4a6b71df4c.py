"""Sphere covering with 50 equal caps - optimized via smooth surrogate refinement."""
import numpy as np
import time

N = 50
_DEADLINE = 115.0  # seconds; safety margin before 120s kill
_START = None


def _normalize(X):
    """Normalize rows of X to unit length."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return X / norms


def _covering_radius_degrees(X):
    """Compute covering radius in degrees.
    
    The covering radius is the maximum over all points on the sphere of the
    angular distance to the nearest site. This equals the maximum circumradius
    of the Delaunay triangulation (convex hull facets) of the sites.
    
    We compute this by finding, for each site, the maximum angular distance
    to that site among all points that are NOT closer to any other site.
    A practical approximation: the farthest Voronoi vertex.
    
    Exact method: the covering radius is max over all y on sphere of
    min_i angle(y, x_i). The worst point y is at a vertex of the Voronoi
    diagram on the sphere (equivalently, a circumcenter of a Delaunay triangle
    or a pole if the configuration doesn't cover).
    
    We use a dense grid + local refinement for robustness.
    """
    X = _normalize(X)
    # Dense grid sampling
    n_theta = 120
    n_phi = 240
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    Y = np.column_stack([
        np.sin(TH) * np.cos(PH).ravel(),
        np.sin(TH) * np.sin(PH).ravel(),
        np.cos(TH).ravel()
    ])
    # For each grid point, find nearest site
    # dot products: (n_grid, N)
    dots = Y @ X.T  # (n_grid, N)
    dots = np.clip(dots, -1.0, 1.0)
    min_dots = np.max(dots, axis=1)  # max dot = min angle
    max_angle_rad = np.arccos(min_dots)
    return np.degrees(max_angle_rad.max())


def _smooth_covering_surrogate(X, temp=0.5):
    """Smooth surrogate: soft-max over grid points of min-angle.
    
    Uses log-sum-exp for the outer max and a soft-min for the inner min.
    Returns (value, gradient) in radians.
    """
    X = _normalize(X)
    n_theta = 80
    n_phi = 160
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    Y = np.column_stack([
        np.sin(TH) * np.cos(PH).ravel(),
        np.sin(TH) * np.sin(PH).ravel(),
        np.cos(TH).ravel()
    ])
    # dots: (n_grid, N)
    dots = Y @ X.T
    dots = np.clip(dots, -1.0 + 1e-12, 1.0 - 1e-12)
    # soft-min of angles = -soft-max of dots (since angle = arccos(dot))
    # Actually: min_i angle(y, x_i) = arccos(max_i dot(y, x_i))
    # So we need soft version of: max over y of arccos(softmax-like)
    # Simpler: compute angles, then soft-max over grid
    angles = np.arccos(dots)  # (n_grid, N)
    # soft-min over sites: -logsumexp(-angles/temp)/... 
    # soft_min_i(a_i) = -logsumexp(-a_i/temp) * temp / N... 
    # Standard soft min: min_i a_i ≈ -temp * log(sum(exp(-a_i/temp))) + temp*log(N)
    neg_angles = -angles / temp
    neg_angles -= neg_angles.max(axis=1, keepdims=True)  # for stability
    logsum = np.log(np.sum(np.exp(neg_angles), axis=1)) + neg_angles.max(axis=1)
    soft_min_angles = -(logsum * temp)  # (n_grid,)
    # Now soft-max over grid points
    sm = soft_min_angles / temp
    sm -= sm.max()
    logsum_grid = np.log(np.sum(np.exp(sm))) + sm.max()
    value = logsum_grid * temp  # smooth max
    
    # Gradient: d/dX value
    # This is complex; use finite differences or analytical chain rule
    # For efficiency, we'll use a simpler approach: weighted sum of gradients
    # weights for each grid point (softmax)
    w_grid = np.exp(sm - logsum_grid)  # (n_grid,)
    # For each grid point, soft-min gradient over sites
    # d(soft_min)/d(angle_j) = exp(-angle_j/temp) / sum(exp(-angle_k/temp))
    probs = np.exp(-angles / temp)  # (n_grid, N)
    probs /= probs.sum(axis=1, keepdims=True)
    # d(angle)/d(dot) = -1/sqrt(1-dot^2)
    # d(dot)/dX_i = Y_j (for site i)
    # Chain: d(value)/dX_i = sum_j w_grid[j] * probs[j,i] * (-1/sqrt(1-dot^2)) * Y_j
    # But dot depends on X_i, and we need to account for normalization
    # Simplified: ignore normalization gradient (X is already normalized)
    grad = np.zeros_like(X)
    for j in range(len(w_grid)):
        if w_grid[j] < 1e-10:
            continue
        for i in range(N):
            d = dots[j, i]
            denom = np.sqrt(max(1.0 - d * d, 1e-12))
            grad[i] += w_grid[j] * probs[j, i] * (-1.0 / denom) * Y[j]
    return value, grad


def _refine_step(X, lr=0.01, temp=0.5, max_iter=500, deadline=None):
    """Gradient descent on smooth surrogate, with projection to sphere."""
    X = _normalize(X.copy())
    best_X = X.copy()
    best_val = _covering_radius_degrees(X)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        val, grad = _smooth_covering_surrogate(X, temp=temp)
        # Project gradient to tangent space: grad - (grad·x_i)x_i
        grad_tangent = grad - (grad * X).sum(axis=1, keepdims=True) * X
        X_new = X - lr * grad_tangent
        X_new = _normalize(X_new)
        # Check improvement
        new_val = _covering_radius_degrees(X_new)
        if new_val < best_val:
            best_val = new_val
            best_X = X_new.copy()
        X = X_new
        # Anneal temperature
        temp = max(0.1, temp * 0.995)
    
    return best_X, best_val


def _perturb(X, scale=0.1):
    """Add small random perturbation and renormalize."""
    noise = np.random.randn(*X.shape) * scale
    X_new = X + noise
    return _normalize(X_new)


def _local_search(X, deadline):
    """Iterative local search with random restarts."""
    best_X = X.copy()
    best_val = _covering_radius_degrees(X)
    
    n_restarts = 0
    while time.time() < deadline:
        n_restarts += 1
        # Perturb current best
        X_pert = _perturb(best_X, scale=0.05 + 0.05 * np.random.rand())
        X_opt, val_opt = _refine_step(X_pert, lr=0.005, temp=0.3, 
                                       max_iter=200, deadline=deadline)
        if val_opt < best_val:
            best_val = val_opt
            best_X = X_opt.copy()
    
    return best_X, best_val


def construct():
    global _START
    if _START is None:
        _START = time.time()
    
    deadline = _START + _DEADLINE
    
    # Start from a good initial configuration
    # Use Fibonacci spiral as base, then optimize
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
    X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    X = _normalize(X)
    
    # First pass: smooth gradient refinement
    X, val = _refine_step(X, lr=0.01, temp=1.0, max_iter=300, deadline=deadline)
    
    # Second pass: local search with restarts
    X, val = _local_search(X, deadline)
    
    # Final polish: very fine refinement
    if time.time() < deadline - 5:
        X, val = _refine_step(X, lr=0.001, temp=0.1, max_iter=200, deadline=deadline)
    
    return _normalize(X)
