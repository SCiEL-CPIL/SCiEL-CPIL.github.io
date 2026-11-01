"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14
D = 3

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _pairwise_dists(pts):
    """Compute all pairwise distances (upper triangle)."""
    n = len(pts)
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    # Upper triangle indices
    iu = np.triu_indices(n, k=1)
    return dists[iu]

def _objective(pts, beta=50.0):
    """Smooth objective: (soft_max / soft_min)^2 using log-sum-exp."""
    d = _pairwise_dists(pts)
    # Avoid zero distances
    d = np.maximum(d, 1e-12)
    # soft max via log-sum-exp
    max_d = np.max(d)
    logsumexp_max = np.log(np.sum(np.exp(d / beta - max_d / beta))) + max_d / beta
    soft_max = beta * logsumexp_max
    # soft min via -log-sum-exp of negative
    min_d = np.min(d)
    logsumexp_min = np.log(np.sum(np.exp(-d / beta + min_d / beta))) - min_d / beta
    soft_min = -beta * logsumexp_min
    if soft_min < 1e-12:
        return 1e12
    ratio = soft_max / soft_min
    return ratio * ratio

def _objective_grad(pts, beta=50.0):
    """Gradient of the smooth objective using numerical differentiation fallback."""
    # Use finite differences for simplicity and robustness
    eps = 1e-7
    grad = np.zeros_like(pts)
    f0 = _objective(pts, beta)
    for i in range(pts.size):
        orig = pts.flat[i]
        pts.flat[i] = orig + eps
        fp = _objective(pts, beta)
        pts.flat[i] = orig
        grad.flat[i] = (fp - f0) / eps
    return grad

def _score(pts):
    """Exact score: (max/min)^2."""
    d = _pairwise_dists(pts)
    if np.min(d) < 1e-12:
        return 1e12
    return (np.max(d) / np.min(d)) ** 2

def _icosahedron_vertices():
    """Return 12 vertices of a regular icosahedron centered at origin."""
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ], dtype=float)
    # Normalize to unit sphere
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    return verts

def _random_spherical(n, rng):
    """Generate n random points on unit sphere."""
    z = rng.standard_normal((n, 3))
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    return z

def _initial_config_icosahedron_plus2(rng):
    """Icosahedron + 2 extra points at random positions on sphere."""
    ico = _icosahedron_vertices()
    # Place 2 extra points: try opposite poles or random
    extra = _random_spherical(2, rng)
    pts = np.vstack([ico, extra])
    return pts

def _initial_config_random(rng):
    """Random points on sphere."""
    return _random_spherical(N, rng)

def _initial_config_fibonacci(rng):
    """Fibonacci sphere for N points."""
    n = N
    golden_angle = np.pi * (3 - np.sqrt(5))
    pts = []
    for i in range(n):
        z = 1 - 2 * (i + 0.5) / n
        r = np.sqrt(1 - z**2)
        theta = golden_angle * i
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        pts.append([x, y, z])
    pts = np.array(pts)
    # Add small random perturbation to break symmetry
    pts += rng.standard_normal(pts.shape) * 0.01
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts /= norms
    return pts

def _optimize_from_start(pts0, deadline, beta_schedule=None):
    """Optimize from a starting configuration. Returns best pts found."""
    if beta_schedule is None:
        beta_schedule = [100.0, 50.0, 20.0, 10.0, 5.0, 2.0]
    
    best_pts = pts0.copy()
    best_score = _score(pts0)
    
    for beta in beta_schedule:
        if time.time() > deadline:
            break
        # Optimize with current beta
        res = minimize(
            lambda x: _objective(x.reshape(-1, D), beta),
            pts0.flatten(),
            jac=lambda x: _objective_grad(x.reshape(-1, D), beta).flatten(),
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 1000}
        )
        if res.success or res.fun < best_score:
            candidate = res.x.reshape(-1, D)
            cand_score = _score(candidate)
            if cand_score < best_score:
                best_pts = candidate
                best_score = cand_score
        pts0 = res.x.reshape(-1, D) if res.x is not None else pts0
    
    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_score = np.inf
    
    # Multiple initial configurations
    init_configs = []
    
    # Icosahedron + 2 random
    for _ in range(3):
        init_configs.append(_initial_config_icosahedron_plus2(rng))
    
    # Fibonacci sphere
    init_configs.append(_initial_config_fibonacci(rng))
    
    # Random spherical
    for _ in range(3):
        init_configs.append(_initial_config_random(rng))
    
    # Icosahedron with 2 points at specific symmetric positions
    ico = _icosahedron_vertices()
    # Add points at north and south poles (relative to icosahedron symmetry)
    extra_north = np.array([[0, 0, 1.0]])
    extra_south = np.array([[0, 0, -1.0]])
    init_configs.append(np.vstack([ico, extra_north, extra_south]))
    
    for i, pts0 in enumerate(init_configs):
        if time.time() > deadline:
            break
        try:
            pts, score = _optimize_from_start(pts0, deadline)
            if score < best_score:
                best_score = score
                best_pts = pts
        except Exception:
            continue
    
    # Final polish: use very low beta for precise optimization
    if best_pts is not None and time.time() < deadline - 10.0:
        try:
            res = minimize(
                lambda x: _objective(x.reshape(-1, D), 1.0),
                best_pts.flatten(),
                jac=lambda x: _objective_grad(x.reshape(-1, D), 1.0).flatten(),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-10, 'maxfun': 2000}
            )
            if res.x is not None:
                polished = res.x.reshape(-1, D)
                pol_score = _score(polished)
                if pol_score < best_score:
                    best_pts = polished
                    best_score = pol_score
        except Exception:
            pass
    
    if best_pts is None:
        # Fallback: return icosahedron + 2 random
        best_pts = _initial_config_icosahedron_plus2(rng)
    
    return best_pts
# EVOLVE-BLOCK-END
