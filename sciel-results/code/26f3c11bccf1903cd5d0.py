import numpy as np
import time
from scipy.optimize import minimize

N = 300
SEED = 12345
T_START = time.time()
DEADLINE = 110.0

def _score(points):
    """Return the feasible common radius (0.0 if invalid)."""
    if points.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    norm = np.linalg.norm(points, axis=1)
    r_bound = 1.0 - norm
    if np.min(r_bound) < 0.0:
        return 0.0
    d = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    r_pair = 0.5 * np.min(d)
    r = min(r_bound.min(), r_pair)
    return float(r) if r > 0.0 else 0.0

def _ring_init():
    """Initialize points on concentric rings with hexagonal-like spacing."""
    pts = []
    # Start from center
    pts.append([0.0, 0.0])
    
    # Add rings outward
    ring_radius = 0.05
    while len(pts) < N and ring_radius < 0.98:
        # Number of points on this ring based on circumference
        n_on_ring = max(1, int(2 * np.pi * ring_radius / (ring_radius * 0.55)))
        angles = np.linspace(0, 2 * np.pi, n_on_ring, endpoint=False)
        for angle in angles:
            if len(pts) >= N:
                break
            x = ring_radius * np.cos(angle)
            y = ring_radius * np.sin(angle)
            pts.append([x, y])
        ring_radius += 0.04
    
    # If we don't have enough, add more on the outermost ring
    if len(pts) < N:
        outer_r = pts[-1][0]**2 + pts[-1][1]**2
        outer_r = np.sqrt(outer_r) if outer_r > 0 else 0.95
        angles = np.linspace(0, 2 * np.pi, N - len(pts), endpoint=False)
        for angle in angles:
            x = outer_r * np.cos(angle)
            y = outer_r * np.sin(angle)
            pts.append([x, y])
    
    return np.array(pts[:N])

def _hex_lattice_init():
    """Generate a hexagonal lattice and pick N points inside the disk."""
    spacing = 0.055
    while True:
        k = int(1.0 / spacing) + 3
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                rows.append((x, y))
        pts = np.array(rows)
        rad = np.hypot(pts[:, 0], pts[:, 1])
        keep = rad <= 1.0 - spacing / 2.0
        pts = pts[keep]
        if len(pts) >= N:
            order = np.argsort(np.hypot(pts[:, 0], pts[:, 1]), kind="stable")
            return pts[order[:N]].copy()
        spacing *= 0.97

def _compute_forces(points, temperature=1.0):
    """
    Compute repulsive forces between all pairs and attractive force toward center.
    Uses a smooth potential to avoid singularities.
    """
    N_pts = points.shape[0]
    forces = np.zeros_like(points)
    
    # Pairwise repulsive forces
    diff = points[:, None, :] - points[None, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    
    # Use a smooth kernel: F = (1/d^2) * direction, but capped
    # To avoid issues, use F = 1/(d + eps)
    eps = 0.01
    safe_dists = dists + eps
    
    # Force magnitude: proportional to 1/d^2 (Coulomb-like) but smooth
    force_mag = 1.0 / (safe_dists ** 2)
    # Cap the force to avoid huge values
    force_mag = np.minimum(force_mag, 100.0)
    
    # Direction: unit vector from j to i
    unit_dirs = diff / safe_dists[:, :, None]
    
    # Total force on each point from all others
    forces = np.sum(force_mag[:, :, None] * unit_dirs, axis=1)
    
    # Add boundary constraint: if point is too close to boundary, push inward
    norms = np.linalg.norm(points, axis=1)
    boundary_margin = 0.02
    for i in range(N_pts):
        if norms[i] > 1.0 - boundary_margin:
            # Push toward center
            forces[i] -= (points[i] / max(norms[i], 1e-10)) * (norms[i] - (1.0 - boundary_margin)) * 10.0
    
    return forces

def _gradient_descent_optimize(points, max_iters=5000, lr_init=0.001, lr_decay=0.999, t0=None, deadline=None):
    """
    Direct gradient descent with adaptive learning rate.
    """
    if t0 is None:
        t0 = time.time()
    if deadline is None:
        deadline = T_START + DEADLINE
    
    best_pts = points.copy()
    best_r = _score(best_pts)
    lr = lr_init
    
    for it in range(max_iters):
        # Check time
        if time.time() > deadline - 10.0:
            break
        
        # Compute forces
        forces = _compute_forces(points)
        
        # Update positions
        points = points + lr * forces
        
        # Keep points inside the disk
        norms = np.linalg.norm(points, axis=1)
        mask = norms > 0.99
        if np.any(mask):
            points[mask] = points[mask] / norms[mask, None] * 0.99
        
        # Decay learning rate
        lr *= lr_decay
        
        # Evaluate score periodically
        if it % 50 == 0 or it == max_iters - 1:
            r = _score(points)
            if r > best_r:
                best_r = r
                best_pts = points.copy()
        
        # Early stopping if no improvement
        if it > 100 and abs(r - best_r) < 1e-10:
            break
    
    return best_pts

def _soft_min_obj_grad(points, beta):
    """
    Compute the soft-min objective and its gradient.
    """
    N_pts = points.shape[0]
    
    # Pairwise distances
    diff = points[:, None, :] - points[None, :, :]
    dists_full = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists_full, np.inf)
    
    # Boundary terms
    norms = np.linalg.norm(points, axis=1)
    bound_terms = 1.0 - norms
    
    # Indices for i<j pairs
    i_idx, j_idx = np.triu_indices(N_pts, k=1)
    d_ij = dists_full[i_idx, j_idx]
    S_pairs = 0.5 * d_ij
    
    # Boundary terms
    S_bounds = 1.0 - norms
    
    # All S values
    S_all = np.concatenate([S_pairs, S_bounds])
    
    # Calculate weights
    max_S = np.max(S_all)
    u = -beta * S_all
    max_u = np.max(u)
    exp_u = np.exp(u - max_u)
    Z = np.sum(exp_u)
    weights = exp_u / Z
    
    w_pairs = weights[:len(S_pairs)]
    w_bounds = weights[len(S_pairs):]
    
    # Objective value
    obj = -1.0 / beta * (max_u + np.log(Z))
    
    # Gradient
    grad = np.zeros_like(points)
    
    # Pair contributions
    diff_ij = points[i_idx] - points[j_idx]
    d_safe = np.where(d_ij < 1e-9, 1e-9, d_ij)
    unit_vec = diff_ij / d_safe[:, None]
    
    contrib = w_pairs[:, None] * 0.5 * unit_vec
    
    np.add.at(grad, i_idx, contrib)
    np.add.at(grad, j_idx, -contrib)
    
    # Boundary contributions
    r_safe = np.where(norms < 1e-9, 1e-9, norms)
    bound_grad = -w_bounds[:, None] * points / r_safe[:, None]
    grad += bound_grad
    
    return obj, grad

def _optimize_lbfgs(points, beta, t_budget, maxiter=200):
    """Run L-BFGS-B optimization for a given beta."""
    start = time.time()
    
    def obj_grad(x):
        p = x.reshape(N, 2)
        val, g = _soft_min_obj_grad(p, beta)
        return val, g.flatten()
    
    bounds = [(-1.5, 1.5)] * (N * 2)
    
    res = minimize(
        obj_grad, 
        points.flatten(), 
        method='L-BFGS-B', 
        jac=True, 
        bounds=bounds,
        options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    
    return res.x.reshape(N, 2)

def construct():
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    # Try multiple initializations
    inits = [
        _hex_lattice_init(),
        _ring_init(),
    ]
    
    # Add jittered versions
    for _ in range(2):
        base = _hex_lattice_init()
        jitter = 0.02 * rng.standard_normal((N, 2))
        init = base + jitter
        rad = np.linalg.norm(init, axis=1)
        mask = rad > 0.99
        if np.any(mask):
            init[mask] = init[mask] / rad[mask, None] * 0.99
        inits.append(init)
    
    best = None
    best_r = 0.0
    
    for init in inits:
        if time.time() - t0 > DEADLINE - 30.0:
            break
        
        # First, use gradient descent
        gd_pts = _gradient_descent_optimize(
            init, 
            max_iters=3000, 
            lr_init=0.002, 
            lr_decay=0.9995,
            t0=t0,
            deadline=T_START + DEADLINE
        )
        r_gd = _score(gd_pts)
        if best is None or r_gd > best_r:
            best_r = r_gd
            best = gd_pts
        
        # Then, polish with L-BFGS using soft-min
        if time.time() - t0 > DEADLINE - 20.0:
            break
        
        # Use increasing beta
        for beta in [50.0, 200.0, 1000.0]:
            if time.time() - t0 > DEADLINE - 10.0:
                break
            
            try:
                lbfgs_pts = _optimize_lbfgs(best, beta, DEADLINE - (time.time() - t0), maxiter=100)
                r_lbfgs = _score(lbfgs_pts)
                if r_lbfgs > best_r:
                    best_r = r_lbfgs
                    best = lbfgs_pts
            except Exception:
                pass
        
        # Final gradient descent polish
        if time.time() - t0 > DEADLINE - 5.0:
            break
        
        gd_final = _gradient_descent_optimize(
            best,
            max_iters=2000,
            lr_init=0.0005,
            lr_decay=0.9999,
            t0=t0,
            deadline=T_START + DEADLINE
        )
        r_final = _score(gd_final)
        if r_final > best_r:
            best_r = r_final
            best = gd_final
    
    return best if best is not None else _hex_lattice_init()
