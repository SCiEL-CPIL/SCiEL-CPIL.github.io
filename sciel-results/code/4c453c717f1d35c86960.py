"""Rewritten program: Halton init + Riemannian L-BFGS-B with separation penalty, basin hopping, and final polish."""
import numpy as np
import math
import time

N = 150

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _log_energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/d_ij) using math.fsum."""
    n = len(pts)
    terms = []
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-15:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)

def _log_energy_fast(pts):
    """Vectorized energy for speed (used in optimization, not final scoring)."""
    n = len(pts)
    # Compute all pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    # Mask diagonal and lower triangle
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d = dist[mask]
    d = np.maximum(d, 1e-15)
    return np.sum(np.log(1.0 / d))

def _log_energy_grad(pts):
    """Compute gradient of log energy w.r.t. points on sphere (Euclidean gradient projected)."""
    n = len(pts)
    grad = np.zeros_like(pts)
    for i in range(n):
        for j in range(i+1, n):
            d_vec = pts[i] - pts[j]
            d = np.linalg.norm(d_vec)
            if d < 1e-15:
                continue
            # d/dx_i log(1/d) = -1/d * d_vec/d = -d_vec/d^2
            g = -d_vec / (d * d)
            grad[i] += g
            grad[j] -= g
    # Project gradient to tangent space
    for i in range(n):
        grad[i] -= np.dot(grad[i], pts[i]) * pts[i]
    return grad

def _log_energy_grad_fast(pts):
    """Vectorized gradient computation."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    dist = np.maximum(dist, 1e-15)
    # Gradient: for each pair (i,j), contribution to grad[i] is -(pts[i]-pts[j])/d^2
    # This is sum_j (pts[j]-pts[i])/d_ij^2 for i != j
    inv_d2 = np.zeros_like(dist)
    mask = ~np.eye(n, dtype=bool)
    inv_d2[mask] = 1.0 / (dist[mask] ** 2)
    np.fill_diagonal(inv_d2, 0.0)
    # grad[i] = sum_j (pts[j] - pts[i]) * inv_d2[i,j]
    # = sum_j pts[j]*inv_d2[i,j] - pts[i]*sum_j inv_d2[i,j]
    row_sums = inv_d2.sum(axis=1, keepdims=True)
    grad = pts @ inv_d2.T - pts * row_sums
    # Project to tangent space
    for i in range(n):
        grad[i] -= np.dot(grad[i], pts[i]) * pts[i]
    return grad

def _separation_penalty(pts, min_sep=0.05, penalty_coeff=100.0):
    """Penalty for pairs closer than min_sep."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d = dist[mask]
    # Soft penalty: sum of max(0, min_sep - d)^2
    violations = np.maximum(0, min_sep - d)
    return penalty_coeff * np.sum(violations ** 2)

def _separation_penalty_grad(pts, min_sep=0.05, penalty_coeff=100.0):
    """Gradient of separation penalty."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    dist = np.maximum(dist, 1e-15)
    mask = ~np.eye(n, dtype=bool)
    # Penalty: penalty_coeff * sum_{i<j} max(0, min_sep - d_ij)^2
    # Gradient contribution for pair (i,j) when d < min_sep:
    # d/dx_i [penalty_coeff * (min_sep - d)^2] = penalty_coeff * 2*(min_sep-d) * (x_i - x_j)/d
    inv_d = np.zeros_like(dist)
    inv_d[mask] = 1.0 / dist[mask]
    np.fill_diagonal(inv_d, 0.0)
    
    violations = np.maximum(0, min_sep - dist)
    # Only non-zero where violation > 0
    coeff = penalty_coeff * 2 * violations * inv_d
    np.fill_diagonal(coeff, 0.0)
    
    grad = np.zeros_like(pts)
    for i in range(n):
        for j in range(n):
            if coeff[i, j] != 0:
                grad[i] += coeff[i, j] * (pts[i] - pts[j])
                grad[j] -= coeff[i, j] * (pts[i] - pts[j])
    # Project to tangent
    for i in range(n):
        grad[i] -= np.dot(grad[i], pts[i]) * pts[i]
    return grad

def _riemannian_lbfgs(pts, max_iter=2000, tol=1e-10, use_penalty=True, min_sep=0.05, penalty_coeff=100.0):
    """
    Riemannian L-BFGS-B style optimization on the sphere.
    Uses Euclidean L-BFGS with retraction (normalization) after each step.
    """
    n = len(pts)
    pts = _normalize(pts.copy())
    
    m = 10  # L-BFGS memory
    W = []  # list of (s, y) pairs
    
    def energy_and_grad(p):
        e = _log_energy_fast(p)
        g = _log_energy_grad_fast(p)
        if use_penalty:
            e += _separation_penalty(p, min_sep, penalty_coeff)
            g += _separation_penalty_grad(p, min_sep, penalty_coeff)
        return e, g
    
    e, g = energy_and_grad(pts)
    
    for it in range(max_iter):
        # Check convergence
        g_norm = np.linalg.norm(g)
        if g_norm < tol:
            break
        
        # L-BFGS two-loop recursion to compute search direction
        r = -g.copy()
        qs = []
        alphas = []
        for s, y in reversed(W):
            qs.append(np.dot(y, r))
            alphas.append(np.dot(s, r) / np.dot(s, y))
        gamma = np.dot(W[-1][0], W[-1][1]) / np.dot(W[-1][1], W[-1][1]) if W else 1.0
        z = r + gamma * g
        for (s, y), q, a in zip(W, qs, alphas):
            r = s * q + y * a  # This isn't quite right, let me redo
        # Actually, let me implement L-BFGS properly
        # Two-loop recursion:
        r = -g.copy()
        qs_vals = []
        alphas_vals = []
        for s, y in reversed(W):
            qs_vals.append(np.dot(y, r))
            alphas_vals.append(np.dot(s, r) / np.dot(s, y))
        gamma = 1.0
        if W:
            s_last, y_last = W[-1]
            denom = np.dot(y_last, y_last)
            if denom > 1e-15:
                gamma = np.dot(s_last, y_last) / denom
        z = r + gamma * g
        for (s, y), q, a in zip(reversed(W), reversed(qs_vals), reversed(alphas_vals)):
            z = z + s * a - y * q
        # Wait, the two-loop recursion is:
        # Forward: alpha_i = <s_i, r> / <y_i, r>
        # Backward: z = r + alpha_i*(s_i - gamma*y_i) ... 
        # Let me just use a simpler approach: use scipy's minimize with a wrapper
        
        # For simplicity and reliability, let's use a different approach:
        # Use scipy.optimize.minimize with L-BFGS-B on the 3N dimensional space
        # with normalization as a retraction
    
    # Actually, let me rewrite this more carefully using scipy
    from scipy.optimize import minimize
    
    def objective(x_flat):
        p = _normalize(x_flat.reshape(n, 3))
        e = _log_energy_fast(p)
        if use_penalty:
            e += _separation_penalty(p, min_sep, penalty_coeff)
        return e
    
    def gradient(x_flat):
        p = _normalize(x_flat.reshape(n, 3))
        g = _log_energy_grad_fast(p)
        if use_penalty:
            g += _separation_penalty_grad(p, min_sep, penalty_coeff)
        return g.flatten()
    
    x0 = pts.flatten()
    res = minimize(objective, x0, jac=gradient, method='L-BFGS-B', 
                   options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': max_iter * 2})
    best_pts = _normalize(res.x.reshape(n, 3))
    return best_pts, res.fun

def _halton_init(n_points, n_dims=3):
    """Generate Halton sequence points on the unit sphere."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    points = []
    for i in range(1, n_points + 1):
        u = [halton(i, 2), halton(i, 3), halton(i, 5)]
        # Map to sphere using inverse CDF for uniform distribution
        # z = 2*u[0] - 1
        # theta = 2*pi*u[1]
        # r = sqrt(1-z^2)
        z = 2.0 * u[0] - 1.0
        theta = 2.0 * math.pi * u[1]
        r = math.sqrt(max(0, 1.0 - z * z))
        points.append([r * math.cos(theta), r * math.sin(theta), z])
    return np.array(points)

def _fibonacci_init(n_points):
    """Fibonacci spiral initialization."""
    i = np.arange(n_points) + 0.5
    z = 1.0 - 2.0 * i / n_points
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    # Initialize with Halton sequence
    pts = _halton_init(N)
    pts = _normalize(pts)
    
    best_pts = pts.copy()
    best_energy = _log_energy_fast(pts)
    
    # Phase 1: Initial optimization with penalty
    if time.time() < deadline - 30.0:
        pts_opt, e = _riemannian_lbfgs(pts, max_iter=1000, use_penalty=True, min_sep=0.08, penalty_coeff=50.0)
        e_true = _log_energy_fast(pts_opt)
        if e_true < best_energy:
            best_energy = e_true
            best_pts = pts_opt.copy()
    
    # Phase 2: Basin hopping with random perturbations
    rng = np.random.default_rng(42)
    n_restarts = 0
    max_restarts = 50
    
    while time.time() < deadline - 20.0 and n_restarts < max_restarts:
        n_restarts += 1
        # Perturb the current best point
        scale = 0.1 * (1.0 - n_restarts / max_restarts) + 0.02
        noise = rng.normal(0, scale, size=(N, 3))
        perturbed = _normalize(best_pts + noise)
        
        # Optimize the perturbed configuration
        if time.time() < deadline - 10.0:
            pts_opt, e = _riemannian_lbfgs(perturbed, max_iter=500, use_penalty=True, min_sep=0.05, penalty_coeff=20.0)
            e_true = _log_energy_fast(pts_opt)
            if e_true < best_energy:
                best_energy = e_true
                best_pts = pts_opt.copy()
    
    # Phase 3: Final high-precision polish without penalty
    if time.time() < deadline - 5.0:
        pts_polish, e = _riemannian_lbfgs(best_pts, max_iter=3000, use_penalty=False)
        e_true = _log_energy_fast(pts_polish)
        if e_true < best_energy:
            best_energy = e_true
            best_pts = pts_polish.copy()
    
    # Final validation
    best_pts = _normalize(best_pts)
    
    # Verify no points are too close
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                # Nudge apart
                best_pts[j] += 1e-8 * rng.normal(size=3)
                best_pts = _normalize(best_pts)
    
    return best_pts
