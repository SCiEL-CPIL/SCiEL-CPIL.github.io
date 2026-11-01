import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
DEADLINE_SECONDS = 145.0  # Safety margin before 150s kill
EPS = 1e-15


def _log_energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|xi-xj|) = -sum_{i<j} log(|xi-xj|)"""
    n = pts.shape[0]
    dots = pts @ pts.T
    d2 = 2.0 - 2.0 * dots
    np.fill_diagonal(d2, 1.0)
    d2 = np.maximum(d2, EPS)
    dist = np.sqrt(d2)
    iu = np.triu_indices(n, k=1)
    log_d = np.log(dist[iu])
    return -math.fsum(log_d)


def _energy_and_grad(pts):
    """
    pts: (N, 3) array of unit vectors.
    Returns (energy, grad) where energy = sum_{i<j} log(1/|xi-xj|) = -sum_{i<j} log(|xi-xj|)
    and grad is the gradient w.r.t. pts in R^{N x 3}.
    
    E = -sum_{i<j} log(d_ij)
    d_ij = |x_i - x_j|
    d_ij^2 = 2 - 2*x_i.x_j
    dE/dx_i = -sum_{j != i} d/dx_i log(d_ij) = -sum_{j != i} (x_i - x_j) / d_ij^2
    """
    n = pts.shape[0]
    dots = pts @ pts.T
    d2 = 2.0 - 2.0 * dots
    np.fill_diagonal(d2, 1.0)
    d2 = np.maximum(d2, EPS)
    
    dist = np.sqrt(d2)
    log_dist = np.log(dist)
    np.fill_diagonal(log_dist, 0.0)
    energy = -0.5 * np.sum(log_dist)
    
    inv_d2 = 1.0 / d2
    np.fill_diagonal(inv_d2, 0.0)
    
    # dE/dx_i = -sum_j inv_d2[i,j] * (x_i - x_j) = -x_i * sum_j inv_d2[i,j] + sum_j inv_d2[i,j]*x_j
    weight_sums = np.sum(inv_d2, axis=1, keepdims=True)
    weighted_pts = inv_d2 @ pts
    
    grad = -(weighted_pts - weight_sums * pts)
    
    return energy, grad


def _tangent_grad(pts, grad):
    """Project Euclidean gradient onto the tangent space."""
    radial = np.sum(grad * pts, axis=1, keepdims=True)
    return grad - radial * pts


def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, EPS)
    return pts / norms


def _retract(pts, step):
    """Move points along step and project back to sphere."""
    new_pts = pts + step
    return _project_to_sphere(new_pts)


def _fibonacci_spiral(n, offset=0.5):
    """Generate n points on sphere using Fibonacci spiral."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _project_to_sphere(pts)


def _golden_spiral_v2(n, offset=0.0):
    """Alternative golden spiral with offset for diversity."""
    i = np.arange(n) + 0.5 + offset
    z = 1.0 - 2.0 * i / n
    z = np.clip(z, -1.0, 1.0)
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _project_to_sphere(pts)


def _random_points(n, seed=42):
    """Generate n random points on sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    return _project_to_sphere(pts)


def _perturbed_points(base_pts, sigma, seed=0):
    """Perturb base points by adding Gaussian noise and reprojecting."""
    rng = np.random.RandomState(seed)
    noise = rng.randn(*base_pts.shape) * sigma
    pts = base_pts + noise
    return _project_to_sphere(pts)


def _icosahedron_subdivide(n, seed=42):
    """Generate points by subdividing an icosahedron."""
    rng = np.random.RandomState(seed)
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    verts = _project_to_sphere(verts)
    
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)
    ]
    
    pts = [v.copy() for v in verts]
    
    edges = set()
    for f in faces:
        for i in range(3):
            a, b = f[i], f[(i + 1) % 3]
            edge = (min(a, b), max(a, b))
            edges.add(edge)
    
    for a, b in edges:
        mid = (verts[a] + verts[b]) / 2.0
        pts.append(_project_to_sphere(mid.reshape(1, 3))[0])
    
    for f in faces:
        center = (verts[f[0]] + verts[f[1]] + verts[f[2]]) / 3.0
        pts.append(_project_to_sphere(center.reshape(1, 3))[0])
    
    pts = np.array(pts)
    if len(pts) > n:
        indices = rng.choice(len(pts), n, replace=False)
        pts = pts[indices]
    elif len(pts) < n:
        remaining = _random_points(n - len(pts), seed=seed + 1000)
        pts = np.vstack([pts, remaining])
    
    return _project_to_sphere(pts)


def _riemannian_descent(pts, max_iter=800, tol=1e-12, deadline=None):
    """
    Riemannian Gradient Descent with Backtracking Line Search and momentum.
    """
    pts = _project_to_sphere(pts.copy())
    energy, grad = _energy_and_grad(pts)
    g_t = _tangent_grad(pts, grad)
    
    best_pts = pts.copy()
    best_energy = energy
    
    lr = 0.02
    momentum = np.zeros_like(pts)
    beta = 0.9
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        gnorm = np.linalg.norm(g_t)
        if gnorm < tol:
            break
        
        # Momentum update
        momentum = beta * momentum + (1 - beta) * g_t
        
        # Backtracking line search
        alpha = lr
        improved = False
        for _ in range(30):
            new_pts = _retract(pts, -alpha * momentum)
            new_energy, _ = _energy_and_grad(new_pts)
            if new_energy < energy - 1e-4 * alpha * np.sum(momentum * momentum):
                pts = new_pts
                energy = new_energy
                improved = True
                break
            alpha *= 0.5
        
        if not improved:
            lr *= 0.5
            if lr < 1e-14:
                break
            continue
        
        _, grad = _energy_and_grad(pts)
        g_t = _tangent_grad(pts, grad)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
        
        lr = max(0.005, lr * 0.998)
    
    return best_pts, best_energy


def _polish_lbfgs(pts, max_iter=500, gtol=1e-14, deadline=None):
    """Polish using L-BFGS-B with tight tolerances."""
    def fun_and_grad(x):
        x_pts = x.reshape(N, 3)
        x_pts = _project_to_sphere(x_pts)
        e, g = _energy_and_grad(x_pts)
        g_t = _tangent_grad(x_pts, g)
        return e, g_t.ravel()
    
    x0 = pts.ravel()
    try:
        if deadline is not None and time.time() > deadline:
            return pts, _log_energy(pts)
        
        res = minimize(fun_and_grad, x0, jac=True, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'gtol': gtol, 'ftol': 1e-16})
        if res.success or np.isfinite(res.fun):
            x_opt = res.x.reshape(N, 3)
            x_opt = _project_to_sphere(x_opt)
            e_opt = _log_energy(x_opt)
            if np.isfinite(e_opt):
                return x_opt, e_opt
    except Exception:
        pass
    return pts, _log_energy(pts)


def _polish_bfgs(pts, max_iter=300, gtol=1e-14, deadline=None):
    """Polish using BFGS with tight tolerances."""
    def fun_and_grad(x):
        x_pts = x.reshape(N, 3)
        x_pts = _project_to_sphere(x_pts)
        e, g = _energy_and_grad(x_pts)
        g_t = _tangent_grad(x_pts, g)
        return e, g_t.ravel()
    
    x0 = pts.ravel()
    try:
        if deadline is not None and time.time() > deadline:
            return pts, _log_energy(pts)
        
        res = minimize(fun_and_grad, x0, jac=True, method='BFGS',
                       options={'maxiter': max_iter, 'gtol': gtol, 'ftol': 1e-16})
        if res.success or np.isfinite(res.fun):
            x_opt = res.x.reshape(N, 3)
            x_opt = _project_to_sphere(x_opt)
            e_opt = _log_energy(x_opt)
            if np.isfinite(e_opt):
                return x_opt, e_opt
    except Exception:
        pass
    return pts, _log_energy(pts)


def _hessian_log_energy(pts):
    """
    Compute the Riemannian Hessian of the logarithmic energy on the sphere.
    
    For E = -sum_{i<j} log(d_ij), the Hessian on the sphere involves:
    - The Euclidean Hessian projected to tangent space
    - Christoffel symbol corrections
    
    For simplicity and robustness, we compute the full Euclidean Hessian and project.
    The Hessian of -log|xi-xj| w.r.t. xi is:
    H_ij = (xi - xj)(xi - xj)^T / d^4 - I / d^2
    """
    n = pts.shape[0]
    # We'll compute the Hessian as a linear operator applied to a tangent vector
    # For efficiency, we compute it via finite differences of the gradient
    # But for a proper Newton step, we need the actual Hessian structure
    
    # Euclidean Hessian of E w.r.t. all points:
    # d2E/dxi/dxj for i != j: (xi-xj)(xi-xj)^T/d^4 - I/d^2 (but this is for -log(d))
    # Actually d(-log d)/dx_i = -(x_i - x_j)/d^2
    # d2(-log d)/dx_i^2 = -(I*d^2 - 2(x_i-x_j)(x_i-x_j)^T)/d^4 = -I/d^2 + 2(x_i-x_j)(x_i-x_j)^T/d^4
    # d2(-log d)/dx_i/dx_j = (x_i-x_j)(x_i-x_j)^T/d^4 - I/d^2 ... wait let me be more careful
    
    # E_pair = -log|xi - xj|
    # dE/dxi = -(xi - xj)/|xi-xj|^2
    # dE/dxj = (xi - xj)/|xi-xj|^2
    # d2E/dxi^2 = -I/d^2 + 2(xi-xj)(xi-xj)^T/d^4
    # d2E/dxi/dxj = I/d^2 - 2(xi-xj)(xi-xj)^T/d^4  (cross term)
    
    # For the Riemannian Hessian on S^2, we need to project:
    # H_riem = P_i * H_eucl * P_j + connection terms
    # where P_i = I - xi xi^T
    
    # For a practical Newton step, we'll use the projected Euclidean Hessian
    # as an approximation (valid near the minimum where gradients are small)
    
    return None  # Placeholder; we'll use a different approach below


def _newton_polish(pts, max_iter=50, deadline=None):
    """
    Newton polish using the Riemannian Hessian approximated via the
    projected Euclidean Hessian. For the logarithmic energy on S^2,
    this provides quadratic convergence near the minimum.
    """
    pts = _project_to_sphere(pts.copy())
    n = pts.shape[0]
    energy = _log_energy(pts)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        e, g = _energy_and_grad(pts)
        g_t = _tangent_grad(pts, g)
        gnorm = np.linalg.norm(g_t)
        if gnorm < 1e-14:
            break
        
        # Build the Riemannian Hessian (projected Euclidean Hessian)
        # H_eucl[i,j] for the pair (i,j):
        # For i != j: contribution to H_ii, H_jj, H_ij, H_ji from pair (i,j)
        # H_ii contribution: -I/d^2 + 2(xi-xj)(xi-xj)^T/d^4
        # H_jj contribution: -I/d^2 + 2(xi-xj)(xi-xj)^T/d^4  
        # H_ij contribution: I/d^2 - 2(xi-xj)(xi-xj)^T/d^4 (but this couples xi and xj)
        
        # Actually, the full Hessian is a (3n x 3n) matrix. For 100 points that's 300x300.
        # We can build it and solve the linear system.
        
        # Build Hessian matrix
        H = np.zeros((3*n, 3*n))
        dots = pts @ pts.T
        d2 = 2.0 - 2.0 * dots
        np.fill_diagonal(d2, 1.0)
        d2 = np.maximum(d2, EPS)
        dist = np.sqrt(d2)
        inv_d2 = 1.0 / d2
        inv_d4 = inv_d2 * inv_d2
        np.fill_diagonal(inv_d2, 0.0)
        np.fill_diagonal(inv_d4, 0.0)
        
        for i in range(n):
            for j in range(i+1, n):
                diff = pts[i] - pts[j]
                d2_ij = d2[i,j]
                inv_d2_ij = 1.0 / d2_ij
                inv_d4_ij = inv_d2_ij * inv_d2_ij
                
                # H_ii block: -I/d^2 + 2*diff*diff^T/d^4
                H_ii = -inv_d2_ij * np.eye(3) + 2.0 * inv_d4_ij * np.outer(diff, diff)
                # H_jj block: same as H_ii
                H_jj = H_ii.copy()
                # H_ij block: I/d^2 - 2*diff*diff^T/d^4  (negative of H_ii)
                H_ij = -H_ii
                H_ji = H_ij.copy()
                
                si = 3*i
                sj = 3*j
                H[si:si+3, si:si+3] += H_ii
                H[sj:sj+3, sj:sj+3] += H_jj
                H[si:si+3, sj:sj+3] += H_ij
                H[sj:sj+3, si:si+3] += H_ji
        
        # Project Hessian to tangent space: H_riem = P * H_eucl * P
        # where P is the block-diagonal projection matrix
        # For efficiency, we'll apply the projection to the gradient and solve
        # P * H * P * v = P * g_t  =>  H * v = P * g_t (since v is already tangent)
        # Actually: (P H P) v = P g_t, and since v is in tangent space, P v = v
        # So: P H v = P g_t => H v = P g_t + (I-P) * something
        # Simpler: solve H v = g_t (projected), then project v to tangent
        
        g_t_flat = g_t.ravel()
        
        # Add small regularization for numerical stability
        reg = 1e-10 * np.eye(3*n)
        H_reg = H + reg
        
        try:
            v = np.linalg.solve(H_reg, g_t_flat)
        except np.linalg.LinAlgError:
            v = np.linalg.lstsq(H_reg, g_t_flat, rcond=None)[0]
        
        v = v.reshape(n, 3)
        # Project to tangent space
        radial = np.sum(v * pts, axis=1, keepdims=True)
        v_t = v - radial * pts
        
        # Line search on Newton step
        alpha = 1.0
        for _ in range(30):
            new_pts = _retract(pts, -alpha * v_t)
            new_e = _log_energy(new_pts)
            if new_e < e:
                pts = new_pts
                energy = new_e
                break
            alpha *= 0.5
        else:
            break
    
    return pts, energy


def construct():
    t_start = time.time()
    deadline = t_start + DEADLINE_SECONDS
    
    # Generate diverse initializations - focus on high-quality starts
    inits = []
    
    # 1. Fibonacci spiral with different offsets (good structured starts)
    for offset in [0.3, 0.5, 0.7]:
        inits.append((f'fib_{offset}', _fibonacci_spiral(N, offset)))
    
    # 2. Golden spiral variants
    for offset in [0.0, 0.25, 0.5, 0.75]:
        inits.append((f'golden_{offset}', _golden_spiral_v2(N, offset)))
    
    # 3. Random with different seeds
    for seed in [1, 2, 3, 4, 5, 6]:
        inits.append((f'rand{seed}', _random_points(N, seed=seed)))
    
    # 4. Perturbed Fibonacci with different scales
    fib_base = _fibonacci_spiral(N)
    for sigma, seed in [(0.05, 10), (0.1, 11), (0.15, 12), (0.2, 13)]:
        inits.append((f'pert{sigma}_{seed}', _perturbed_points(fib_base, sigma, seed=seed)))
    
    # 5. Icosahedron subdivision
    inits.append(('ico', _icosahedron_subdivide(N, seed=42)))
    inits.append(('ico_pert', _perturbed_points(_icosahedron_subdivide(N, seed=42), 0.1, seed=50)))
    
    # 6. Perturbed random
    rand_base = _random_points(N, seed=100)
    for sigma, seed in [(0.1, 20), (0.2, 21)]:
        inits.append((f'randpert{sigma}_{seed}', _perturbed_points(rand_base, sigma, seed=seed)))
    
    best_pts = None
    best_energy = np.inf
    
    # Optimize each initialization
    for name, init_pts in inits:
        elapsed = time.time() - t_start
        if elapsed > DEADLINE_SECONDS - 30:
            break
        
        # Stage 1: Riemannian Gradient Descent with momentum
        rd_deadline = min(deadline - 20, t_start + elapsed + 12)
        pts_opt, energy_opt = _riemannian_descent(init_pts, max_iter=800, tol=1e-12,
                                                  deadline=rd_deadline)
        
        # Stage 2: L-BFGS-B Polish
        if time.time() < deadline - 15:
            pts_opt, energy_opt = _polish_lbfgs(pts_opt, max_iter=300, gtol=1e-14,
                                                deadline=deadline - 12)
        
        # Stage 3: BFGS Polish for additional refinement
        if time.time() < deadline - 10:
            pts_opt, energy_opt = _polish_bfgs(pts_opt, max_iter=200, gtol=1e-15,
                                               deadline=deadline - 8)
        
        if energy_opt < best_energy:
            best_energy = energy_opt
            best_pts = pts_opt.copy()
    
    # Final high-precision polish on the best result
    if best_pts is not None and time.time() < deadline - 8:
        best_pts, best_energy = _polish_lbfgs(best_pts, max_iter=1000, gtol=1e-16,
                                              deadline=deadline - 5)
        
        # Newton polish for quadratic convergence
        if time.time() < deadline - 3:
            best_pts, best_energy = _newton_polish(best_pts, max_iter=50,
                                                   deadline=deadline - 2)
    
    # Final Check and Return
    if best_pts is None:
        best_pts = _fibonacci_spiral(N)
    
    best_pts = _project_to_sphere(best_pts)
    return best_pts
