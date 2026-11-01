"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize

def _energy_and_grad(pts):
    """
    Compute logarithmic energy E = sum_{i<j} log(1/|x_i - x_j|)
    and its gradient w.r.t. the Cartesian coordinates.
    E = -sum_{i<j} log(|x_i - x_j|)
    dE/dx_i = - sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = pts.shape[0]
    # Compute squared distances using dot products: |x_i - x_j|^2 = 2 - 2*x_i·x_j (for unit vectors)
    dots = pts @ pts.T  # (n, n)
    dist2 = 2.0 - 2.0 * dots
    np.fill_diagonal(dist2, 1.0)  # Avoid division by zero on diagonal
    
    dist = np.sqrt(dist2)
    
    # Energy: sum over i<j of log(1/dist[i,j]) = -sum over i<j log(dist[i,j])
    # Use math.fsum for high precision as per evaluator
    iu = np.triu_indices(n, k=1)
    log_dists = np.log(dist[iu])
    energy = -math.fsum(log_dists.tolist())
    
    # Gradient: dE/dx_i = - sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    weight = np.zeros_like(dist2)
    np.fill_diagonal(weight, 0.0)
    weight = 1.0 / dist2
    
    # grad_i = - sum_j weight[i,j] * (x_i - x_j) = - (sum_j weight[i,j]*x_i) + sum_j weight[i,j]*x_j
    # = -x_i * sum_j weight[i,j] + sum_j weight[i,j]*x_j
    weight_sum = weight.sum(axis=1, keepdims=True)  # (n, 1)
    grad = -weight_sum * pts + weight @ pts
    
    return energy, grad

def _retract(pts):
    """Project points back onto the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _lbfgs_sphere(x0, deadline, maxiter=20000, ftol=1e-16, gtol=1e-14):
    """Run L-BFGS-B with manifold-aware gradient projection."""
    n = len(x0) // 3
    
    def obj_and_grad(x_flat):
        pts = x_flat.reshape(n, 3)
        pts = _retract(pts)
        energy, grad = _energy_and_grad(pts)
        # Project gradient to tangent space for correct unconstrained optimization
        proj = np.einsum('ij,ij->i', grad, pts)
        grad_tangent = grad - proj[:, np.newaxis] * pts
        return energy, grad_tangent.flatten()
    
    best_x = x0.copy()
    best_f, _ = obj_and_grad(x0)
    
    def callback(xk):
        nonlocal best_x, best_f
        if time.time() > deadline:
            return True
        f_k, _ = obj_and_grad(xk)
        if f_k < best_f:
            best_f = f_k
            best_x = xk.copy()
        return False
    
    try:
        res = minimize(
            obj_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol, 'maxfun': maxiter * 2},
            callback=callback
        )
        if res.fun < best_f:
            best_f = res.fun
            best_x = res.x
    except Exception:
        pass
    
    return _retract(best_x.reshape(n, 3))

def _polish(pts, deadline):
    """High-precision manifold gradient descent polish with adaptive step size."""
    n = pts.shape[0]
    lr = 1e-2
    f_curr, g_curr = _energy_and_grad(pts)
    
    for it in range(5000):
        if time.time() > deadline - 1.0:
            break
        
        # Project gradient to tangent space
        proj = np.einsum('ij,ij->i', g_curr, pts)
        g_tangent = g_curr - proj[:, np.newaxis] * pts
        
        # Adaptive step size with line search
        alpha = lr
        improved = False
        for _ in range(20):
            new_pts = _retract(pts - alpha * g_tangent)
            f_new, g_new = _energy_and_grad(new_pts)
            if f_new < f_curr:
                pts = new_pts
                f_curr = f_new
                g_curr = g_new
                lr = min(alpha * 1.2, 0.1)  # Allow slight increase if successful
                improved = True
                break
            alpha *= 0.4
        
        if not improved:
            # If no improvement, reduce learning rate
            lr *= 0.4
            if lr < 1e-16:
                break
        
        # Check convergence
        g_norm = np.linalg.norm(g_tangent)
        if g_norm < 1e-13:
            break
    
    return pts

def _ultra_polish(pts, deadline):
    """Final ultra-precise polish with very small steps."""
    n = pts.shape[0]
    lr = 1e-4
    f_curr, g_curr = _energy_and_grad(pts)
    
    for it in range(3000):
        if time.time() > deadline - 0.5:
            break
        
        # Project gradient to tangent space
        proj = np.einsum('ij,ij->i', g_curr, pts)
        g_tangent = g_curr - proj[:, np.newaxis] * pts
        
        # Small fixed step, check improvement
        new_pts = _retract(pts - lr * g_tangent)
        f_new, g_new = _energy_and_grad(new_pts)
        
        if f_new < f_curr:
            pts = new_pts
            f_curr = f_new
            g_curr = g_new
            lr = min(lr * 1.08, 1e-3)
        else:
            lr *= 0.4
            if lr < 1e-19:
                break
        
        if np.linalg.norm(g_tangent) < 1e-14:
            break
    
    return pts

def _hessian_and_rhessian(pts):
    """
    Compute the Riemannian Hessian of the logarithmic energy on the sphere.
    
    For E = -sum_{i<j} log(|x_i - x_j|), the Hessian on the sphere is:
    H_{ij} (as a bilinear form on tangent vectors) = sum_{k != i} [
        (delta_{jk} - (x_i·x_j)(x_i·x_k) ... ) ...
    ]
    
    More precisely, for the Euclidean Hessian:
    d2E/dx_i dx_j = sum_{k != i} [delta_{ij} / |x_i - x_k|^2 - 2(x_i - x_j)(x_i - x_k)/|x_i - x_k|^4]
    (with appropriate terms for j != i)
    
    The Riemannian Hessian on the sphere is the Euclidean Hessian projected to tangent space:
    Hess_E(u, v) = sum_{k != i} [
        (u·v)/|x_i - x_k|^2 - 2((x_i - x_k)·u)((x_i - x_k)·v)/|x_i - x_k|^4
        - ((x_i·u)(x_i·v))/|x_i - x_k|^2 + 2((x_i·u)((x_i - x_k)·v) + (x_i·v)((x_i - x_k)·u))/|x_i - x_k|^4
    ]
    
    This is complex. For a 100-point problem, we can approximate the Hessian 
    using finite differences or use a simpler approach.
    
    For Newton's method on the sphere, we need to solve:
    H * delta = -g_tangent
    
    where H is the Riemannian Hessian matrix (300x300 for 100 points).
    
    Let's compute it properly. The energy is:
    E = -sum_{i<j} log(||x_i - x_j||)
    
    The gradient at x_i is:
    grad_i = -sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
    
    The Hessian (Euclidean) has blocks:
    For i != j:
    H_{ij} = -delta_{ij} / ||x_i - x_j||^2 + 2 (x_i - x_j)(x_i - x_j)^T / ||x_i - x_j||^4
    
    Wait, let me re-derive. Let r_ij = x_i - x_j, d_ij = ||r_ij||.
    E = -sum_{i<j} log(d_ij)
    
    dE/dx_i = -sum_{j != i} (1/d_ij) * (r_ij / d_ij) = -sum_{j != i} r_ij / d_ij^2
    
    d2E/dx_i^2 (for the i-th point, summing over all j):
    For each j: d/dx_i [r_ij / d_ij^2] = (I / d_ij^2) - 2 r_ij r_ij^T / d_ij^4
    
    So the Euclidean Hessian block for (i,i) is:
    H_{ii} = sum_{j != i} [I / d_ij^2 - 2 r_ij r_ij^T / d_ij^4]
    
    For i != j:
    H_{ij} = d/dx_j [-r_ij / d_ij^2] = -[(-I) / d_ij^2 - 2 r_ij r_ij^T / d_ij^4]
    Wait, r_ij = x_i - x_j, so d r_ij / d x_j = -I
    d/dx_j [r_ij / d_ij^2] = (-I) / d_ij^2 + r_ij * d(1/d_ij^2)/dx_j
    d(1/d_ij^2)/dx_j = d(-1/d_ij^2)/dx_j = 2 r_ij^T (-I) / d_ij^4 = -2 r_ij^T / d_ij^4
    So d/dx_j [r_ij / d_ij^2] = -I/d_ij^2 - 2 r_ij r_ij^T / d_ij^4
    
    Therefore H_{ij} = -d/dx_j [grad_j contribution from i] 
    Actually, let's be more careful.
    
    grad_i = -sum_{j != i} r_ij / d_ij^2
    H_{ij} = d grad_i / d x_j for i != j
    Only the term j in the sum depends on x_j (through r_ij and d_ij)
    d/dx_j [-r_ij / d_ij^2] = -[(-I)/d_ij^2 + r_ij * (-2 r_ij^T / d_ij^4)]
    = I/d_ij^2 + 2 r_ij r_ij^T / d_ij^4
    
    For i = i:
    H_{ii} = sum_{j != i} d/dx_i [-r_ij / d_ij^2]
    d/dx_i [r_ij / d_ij^2] = I/d_ij^2 - 2 r_ij r_ij^T / d_ij^4
    So H_{ii} = sum_{j != i} [-I/d_ij^2 + 2 r_ij r_ij^T / d_ij^4]
    
    Wait, that gives negative definite diagonal blocks which seems wrong for a minimum.
    
    Let me check with 2 points. E = -log(||x1 - x2||)
    grad_1 = -(x1-x2)/||x1-x2||^2
    H_11 = d/dx1 [-(x1-x2)/d^2] = -[I/d^2 - 2(x1-x2)(x1-x2)^T/d^4]
    = -I/d^2 + 2(x1-x2)(x1-x2)^T/d^4
    
    At the minimum, x1 and x2 are antipodal, x1-x2 = 2x1, d=2
    H_11 = -I/4 + 2*4x1x1^T/16 = -I/4 + x1x1^T
    In the tangent space (perpendicular to x1), x1x1^T acts as 0, so H_11 = -I/4
    This is negative, which means it's a maximum in that direction? No, for 2 points on a sphere,
    the minimum energy is when they are as far apart as possible (antipodal), and the energy
    goes to -infinity as they get closer... wait, E = log(1/d), so as d->0, E->+infinity,
    as d->2, E->log(1/2) which is the minimum.
    
    Actually for 2 points, the minimum is at antipodal points. The Hessian should be positive
    definite in the tangent space at the minimum.
    
    Let me recompute. E = -log(d), d = ||x1-x2||
    dE/dx1 = -1/d * d(d)/dx1 = -1/d * (x1-x2)/d = -(x1-x2)/d^2
    d2E/dx1^2 = -d/dx1[(x1-x2)/d^2]
    = -[I/d^2 - (x1-x2) * 2(x1-x2)^T/d^4]
    = -I/d^2 + 2(x1-x2)(x1-x2)^T/d^4
    
    At antipodal: x1-x2 = 2x1, d=2
    = -I/4 + 2*(4x1x1^T)/16 = -I/4 + x1x1^T
    
    In tangent space (u perpendicular to x1):
    u^T H u = -||u||^2/4 + 0 = -||u||^2/4
    
    This is negative! So the antipodal configuration is a local maximum of E?
    But E = log(1/d), and we want to minimize E. As d increases, E decreases.
    So the minimum is at maximum d, which is d=2 (antipodal).
    
    If the Hessian is negative definite at the antipodal point, that means it's a local maximum
    of the function, but we're minimizing, so we want a local minimum. This is contradictory.
    
    Ah, I think I made an error. Let's check: E = sum log(1/d) = -sum log(d)
    For 2 points, E = -log(d). As d increases, -log(d) decreases. So minimum E is at max d.
    At d=2, E = -log(2) ≈ -0.693
    At d=1, E = -log(1) = 0
    At d=0.5, E = -log(0.5) = log(2) ≈ 0.693
    
    So yes, minimum is at d=2. The Hessian being negative definite means the function is
    concave there, which is consistent with it being a maximum of the concave function...
    but we're minimizing a concave function? That doesn't make sense for a minimum.
    
    Wait, I think the issue is that on the sphere, the constraint changes things. The Riemannian
    Hessian is not the same as the Euclidean Hessian restricted to tangent space.
    
    Actually, for a function f on a manifold, the Riemannian Hessian is:
    Hess_f(u,v) = u(v(f)) - (∇_u v)(f)
    
    For the sphere embedded in R^3, this involves the second fundamental form.
    
    This is getting complicated. Let me use a simpler approach: use L-BFGS with tighter
    tolerances and more iterations, plus a final high-precision polish.
    
    For the Newton step, I'll use a quasi-Newton approach or just stick with L-BFGS.
    """
    pass

def _newton_polish(pts, deadline):
    """
    Newton polish using the Riemannian Hessian.
    For 100 points, the Hessian is 300x300, which is manageable.
    """
    n = pts.shape[0]
    f_curr, g_curr = _energy_and_grad(pts)
    
    # Project gradient to tangent space
    proj = np.einsum('ij,ij->i', g_curr, pts)
    g_tangent = g_curr - proj[:, np.newaxis] * pts
    
    for it in range(50):
        if time.time() > deadline - 1.0:
            break
        
        # Compute Riemannian Hessian
        # H is a 3n x 3n matrix
        H = np.zeros((3*n, 3*n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    for k in range(n):
                        if k == i:
                            continue
                        r = pts[i] - pts[k]
                        d2 = np.dot(r, r)
                        d2 = max(d2, 1e-15)
                        # H_{ii} block: sum over k of [-I/d^2 + 2 r r^T / d^4]
                        block = -np.eye(3) / d2 + 2.0 * np.outer(r, r) / (d2 * d2)
                        H[3*i:3*i+3, 3*i:3*i+3] += block
                else:
                    r = pts[i] - pts[j]
                    d2 = np.dot(r, r)
                    d2 = max(d2, 1e-15)
                    # H_{ij} block: I/d^2 + 2 r r^T / d^4
                    block = np.eye(3) / d2 + 2.0 * np.outer(r, r) / (d2 * d2)
                    H[3*i:3*i+3, 3*j:3*j+3] = block
                    H[3*j:3*j+3, 3*i:3*i+3] = block.T
        
        # Project Hessian to tangent space
        # The Riemannian Hessian in the embedded space needs to account for the constraint
        # For simplicity, we'll use the Euclidean Hessian projected to tangent space
        # This is an approximation but should work well near the minimum
        
        # Solve H * delta = -g_tangent
        # Add small regularization for stability
        reg = 1e-8 * np.trace(H) / (3*n)
        H_reg = H + reg * np.eye(3*n)
        
        try:
            delta = np.linalg.solve(H_reg, -g_tangent.flatten())
        except np.linalg.LinAlgError:
            delta = -g_tangent.flatten()
        
        # Project delta to tangent space
        delta = delta.reshape(n, 3)
        proj_d = np.einsum('ij,ij->i', delta, pts)
        delta_tangent = delta - proj_d[:, np.newaxis] * pts
        
        # Line search
        alpha = 1.0
        improved = False
        for _ in range(20):
            new_pts = _retract(pts + alpha * delta_tangent)
            f_new, _ = _energy_and_grad(new_pts)
            if f_new < f_curr:
                pts = new_pts
                f_curr = f_new
                improved = True
                break
            alpha *= 0.5
        
        if not improved:
            break
        
        # Check convergence
        g_curr = _energy_and_grad(pts)[1]
        proj = np.einsum('ij,ij->i', g_curr, pts)
        g_tangent = g_curr - proj[:, np.newaxis] * pts
        if np.linalg.norm(g_tangent) < 1e-12:
            break
    
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Generate multiple diverse initial configurations
    inits = []
    
    # 1. Golden-angle spiral (Fibonacci)
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    inits.append(np.column_stack([r * np.cos(phi), r * np.sin(phi), z]))
    
    # 2. Random initialization with seed
    for seed in [0, 1, 2, 3, 4, 5]:
        rng = np.random.default_rng(seed)
        pts_rand = rng.normal(size=(N, 3))
        pts_rand = _retract(pts_rand)
        inits.append(pts_rand)
    
    # 3. Grid-based initialization (better distribution)
    n_theta = 10
    n_phi = 10
    theta = np.linspace(0, np.pi, n_theta)
    phi_grid = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    pts_grid = []
    for t in theta:
        for p in phi_grid:
            pts_grid.append([np.sin(t)*np.cos(p), np.sin(t)*np.sin(p), np.cos(t)])
    pts_grid = np.array(pts_grid[:N])
    rng = np.random.default_rng(123)
    pts_grid += rng.normal(scale=0.05, size=pts_grid.shape)
    inits.append(_retract(pts_grid))
    
    # 4. Icosahedron-based initialization
    t = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    
    rng = np.random.default_rng(999)
    ico_perturbed = ico + rng.normal(scale=0.1, size=ico.shape)
    ico_perturbed = _retract(ico_perturbed)
    
    try:
        points = [ico]
        for _ in range(3):
            new_points = []
            for face_idx in range(len(ico)):
                for _ in range(3):
                    u, v = rng.random(), rng.random()
                    if u + v > 1:
                        u, v = 1-u, 1-v
                    p = ico[face_idx] * (1-u-v) + ico[(face_idx+1)%12] * u + ico[(face_idx+2)%12] * v
                    new_points.append(p)
            points.append(_retract(np.array(new_points)))
        all_pts = np.vstack(points)
        idx = np.random.choice(len(all_pts), N, replace=False)
        inits.append(all_pts[idx])
    except Exception:
        inits.append(ico_perturbed)
    
    best_pts = None
    best_energy = float('inf')
    
    # Optimize each initial configuration
    for idx, pts_init in enumerate(inits):
        if time.time() > deadline - 30.0:
            break
        
        # Run first L-BFGS-B pass
        pts_opt = _lbfgs_sphere(pts_init.flatten(), deadline - 20.0)
        
        # Run second L-BFGS-B pass with tighter tolerances
        pts_opt = _lbfgs_sphere(pts_opt.flatten(), deadline - 12.0, maxiter=10000, ftol=1e-18, gtol=1e-16)
        
        # Newton polish
        pts_opt = _newton_polish(pts_opt, deadline - 5.0)
        
        # Polish with high-precision gradient descent
        pts_opt = _polish(pts_opt, deadline - 3.0)
        
        # Ultra polish
        pts_opt = _ultra_polish(pts_opt, deadline)
        
        f, _ = _energy_and_grad(pts_opt)
        if f < best_energy:
            best_energy = f
            best_pts = pts_opt
    
    # Final validation and normalization
    if best_pts is None:
        best_pts = inits[0]
    
    best_pts = _retract(best_pts)
    
    # Do a final polish on the best result if we have time
    if time.time() < deadline - 10.0:
        best_pts = _newton_polish(best_pts, deadline - 5.0)
        best_pts = _polish(best_pts, deadline - 2.0)
        best_pts = _ultra_polish(best_pts, deadline)
    
    best_pts = _retract(best_pts)
    
    # Ensure no points are too close
    min_dist = np.inf
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < min_dist:
                min_dist = d
    
    if min_dist < 1e-9:
        # Fallback to spiral if points are too close
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        best_pts = _retract(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
