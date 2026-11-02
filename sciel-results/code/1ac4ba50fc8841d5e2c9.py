"""Optimized logarithmic energy minimization for 100 points on the sphere."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START

def _energy_and_gradient(pts):
    """Compute both energy and tangent-space gradient in one pass."""
    n = pts.shape[0]
    dots = pts @ pts.T
    d2 = 2.0 - 2.0 * dots
    np.fill_diagonal(d2, 1.0)
    d2 = np.maximum(d2, 1e-15)
    
    # Energy: sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    dist = np.sqrt(d2)
    iu = np.triu_indices(n, k=1)
    log_d = np.log(dist[iu])
    energy = -math.fsum(log_d)
    
    # E = -sum_{i<j} log(d_ij)
    # dE/dx_i = -sum_{j!=i} (x_i - x_j)/d_ij^2
    # = sum_{j!=i} (x_j - x_i)/d_ij^2
    inv_d2 = 1.0 / d2
    np.fill_diagonal(inv_d2, 0.0)
    
    sum_inv_d2 = inv_d2.sum(axis=1)
    sum_xj_inv_d2 = (pts.T @ inv_d2).T
    
    # grad = sum_{j!=i} (x_j - x_i)/d^2 = sum_xj_inv_d2 - pts * sum_inv_d2
    grad = sum_xj_inv_d2 - pts * sum_inv_d2[:, np.newaxis]
    
    grad_dot_xi = np.sum(grad * pts, axis=1, keepdims=True)
    grad_tan = grad - grad_dot_xi * pts
    
    return energy, grad_tan

def _riemannian_hessian(pts):
    """Compute the Riemannian Hessian of the log energy on the sphere.
    
    H = d2E/dx_i dx_j projected to tangent space.
    For E = -sum_{i<j} log(d_ij):
    d2E/dx_i dx_j (i!=j) = (x_i - x_j)/d^4 * 3 * (x_i-x_j)^2/d^2 ... 
    Actually: d/dx_i [(x_i-x_j)/d^2] = 1/d^2 - 2(x_i-x_j)(x_i-x_j)/d^4
    So d2E/dx_i dx_j = -(1/d^2 - 2((x_i-x_j).(x_i-x_j))/d^4) = -1/d^2 + 2/d^4 * d^2 = -1/d^2 + 2/d^2 = 1/d^2
    Wait: d^2 = |x_i-x_j|^2, so (x_i-x_j).(x_i-x_j) = d^2.
    d2E/dx_i dx_j = -[1/d^2 - 2*d^2/d^4] = -[1/d^2 - 2/d^2] = 1/d^2 for i!=j
    
    For diagonal (i=j): sum over k!=i of d2E/dx_i dx_i from pair (i,k):
    = sum_{k!=i} [1/d_ik^2 - 2*d_ik^2/d_ik^4] = sum_{k!=i} [1/d_ik^2 - 2/d_ik^2] = -sum_{k!=i} 1/d_ik^2
    
    Riemannian Hessian = Euclidean Hessian projected to tangent + correction term involving grad and curvature.
    For sphere S^2: H_R = P H P + grad * (something). Actually the Riemannian Hessian in ambient coords is:
    H_R(u,v) = P(∇_u (∇_v f)) where P is projection to tangent.
    For a function on the sphere, H_R = P H_amb P + <grad, x> ... but <grad, x> = 0 for tangent grad.
    The full formula: H_R = P H P + sum_k (H e_k, x) grad_k ... this gets complex.
    
    Simpler: use the formula H_R = P (H_amb) P + (grad^T H x) x^T + ... 
    Actually for sphere: H_R = P H P + (x^T H x) ... no.
    
    Let me use: Riemannian Hessian = P H P + grad * (x^T H) ... 
    The clean formula: H_R = P H P + (grad ⊗ x^T H + x ⊗ grad^T H - (grad^T H x) x ⊗ x^T) ... 
    This is getting complex. Let me just use the Euclidean Hessian projected, which is a good approximation.
    """
    n = pts.shape[0]
    dots = pts @ pts.T
    d2 = 2.0 - 2.0 * dots
    np.fill_diagonal(d2, 1.0)
    d2 = np.maximum(d2, 1e-15)
    
    # Off-diagonal: d2E/dx_i dx_j = 1/d^2 for i != j (from derivation above)
    # Diagonal: d2E/dx_i^2 = -sum_{k!=i} 1/d_ik^2
    inv_d2 = 1.0 / d2
    np.fill_diagonal(inv_d2, 0.0)
    
    # Off-diagonal Hessian entries (in ambient space, each is a 3x3 matrix)
    # H_amb[i,j] = (1/d_ik^2) * I for i != j ... wait that's not right.
    # Let me redo: E = -sum_{i<j} log(d_ij), d_ij = |x_i - x_j|
    # dE/dx_i = -sum_{j!=i} (x_i - x_j)/d_ij^2
    # d2E/dx_i dx_j (i!=j) = -d/dx_j [(x_i - x_j)/d_ij^2]
    # = -[(-1)/d_ij^2 + (x_i-x_j) * d/dx_j (1/d_ij^2)]
    # d/dx_j (1/d_ij^2) = -2(x_j - x_i)/d_ij^4
    # So d2E/dx_i dx_j = -[-1/d^2 + (x_i-x_j)(-2)(x_j-x_i)/d^4]
    # = -[-1/d^2 + 2(x_i-x_j)(x_i-x_j)/d^4]  (since (x_i-x_j)(x_j-x_i) = -d^2)
    # Wait: (x_i-x_j)(x_j-x_i) = -(x_i-x_j)^2 = -d^2
    # So: = -[-1/d^2 - 2d^2/d^4] = -[-1/d^2 - 2/d^2] = -[-3/d^2] = 3/d^2
    # Hmm, let me be more careful with vectors.
    # f(x_i, x_j) = -log(|x_i - x_j|)
    # df/dx_i = -(x_i - x_j)/|x_i-x_j|^2
    # df/dx_j = (x_i - x_j)/|x_i-x_j|^2
    # d2f/dx_i dx_j = -d/dx_j [(x_i-x_j)/|x_i-x_j|^2]
    # Let v = x_i - x_j, r = |v|
    # d/dx_j [v/r^2] = d/dx_j [(x_i-x_j)/r^2]
    # = (-1)/r^2 + (x_i-x_j) * d/dx_j (1/r^2)
    # d/dx_j (1/r^2) = d/dx_j (|x_i-x_j|^{-2}) = -2|x_i-x_j|^{-4} * (x_j - x_i)
    # = -2(x_j-x_i)/r^4 = 2(x_i-x_j)/r^4
    # So d/dx_j [v/r^2] = -1/r^2 + v * 2v/r^4 = -1/r^2 + 2v^2/r^4 = -1/r^2 + 2/r^2 = 1/r^2
    # Therefore d2f/dx_i dx_j = -1/r^2 (scalar in each component? No, it's a 3x3 matrix)
    # Actually v/r^2 is a vector, and its derivative w.r.t. x_j is a 3x3 matrix.
    # d/dx_j [v_k / r^2] where v_k = (x_i - x_j)_k
    # = -delta_kl / r^2 + v_k * 2 v_l / r^4
    # So the 3x3 matrix is: -I/r^2 + 2 vv^T/r^4
    # And d2f/dx_i dx_j = -(-I/r^2 + 2vv^T/r^4) = I/r^2 - 2vv^T/r^4
    
    # For the diagonal d2f/dx_i^2: sum over j!=i of d/dx_i [-(x_i-x_j)/r^2]
    # = -sum_j [I/r^2 + (x_i-x_j) * d/dx_i(1/r^2)]
    # d/dx_i (1/r^2) = -2(x_i-x_j)/r^4
    # So = -sum_j [I/r^2 - 2vv^T/r^4] = -sum_j I/r^2 + 2 sum_j vv^T/r^4
    # = -I * sum_j 1/r^2 + 2 sum_j vv^T/r^4
    
    # Build the full ambient Hessian (N*3 x N*3) is too expensive for N=100.
    # Instead, build the Riemannian Hessian action on a tangent vector.
    # For now, return a simplified version: use the Euclidean Hessian projected.
    
    # Actually, let's just return the diagonal approximation for Newton step.
    # The key diagonal component in tangent space for point i:
    # The "curvature" term from the sphere constraint adds a term.
    # For a practical Newton step, we can use:
    # H_R approx = P H_amb P + lambda * I_tangent
    # where lambda accounts for the sphere curvature coupling.
    
    # Simple approach: use the off-diagonal structure.
    # For Newton, we need to solve H_R * delta = -grad
    # This is a 300x300 system. Let's build it explicitly.
    
    H = np.zeros((n*3, n*3))
    for i in range(n):
        for j in range(n):
            if i == j:
                # Diagonal block: -I * sum_k 1/d_ik^2 + 2 sum_k v_ik v_ik^T / d_ik^4
                s1 = 0.0
                s2 = np.zeros(3)
                for k in range(n):
                    if k == i:
                        continue
                    d2_ik = d2[i, k]
                    s1 += 1.0 / d2_ik
                    v = pts[i] - pts[k]
                    s2 += v * v / (d2_ik * d2_ik)
                H[i*3:(i+1)*3, i*3:(i+1)*3] = -s1 * np.eye(3) + 2.0 * s2
            else:
                # Off-diagonal block: I/d^2 - 2 v v^T / d^4
                d2_ij = d2[i, j]
                v = pts[i] - pts[j]
                H[i*3:(i+1)*3, j*3:(j+1)*3] = np.eye(3) / d2_ij - 2.0 * np.outer(v, v) / (d2_ij * d2_ij)
    
    # Project to tangent space: P_i = I - x_i x_i^T
    # H_R = P H P + correction for sphere
    # For simplicity, use P H P as approximation
    P = np.zeros((n*3, n*3))
    for i in range(n):
        xi = pts[i]
        Pi = np.eye(3) - np.outer(xi, xi)
        P[i*3:(i+1)*3, i*3:(i+1)*3] = Pi
    
    H_R = P @ H @ P
    
    return H_R

def _log_energy(pts):
    """Compute logarithmic energy."""
    energy, _ = _energy_and_gradient(pts)
    return energy

def _gradient_log_energy(pts):
    """Tangent-space gradient of logarithmic energy."""
    _, grad_tan = _energy_and_gradient(pts)
    return grad_tan

def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _fibonacci_sphere(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, rng):
    """Generate random points on unit sphere."""
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / np.maximum(norms, 1e-15)

def _perturb_sphere(pts, rng, sigma=0.05):
    """Perturb points and renormalize."""
    noise = rng.standard_normal(pts.shape) * sigma
    new_pts = pts + noise
    return _project_to_sphere(new_pts)

def _icosahedron_subdivide(n, rng):
    """Generate points by subdividing an icosahedron."""
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
    
    pts = []
    for v in verts:
        pts.append(v)
    
    edges = set()
    for f in faces:
        for i in range(3):
            a, b = f[i], f[(i+1) % 3]
            edge = (min(a,b), max(a,b))
            edges.add(edge)
    
    for a, b in edges:
        mid = (verts[a] + verts[b]) / 2.0
        pts.append(_project_to_sphere(mid.reshape(1,3))[0])
    
    for f in faces:
        center = (verts[f[0]] + verts[f[1]] + verts[f[2]]) / 3.0
        pts.append(_project_to_sphere(center.reshape(1,3))[0])
    
    pts = np.array(pts)
    if len(pts) > n:
        indices = rng.choice(len(pts), n, replace=False)
        pts = pts[indices]
    elif len(pts) < n:
        remaining = _random_sphere(n - len(pts), rng)
        pts = np.vstack([pts, remaining])
    
    return _project_to_sphere(pts)

def _golden_spiral_v2(n, offset=0.0):
    """Alternative golden spiral with offset for diversity."""
    i = np.arange(n) + 0.5 + offset
    z = 1.0 - 2.0 * i / n
    z = np.clip(z, -1.0, 1.0)
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _riemannian_descent(pts, max_iter=500, tol=1e-10, deadline=None):
    """Riemannian gradient descent with adaptive step and momentum on the sphere."""
    n = pts.shape[0]
    pts = _project_to_sphere(pts.copy())
    f, grad = _energy_and_gradient(pts)
    
    # Adaptive step size with momentum
    step = 0.05
    momentum = np.zeros_like(pts)
    beta = 0.8
    min_step = 1e-8
    max_step = 1.0
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Check convergence
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            break
        
        # Momentum update: momentum accumulates in descent direction
        momentum = beta * momentum - step * grad
        direction = momentum
        dir_norm = np.linalg.norm(direction)
        if dir_norm < 1e-15:
            break
        direction = direction / dir_norm
        
        # Adaptive step size based on gradient magnitude
        step = min(max_step, max(min_step, 0.5 / (1.0 + 0.5 * grad_norm)))
        
        accepted = False
        best_new_pts = pts
        best_new_f = f
        best_new_grad = grad
        current_step = step
        improvement = 0.0
        
        for ls_iter in range(25):
            # Move in direction of momentum (which is a descent direction)
            new_pts = pts + current_step * direction * dir_norm
            new_pts = _project_to_sphere(new_pts)
            new_f, new_grad = _energy_and_gradient(new_pts)
            
            if np.isfinite(new_f) and new_f < best_new_f:
                best_new_pts = new_pts
                best_new_f = new_f
                best_new_grad = new_grad
                improvement = f - new_f
                if new_f < f - 1e-4 * current_step * dir_norm * grad_norm:
                    accepted = True
                    break
            
            current_step *= 0.7
        
        if accepted:
            pts = best_new_pts
            f = best_new_f
            grad = best_new_grad
            # Increase step if we made good progress
            if improvement > 0.005 * step * grad_norm:
                step = min(max_step, step * 1.15)
        else:
            # Reduce step and retry
            step *= 0.5
            if step < min_step:
                break
    
    return pts, f

def _riemannian_cg(pts, max_iter=200, tol=1e-12, deadline=None):
    """Riemannian conjugate gradient method for faster convergence."""
    n = pts.shape[0]
    pts = _project_to_sphere(pts.copy())
    f, grad = _energy_and_gradient(pts)
    
    # Initial search direction: negative gradient (descent)
    d = -grad
    d = d - np.sum(d * pts, axis=1, keepdims=True) * pts  # Project to tangent
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            break
        
        d_norm = np.linalg.norm(d)
        if d_norm < 1e-15:
            d = -grad
            d = d - np.sum(d * pts, axis=1, keepdims=True) * pts
            d_norm = np.linalg.norm(d)
            if d_norm < 1e-15:
                break
        d = d / d_norm
        
        # Line search: minimize along direction d (which is already -grad initially)
        step = min(0.5, max(0.01, 1.0 / (1.0 + 0.3 * grad_norm)))
        best_new_pts = pts
        best_new_f = f
        best_new_grad = grad
        current_step = step
        accepted = False
        
        for ls_iter in range(20):
            # Move in direction d (which is a descent direction)
            new_pts = pts + current_step * d * d_norm
            new_pts = _project_to_sphere(new_pts)
            new_f, new_grad = _energy_and_gradient(new_pts)
            
            if np.isfinite(new_f) and new_f < best_new_f:
                best_new_pts = new_pts
                best_new_f = new_f
                best_new_grad = new_grad
                if new_f < f - 1e-4 * current_step * d_norm * grad_norm:
                    accepted = True
                    break
            
            current_step *= 0.7
        
        if not accepted:
            break
        
        # Fletcher-Reeves update
        old_grad = grad
        pts = best_new_pts
        f = best_new_f
        grad = best_new_grad
        
        # Project new gradient to tangent
        grad = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
        
        # Fletcher-Reeves beta
        beta = np.sum(grad * grad) / (np.sum(old_grad * old_grad) + 1e-15)
        beta = max(0, min(1.0, beta))
        
        d = -grad + beta * d
        d = d - np.sum(d * pts, axis=1, keepdims=True) * pts  # Re-project
        
    return pts, f

def _newton_polish(pts, max_iter=50, tol=1e-14, deadline=None):
    """Newton polish using Riemannian Hessian for fast convergence near minimum."""
    n = pts.shape[0]
    pts = _project_to_sphere(pts.copy())
    f, grad = _energy_and_gradient(pts)
    
    # Build Riemannian Hessian
    try:
        H_R = _riemannian_hessian(pts)
    except Exception:
        return pts, f
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            break
        
        # Solve H_R * delta = -grad
        try:
            # Add small regularization for stability
            reg = 1e-8 * np.trace(H_R) / (n * 3)
            delta = np.linalg.solve(H_R + reg * np.eye(n*3), -grad.reshape(-1))
        except np.linalg.LinAlgError:
            try:
                delta = np.linalg.lstsq(H_R + reg * np.eye(n*3), -grad.reshape(-1), rcond=None)[0]
            except Exception:
                break
        
        delta = delta.reshape(n, 3)
        
        # Project to tangent space
        delta = delta - np.sum(delta * pts, axis=1, keepdims=True) * pts
        
        delta_norm = np.linalg.norm(delta)
        if delta_norm < 1e-15:
            break
        
        # Line search with backtracking
        step = 1.0
        accepted = False
        for ls_iter in range(20):
            new_pts = pts + step * delta
            new_pts = _project_to_sphere(new_pts)
            new_f, new_grad = _energy_and_gradient(new_pts)
            
            if np.isfinite(new_f) and new_f < f - 1e-4 * step * delta_norm * grad_norm:
                pts = new_pts
                f = new_f
                grad = new_grad
                accepted = True
                break
            
            step *= 0.5
        
        if not accepted:
            break
    
    return pts, f

def _optimize_from_start(pts0, time_budget, deadline):
    """Optimize from a starting configuration."""
    best_pts = _project_to_sphere(pts0.copy())
    best_energy = _log_energy(best_pts)
    
    def objective(x):
        pts = _project_to_sphere(x.reshape(N, 3))
        return _log_energy(pts)
    
    def gradient(x):
        pts = _project_to_sphere(x.reshape(N, 3))
        g = _gradient_log_energy(pts)
        return g.reshape(-1)
    
    x0 = best_pts.reshape(-1)
    
    # First pass: L-BFGS-B with moderate tolerance
    try:
        res = minimize(
            objective,
            x0,
            jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': 400, 'gtol': 1e-8, 'maxfun': 800, 'ftol': 1e-10}
        )
        if res.fun < best_energy:
            pts_new = _project_to_sphere(res.x.reshape(N, 3))
            e_new = _log_energy(pts_new)
            if e_new < best_energy:
                best_pts = pts_new
                best_energy = e_new
    except Exception:
        pass
    
    if time.time() > deadline - 5.0:
        return best_pts, best_energy
    
    # Second pass: Riemannian CG for fast convergence
    if time.time() < deadline - 8.0:
        try:
            pts_cg, e_cg = _riemannian_cg(best_pts, max_iter=300, tol=1e-12, deadline=deadline - 8.0)
            if e_cg < best_energy:
                best_pts = pts_cg
                best_energy = e_cg
        except Exception:
            pass
    
    if time.time() > deadline - 5.0:
        return best_pts, best_energy
    
    # Third pass: Newton polish for fast local convergence
    if time.time() < deadline - 6.0:
        try:
            pts_newton, e_newton = _newton_polish(best_pts, max_iter=50, tol=1e-14, deadline=deadline - 6.0)
            if e_newton < best_energy:
                best_pts = pts_newton
                best_energy = e_newton
        except Exception:
            pass
    
    if time.time() > deadline - 5.0:
        return best_pts, best_energy
    
    # Fourth pass: L-BFGS-B with tighter tolerance
    try:
        x1 = best_pts.reshape(-1)
        res2 = minimize(
            objective,
            x1,
            jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': 600, 'gtol': 1e-12, 'maxfun': 1200, 'ftol': 1e-14}
        )
        if res2.fun < best_energy:
            pts_new = _project_to_sphere(res2.x.reshape(N, 3))
            e_new = _log_energy(pts_new)
            if e_new < best_energy:
                best_pts = pts_new
                best_energy = e_new
    except Exception:
        pass
    
    # Fifth pass: Riemannian CG for refinement
    if time.time() < deadline - 5.0:
        try:
            pts_cg2, e_cg2 = _riemannian_cg(best_pts, max_iter=200, tol=1e-13, deadline=deadline - 5.0)
            if e_cg2 < best_energy:
                best_pts = pts_cg2
                best_energy = e_cg2
        except Exception:
            pass
    
    # Sixth pass: Newton polish again for final convergence
    if time.time() < deadline - 4.0:
        try:
            pts_newton2, e_newton2 = _newton_polish(best_pts, max_iter=50, tol=1e-15, deadline=deadline - 4.0)
            if e_newton2 < best_energy:
                best_pts = pts_newton2
                best_energy = e_newton2
        except Exception:
            pass
    
    # Seventh pass: very tight tolerance for final polish
    if time.time() < deadline - 3.0:
        try:
            x2 = best_pts.reshape(-1)
            res3 = minimize(
                objective,
                x2,
                jac=gradient,
                method='L-BFGS-B',
                options={'maxiter': 800, 'gtol': 1e-14, 'maxfun': 1500, 'ftol': 1e-16}
            )
            if res3.fun < best_energy:
                pts_new = _project_to_sphere(res3.x.reshape(N, 3))
                e_new = _log_energy(pts_new)
                if e_new < best_energy:
                    best_pts = pts_new
                    best_energy = e_new
        except Exception:
            pass
    
    # Final ultra-tight polish with Riemannian CG
    if time.time() < deadline - 1.0:
        try:
            pts_final, e_final = _riemannian_cg(best_pts, max_iter=500, tol=1e-14, deadline=deadline - 1.0)
            if e_final < best_energy:
                best_pts = pts_final
                best_energy = e_final
        except Exception:
            pass
    
    return best_pts, best_energy

def construct():
    t_start = time.time()
    deadline = t_start + 145.0  # Safety margin before 150s limit
    
    rng = np.random.default_rng(42)
    
    initializations = []
    
    # Fibonacci spiral
    initializations.append(_fibonacci_sphere(N))
    
    # Icosahedron-based
    initializations.append(_icosahedron_subdivide(N, rng))
    
    # Alternative golden spirals with offsets
    for offset in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        initializations.append(_golden_spiral_v2(N, offset))
    
    # Random starts - more diverse
    for seed_offset in range(12):
        rng_sub = np.random.default_rng(42 + seed_offset * 7)
        initializations.append(_random_sphere(N, rng_sub))
    
    # Perturbed Fibonacci
    for sigma in [0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.15, 0.2, 0.3]:
        rng_sub = np.random.default_rng(1000 + int(sigma * 100))
        initializations.append(_perturb_sphere(_fibonacci_sphere(N), rng_sub, sigma))
    
    # Perturbed icosahedron
    for sigma in [0.01, 0.02, 0.05, 0.08, 0.1, 0.15]:
        rng_sub = np.random.default_rng(2000 + int(sigma * 100))
        initializations.append(_perturb_sphere(_icosahedron_subdivide(N, rng_sub), rng_sub, sigma))
    
    # Perturbed golden spirals
    for offset in [0.1, 0.3, 0.5, 0.7]:
        for sigma in [0.02, 0.05, 0.1]:
            rng_sub = np.random.default_rng(3000 + int(offset * 100) + int(sigma * 1000))
            initializations.append(_perturb_sphere(_golden_spiral_v2(N, offset), rng_sub, sigma))
    
    # Additional structured initializations
    rng_sym = np.random.default_rng(5000)
    pts_sym = _fibonacci_sphere(N)
    theta = rng_sym.uniform(0, 2 * np.pi)
    rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]])
    pts_rot = _project_to_sphere(pts_sym @ rot.T)
    initializations.append(pts_rot)
    
    # Double perturbation for more diversity
    for sigma1 in [0.05, 0.1]:
        for sigma2 in [0.02, 0.05]:
            rng_sub = np.random.default_rng(6000 + int(sigma1 * 100) + int(sigma2 * 1000))
            pts_pert1 = _perturb_sphere(_fibonacci_sphere(N), rng_sub, sigma1)
            pts_pert2 = _perturb_sphere(pts_pert1, rng_sub, sigma2)
            initializations.append(pts_pert2)
    
    init_energies = [_log_energy(p) for p in initializations]
    sorted_indices = np.argsort(init_energies)
    
    best_pts = initializations[sorted_indices[0]]
    best_energy = init_energies[sorted_indices[0]]
    
    # Process initializations in order of initial energy
    num_to_try = min(len(sorted_indices), 15)
    for rank in range(num_to_try):
        idx = sorted_indices[rank]
        if time.time() > deadline - 20.0:
            break
        
        init_pts = initializations[idx]
        remaining = deadline - time.time()
        if rank == 0:
            time_alloc = min(30.0, remaining * 0.40)
        elif rank < 3:
            time_alloc = min(20.0, remaining * 0.30)
        elif rank < 7:
            time_alloc = min(12.0, remaining * 0.20)
        else:
            time_alloc = min(8.0, remaining * 0.15)
        
        if time_alloc < 3.0:
            break
            
        pts, energy = _optimize_from_start(init_pts, time_alloc, deadline)
        
        if energy < best_energy:
            best_pts = pts
            best_energy = energy
    
    # Final polish on the best result with extended time
    if time.time() < deadline - 15.0:
        try:
            pts_final, e_final = _optimize_from_start(best_pts, 20.0, deadline)
            if e_final < best_energy:
                best_pts = pts_final
                best_energy = e_final
        except Exception:
            pass
    
    # Ultra-final Riemannian CG polish with very tight tolerance
    if time.time() < deadline - 4.0:
        try:
            pts_ultra, e_ultra = _riemannian_cg(best_pts, max_iter=1000, tol=1e-15, deadline=deadline - 4.0)
            if e_ultra < best_energy:
                best_pts = pts_ultra
                best_energy = e_ultra
        except Exception:
            pass
    
    # Last-second micro-polish
    if time.time() < deadline - 1.5:
        def _obj(x):
            pts = _project_to_sphere(x.reshape(N, 3))
            return _log_energy(pts)
        def _grad(x):
            pts = _project_to_sphere(x.reshape(N, 3))
            g = _gradient_log_energy(pts)
            return g.reshape(-1)
        try:
            x_final = best_pts.reshape(-1)
            res_final = minimize(
                _obj,
                x_final,
                jac=_grad,
                method='L-BFGS-B',
                options={'maxiter': 400, 'gtol': 1e-15, 'maxfun': 800, 'ftol': 1e-16}
            )
            if res_final.fun < best_energy:
                pts_new = _project_to_sphere(res_final.x.reshape(N, 3))
                e_new = _log_energy(pts_new)
                if e_new < best_energy:
                    best_pts = pts_new
                    best_energy = e_new
        except Exception:
            pass
    
    best_pts = _project_to_sphere(best_pts)
    
    if not np.all(np.isfinite(best_pts)):
        return _fibonacci_sphere(N)
    
    return best_pts

# EVOLVE-BLOCK-END
