"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import math
import time

N = 150


def _energy_and_grad(pts):
    """
    Compute logarithmic energy and its gradient for points on the unit sphere.
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Gradient: for each point i, grad_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    """
    # Compute pairwise distances
    # pts is (N, 3)
    # diff[i, j] = pts[i] - pts[j]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 3)
    dist = np.sqrt(np.sum(diff * diff, axis=2))  # (N, N)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist, 1.0)
    
    # Energy: sum over i<j of -log(dist[i,j])
    # Use upper triangle
    iu = np.triu_indices(N, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient: for each point i, sum over j!=i of (pts[i] - pts[j]) / dist[i,j]^2
    # diff[i,j] = pts[i] - pts[j]
    # grad[i] = sum_j (diff[i,j]) / dist[i,j]^2
    # Note: dist[i,j]^2 = sum(diff[i,j]^2)
    dist_sq = dist * dist
    np.fill_diagonal(dist_sq, 1.0)
    
    # grad[i,j] component = diff[i,j] / dist_sq[i,j]
    # Sum over j for each i
    grad = np.sum(diff / dist_sq[:, :, np.newaxis], axis=1)
    
    return energy, grad


def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return pts / norms


def _halton_sequence(n, dims=3):
    """Generate Halton sequence points in [0,1)^dims."""
    points = np.zeros((n, dims))
    # First few primes for 3D
    primes = [2, 3, 5]
    for d in range(dims):
        p = primes[d]
        for i in range(n):
            f = 1.0
            val = 0.0
            j = i + 1
            while j > 0:
                f /= p
                val += f * (j % p)
                j //= p
            points[i, d] = val
    return points


def _halton_to_sphere(n):
    """Map Halton sequence to sphere using inverse CDF for uniform distribution."""
    h = _halton_sequence(n, 3)
    # Use first two dimensions for theta and phi
    # Actually, for uniform on sphere: z = 2*u - 1, phi = 2*pi*v
    u = h[:, 0]
    v = h[:, 1]
    z = 2.0 * u - 1.0
    phi = 2.0 * np.pi * v
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _project_to_sphere(pts)


def _fibonacci_sphere(n):
    """Golden-angle Fibonacci spiral points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _project_to_sphere(pts)


def _cartesian_gradient_step(pts, grad, lr):
    """
    Take a gradient step in Cartesian space, then project back to sphere.
    We minimize energy, so we move in direction -grad.
    """
    new_pts = pts - lr * grad
    return _project_to_sphere(new_pts)


def _riemannian_cg_step(pts, grad, prev_search_dir, alpha, beta):
    """
    Riemannian conjugate gradient step on the sphere.
    The Riemannian gradient is the tangential component of the Euclidean gradient.
    For points on the unit sphere, the tangent space at x is {v : x·v = 0}.
    Tangential component: grad_tan = grad - (grad·x) * x
    """
    # Compute tangential gradient
    grad_dot_x = np.sum(grad * pts, axis=1, keepdims=True)
    grad_tan = grad - grad_dot_x * pts
    
    # Conjugate gradient search direction
    if prev_search_dir is None:
        search_dir = grad_tan
    else:
        # Beta coefficient (Polak-Ribiere)
        grad_dot_prev = np.sum(grad_tan * prev_search_dir, axis=1)
        prev_norm_sq = np.sum(prev_search_dir * prev_search_dir, axis=1)
        prev_norm_sq = np.maximum(prev_norm_sq, 1e-12)
        beta = np.maximum(0.0, grad_dot_prev / prev_norm_sq)
        # Vector beta per point? Actually, standard CG uses a scalar beta.
        # Let's use a global beta for simplicity, or per-point.
        # Per-point beta:
        search_dir = grad_tan + beta * prev_search_dir
        # Re-orthogonalize search direction to tangent space
        sd_dot_x = np.sum(search_dir * pts, axis=1, keepdims=True)
        search_dir = search_dir - sd_dot_x * pts
    
    # Line search: backtracking
    current_energy, _ = _energy_and_grad(pts)
    step = 1.0
    max_iter = 20
    best_pts = pts.copy()
    best_energy = current_energy
    
    for _ in range(max_iter):
        # Exponential map on sphere: exp_x(v) = x*cos(|v|) + (v/|v|)*sin(|v|)
        # For small steps, this is approximately x + v
        v = alpha * step * search_dir
        v_norm = np.linalg.norm(v, axis=1, keepdims=True)
        v_norm_safe = np.maximum(v_norm, 1e-12)
        # exp_x(v) = x * cos(theta) + (v/|v|) * sin(theta), where theta = |v|
        # For small theta, cos(theta) ≈ 1, sin(theta)/theta ≈ 1
        # More precisely:
        theta = v_norm
        cos_t = np.cos(theta)
        sin_t_over_theta = np.ones_like(theta)
        mask = theta > 1e-8
        sin_t_over_theta[mask] = np.sin(theta[mask]) / theta[mask]
        
        new_pts = pts * cos_t + v * sin_t_over_theta
        new_pts = _project_to_sphere(new_pts)
        
        new_energy, _ = _energy_and_grad(new_pts)
        if new_energy < best_energy:
            best_energy = new_energy
            best_pts = new_pts.copy()
        # Armijo condition
        if new_energy <= current_energy - 1e-4 * step * np.sum(np.abs(grad_tan * search_dir)):
            return best_pts, search_dir
        step *= 0.5
    
    return best_pts, search_dir


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Generate multiple initializations
    inits = []
    inits.append(_fibonacci_sphere(N))
    inits.append(_halton_to_sphere(N))
    
    # Add some random initializations
    rng = np.random.RandomState(42)
    for _ in range(5):
        pts = rng.randn(N, 3)
        pts = _project_to_sphere(pts)
        inits.append(pts)
    
    best_energy = float('inf')
    best_pts = inits[0].copy()
    
    for init_pts in inits:
        if time.time() > deadline:
            break
        
        pts = init_pts.copy()
        energy, grad = _energy_and_grad(pts)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
        
        # Phase 1: Cartesian gradient descent with adaptive step size
        lr = 0.01
        prev_energy = energy
        stall_count = 0
        
        max_cart_steps = 2000
        for step in range(max_cart_steps):
            if time.time() > deadline:
                break
            
            new_pts = _cartesian_gradient_step(pts, grad, lr)
            new_energy, new_grad = _energy_and_grad(new_pts)
            
            if new_energy < prev_energy - 1e-10:
                pts = new_pts
                grad = new_grad
                prev_energy = new_energy
                stall_count = 0
                # Adaptive step size
                if step % 100 == 0:
                    lr *= 1.1
            else:
                stall_count += 1
                lr *= 0.5
                if stall_count > 50:
                    break
            
            if energy < best_energy:
                best_energy = energy
                best_pts = pts.copy()
            energy = new_energy
        
        # Phase 2: Riemannian conjugate gradient
        energy, grad = _energy_and_grad(pts)
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
        
        prev_search_dir = None
        alpha = 0.01
        max_cg_steps = 500
        stall_count = 0
        prev_energy = energy
        
        for step in range(max_cg_steps):
            if time.time() > deadline:
                break
            
            pts, prev_search_dir = _riemannian_cg_step(pts, grad, prev_search_dir, alpha, None)
            new_energy, new_grad = _energy_and_grad(pts)
            
            if new_energy < prev_energy - 1e-10:
                grad = new_grad
                prev_energy = new_energy
                stall_count = 0
                # Adaptive step size
                if step % 50 == 0:
                    alpha *= 1.05
            else:
                stall_count += 1
                alpha *= 0.5
                if stall_count > 30:
                    break
            
            if energy < best_energy:
                best_energy = energy
                best_pts = pts.copy()
            energy = new_energy
        
        # Phase 3: Final fine-tuning with small Cartesian steps
        lr = 0.001
        for step in range(500):
            if time.time() > deadline:
                break
            
            new_pts = _cartesian_gradient_step(pts, grad, lr)
            new_energy, new_grad = _energy_and_grad(new_pts)
            
            if new_energy < energy - 1e-12:
                pts = new_pts
                grad = new_grad
                energy = new_energy
                if energy < best_energy:
                    best_energy = energy
                    best_pts = pts.copy()
            else:
                lr *= 0.5
                if lr < 1e-8:
                    break
    
    # Ensure valid output
    best_pts = _project_to_sphere(best_pts)
    
    # Check for non-finite values
    if not np.all(np.isfinite(best_pts)):
        return _fibonacci_sphere(N)
    
    # Check for points too close
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, 1.0)
    if np.min(dist) < 1e-9:
        return _fibonacci_sphere(N)
    
    return best_pts


# EVOLVE-BLOCK-START
# (The construct function is defined above; this block is kept for structural compatibility)
# EVOLVE-BLOCK-END
