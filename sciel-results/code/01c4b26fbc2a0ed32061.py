"""Direct approach for minimal logarithmic energy of 150 points on sphere."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150

# Halton sequence for better initial coverage
def _halton_sequence(n, base):
    """Generate Halton sequence in [0,1) for given base."""
    seq = []
    f = 1.0
    r = 0.0
    for i in range(1, n + 1):
        f /= base
        r += f
        while i % base == 0:
            f /= base
            i //= base
    seq = []
    f = 1.0
    r = 0.0
    for i in range(1, n + 1):
        f /= base
        r += f
        while i % base == 0:
            f /= base
            i //= base
        seq.append(r)
    return np.array(seq)

def _halton_sphere(n):
    """Generate Halton sequence points on unit sphere using 3D Halton + mapping."""
    # Use 3D Halton sequence and map to sphere via inverse CDF of normal
    x = _halton_sequence(n, 2)
    y = _halton_sequence(n, 3)
    z = _halton_sequence(n, 5)
    
    # Convert uniform [0,1) to spherical coordinates
    # z uniform in [-1,1]
    z = 2.0 * z - 1.0
    # theta uniform in [0, 2*pi)
    theta = 2.0 * np.pi * x
    # phi from y: sin(phi) = sqrt(1-z^2), use y to determine radius distribution
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    
    pts = np.column_stack([
        r * np.cos(theta),
        r * np.sin(theta),
        z
    ])
    pts = _normalize(pts)
    return pts


def _log_energy(pts):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|) using math.fsum."""
    n = len(pts)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-12:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)


def _energy_and_grad(pts):
    """Vectorized energy and gradient (projected to tangent space)."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, 1.0)
    inv_d2 = 1.0 / (dists ** 2)
    np.fill_diagonal(inv_d2, 0.0)
    
    weighted_sum_pts = inv_d2 @ pts
    sum_inv_d2 = np.sum(inv_d2, axis=1)
    grad = -pts * sum_inv_d2[:, np.newaxis] + weighted_sum_pts
    
    # Project gradient to tangent space
    dot_prod = np.sum(grad * pts, axis=1)
    grad -= dot_prod[:, np.newaxis] * pts
    
    # Compute energy (upper triangle only)
    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    d = np.maximum(d, 1e-15)
    energy = np.sum(np.log(1.0 / d))
    
    return energy, grad


def _log_energy_fast(pts):
    """Vectorized log energy computation for speed."""
    n = len(pts)
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    d = np.maximum(d, 1e-15)
    return np.sum(np.log(1.0 / d))


def _fibonacci_sphere(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _riemannian_cg(pts, time_limit=15.0):
    """Riemannian conjugate gradient on the sphere."""
    start = time.time()
    current = pts.copy()
    energy, grad = _energy_and_grad(current)
    
    # Initial direction: negative gradient
    d = -grad.copy()
    # Ensure d is in tangent space
    dot_d = np.sum(d * current, axis=1)
    d -= dot_d[:, np.newaxis] * current
    
    beta = 0.0
    max_iter = 500
    
    for it in range(max_iter):
        if time.time() - start > time_limit - 0.5:
            break
        
        # Line search along geodesic
        alpha = 0.1
        best_energy = energy
        best_pts = current.copy()
        best_alpha = 0.0
        
        for ls in range(30):
            # Move along geodesic: exp_{current}(alpha * d)
            # For small alpha, use approximation: current + alpha*d, then normalize
            # Better: use Rodrigues' rotation or exponential map
            # Simple approach: normalize(current + alpha * d)
            new_pts = current + alpha * d
            new_pts = _normalize(new_pts)
            
            new_energy = _log_energy_fast(new_pts)
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = new_pts.copy()
                best_alpha = alpha
            
            if new_energy >= energy:
                alpha *= 0.5
            else:
                break
        
        if best_alpha == 0.0:
            break
        
        current = best_pts
        energy = best_energy
        
        # Compute new gradient
        new_energy, new_grad = _energy_and_grad(current)
        new_grad_norm = np.linalg.norm(new_grad)
        
        if new_grad_norm < 1e-14:
            break
        
        # Update direction (Polak-Ribiere)
        old_grad = grad
        grad = new_grad
        
        # Ensure grad is in tangent space
        dot_grad = np.sum(grad * current, axis=1)
        grad -= dot_grad[:, np.newaxis] * current
        
        # Polak-Ribiere beta
        num = np.sum(grad * (grad - old_grad))
        den = np.sum(old_grad * old_grad)
        if den > 1e-30:
            beta = max(0.0, num / den)
        else:
            beta = 0.0
        
        d = -grad + beta * d
        
        # Ensure d is in tangent space
        dot_d = np.sum(d * current, axis=1)
        d -= dot_d[:, np.newaxis] * current
        
        # Check if d is a descent direction
        if np.sum(d * grad) > 0:
            d = -grad.copy()
            dot_d = np.sum(d * current, axis=1)
            d -= dot_d[:, np.newaxis] * current
    
    return current, energy


def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _lbfgs_optimize(pts, time_limit=30.0):
    """Optimize using L-BFGS-B with projected gradient."""
    start = time.time()
    
    # Flatten points for L-BFGS-B
    x0 = pts.flatten()
    
    def fun_and_grad(x):
        p = x.reshape(N, 3)
        p = _normalize(p)
        energy, grad = _energy_and_grad(p)
        # Project gradient back to flat space
        return energy, grad.flatten()
    
    # Use L-BFGS-B with bounds to keep points reasonable
    # No explicit bounds needed since we normalize after each step
    res = minimize(fun_and_grad, x0, method='L-BFGS-B', jac=True,
                   options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12,
                            'maxfun': 10000, 'maxls': 50})
    
    # Check time
    if time.time() - start > time_limit:
        return pts, _log_energy_fast(pts)
    
    # Extract optimized points
    x_opt = res.x.reshape(N, 3)
    x_opt = _normalize(x_opt)
    energy = _log_energy_fast(x_opt)
    
    return x_opt, energy


def _polish(pts, time_limit=5.0):
    """High-precision polish with very small steps and adaptive strategy."""
    start = time.time()
    current = pts.copy()
    energy = _log_energy_fast(current)
    step_size = 1e-3
    
    for it in range(20000):
        if time.time() - start > time_limit - 0.1:
            break
        
        e, grad = _energy_and_grad(current)
        grad_norm = np.linalg.norm(grad)
        
        if grad_norm < 1e-15:
            break
        
        # Backtracking line search with very tight tolerance
        alpha = step_size
        accepted = False
        for _ in range(40):
            new_pts = current - alpha * grad
            new_pts = _normalize(new_pts)
            new_energy = _log_energy_fast(new_pts)
            
            if new_energy < e - 1e-15:
                accepted = True
                break
            alpha *= 0.5
        
        if accepted:
            current = new_pts
            energy = new_energy
            # Increase step size if we took a full step
            if alpha >= step_size * 0.99:
                step_size = min(step_size * 1.1, 0.05)
            else:
                step_size = min(alpha * 1.05, 0.05)
        else:
            step_size *= 0.5
            if step_size < 1e-17:
                break
    
    return current, energy


def _grad_descent(pts, time_limit=10.0):
    """Gradient descent with adaptive step size."""
    start = time.time()
    current = pts.copy()
    energy = _log_energy_fast(current)
    step_size = 0.01
    
    for it in range(5000):
        if time.time() - start > time_limit - 0.1:
            break
        
        e, grad = _energy_and_grad(current)
        grad_norm = np.linalg.norm(grad)
        
        if grad_norm < 1e-14:
            break
        
        # Backtracking line search
        alpha = step_size
        accepted = False
        for _ in range(50):
            new_pts = current - alpha * grad
            new_pts = _normalize(new_pts)
            new_energy = _log_energy_fast(new_pts)
            
            if new_energy < e - 1e-12:
                accepted = True
                break
            alpha *= 0.5
        
        if accepted:
            current = new_pts
            energy = new_energy
            # Increase step size if we took a full step
            if alpha == step_size:
                step_size = min(step_size * 1.1, 0.1)
        else:
            step_size *= 0.5
            if step_size < 1e-16:
                break
    
    return current, energy


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initial Fibonacci spiral
    pts_fib = _fibonacci_sphere(N)
    
    best_pts = pts_fib.copy()
    best_energy = _log_energy_fast(pts_fib)
    
    # Generate diverse initializations
    restarts = [
        ("fibonacci", pts_fib.copy()),
        ("halton", _halton_sphere(N).copy()),
    ]
    
    # Random initializations with different seeds
    for seed in range(6):
        rng_local = np.random.default_rng(seed * 123 + 7)
        pts_rand = rng_local.standard_normal((N, 3))
        pts_rand = _normalize(pts_rand)
        restarts.append((f"random_{seed}", pts_rand))
    
    # Perturbed Fibonacci
    for seed in range(3):
        rng_local = np.random.default_rng(seed * 456 + 13)
        noise_level = 0.02 * (seed + 1)
        pts_pert = pts_fib + rng_local.standard_normal((N, 3)) * noise_level
        pts_pert = _normalize(pts_pert)
        restarts.append((f"perturbed_{seed}", pts_pert))
    
    # Perturbed Halton
    for seed in range(2):
        rng_local = np.random.default_rng(seed * 789 + 23)
        noise_level = 0.015 * (seed + 1)
        pts_halt = _halton_sphere(N)
        pts_pert = pts_halt + rng_local.standard_normal((N, 3)) * noise_level
        pts_pert = _normalize(pts_pert)
        restarts.append((f"halton_pert_{seed}", pts_pert))
    
    # Optimize each restart
    for name, init_pts in restarts:
        if time.time() > deadline - 20:
            break
        
        # First pass: L-BFGS-B optimization
        time_left = deadline - time.time() - 8
        if time_left < 3:
            break
        
        opt_pts, opt_energy = _lbfgs_optimize(init_pts, time_limit=min(time_left, 12.0))
        
        # Second pass: Riemannian conjugate gradient
        if time.time() < deadline - 6:
            cg_time = min(deadline - time.time() - 3, 8.0)
            if cg_time > 1.0:
                cg_pts, cg_energy = _riemannian_cg(opt_pts, time_limit=cg_time)
                if cg_energy < opt_energy:
                    opt_pts = cg_pts
                    opt_energy = cg_energy
        
        # Third pass: Gradient descent for fine-tuning
        if time.time() < deadline - 4:
            gd_time = min(deadline - time.time() - 2, 4.0)
            if gd_time > 0.5:
                gd_pts, gd_energy = _grad_descent(opt_pts, time_limit=gd_time)
                if gd_energy < opt_energy:
                    opt_pts = gd_pts
                    opt_energy = gd_energy
        
        # Fourth pass: Polish
        if time.time() < deadline - 2:
            polish_time = min(deadline - time.time() - 0.5, 4.0)
            if polish_time > 0.3:
                polished_pts, polished_energy = _polish(opt_pts, time_limit=polish_time)
                if polished_energy < opt_energy:
                    opt_pts = polished_pts
                    opt_energy = polished_energy
        
        # Update best
        if opt_energy < best_energy:
            best_energy = opt_energy
            best_pts = opt_pts.copy()
    
    # Final high-precision polish on best solution
    if time.time() < deadline - 1:
        final_time = min(deadline - time.time() - 0.3, 6.0)
        if final_time > 0.3:
            final_pts, final_energy = _polish(best_pts, time_limit=final_time)
            if final_energy < best_energy:
                best_energy = final_energy
                best_pts = final_pts
    
    # Final normalization
    best_pts = _normalize(best_pts)
    
    return best_pts
