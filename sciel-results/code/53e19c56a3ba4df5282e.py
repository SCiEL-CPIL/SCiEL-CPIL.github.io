"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

N = 150

# EVOLVE-BLOCK-START
def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _energy(pts):
    """Compute log(1/d) energy for points on unit sphere."""
    n = pts.shape[0]
    # Compute all pairwise distances
    # Use broadcasting for efficiency
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, 1.0)  # avoid log(0)
    # Only sum over i < j
    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    # Use math.fsum for precision as specified
    terms = [math.log(1.0 / di) for di in d]
    return math.fsum(terms)


def _energy_fast(pts):
    """Faster energy computation using numpy, slightly less precise but good enough for optimization."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, 1.0)
    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    # Clip to avoid log(0)
    d = np.maximum(d, 1e-15)
    return np.sum(np.log(1.0 / d))


def _gradient(pts):
    """Compute gradient of log(1/d) energy.
    
    E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    dE/dx_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = pts.shape[0]
    grad = np.zeros_like(pts)
    for i in range(n):
        diff = pts[i] - pts  # shape (n, 3)
        dists = np.sqrt(np.sum(diff * diff, axis=1))
        dists[i] = 1.0  # avoid division by zero
        # grad_i = sum_j (x_i - x_j) / |x_i - x_j|^2
        weights = 1.0 / (dists * dists)
        weights[i] = 0.0
        grad[i] = np.sum(diff * weights[:, np.newaxis], axis=0)
    return grad


def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.sqrt(np.sum(pts * pts, axis=1))
    norms = np.maximum(norms, 1e-15)
    return pts / norms[:, np.newaxis]


def _project_to_sphere(pts, grad):
    """Project gradient to tangent space of sphere."""
    # Tangent space: remove radial component
    radial = np.sum(pts * grad, axis=1, keepdims=True)
    return grad - radial * pts


def _lbfgs_sphere(pts, max_iter=2000, tol=1e-10):
    """L-BFGS optimization on the sphere using retraction (normalize after each step)."""
    n = pts.shape[0]
    x = pts.copy()
    
    # L-BFGS parameters
    m = 10  # memory size
    s_history = []
    y_history = []
    
    prev_grad = None
    prev_x = None
    
    for it in range(max_iter):
        grad = _gradient(x)
        # Project to tangent space
        grad_tan = _project_to_sphere(x, grad)
        
        # Check convergence
        if np.max(np.abs(grad_tan)) < tol:
            break
        
        # L-BFGS two-loop recursion
        if prev_grad is not None:
            # Compute alpha using two-loop recursion
            r = -grad_tan
            alphas = np.zeros(len(s_history))
            betas = np.zeros(len(s_history))
            
            for i in range(len(s_history) - 1, -1, -1):
                s = s_history[i]
                y = y_history[i]
                ys = np.sum(s * y)
                if abs(ys) > 1e-30:
                    alphas[i] = np.sum(s * r) / ys
                r = r - alphas[i] * y
            
            # Initial step size estimate
            gamma = 1.0
            if len(s_history) > 0:
                s_last = s_history[-1]
                y_last = y_history[-1]
                ys_last = np.sum(s_last * y_last)
                yy_last = np.sum(y_last * y_last)
                if abs(yy_last) > 1e-30 and abs(ys_last) > 1e-30:
                    gamma = ys_last / yy_last
            
            d = gamma * r
            for i in range(len(s_history)):
                s = s_history[i]
                y = y_history[i]
                ys = np.sum(s * y)
                if abs(ys) > 1e-30:
                    betas[i] = np.sum(y * d) / ys
                d = d + (alphas[i] - betas[i]) * s
            
            # Line search
            step = 1.0
            f_curr = _energy_fast(x)
            for ls in range(20):
                x_new = x + step * d
                x_new = _normalize(x_new)
                f_new = _energy_fast(x_new)
                if f_new < f_curr - 1e-4 * step * np.sum(grad_tan * d):
                    break
                step *= 0.5
            
            # Update L-BFGS memory
            x_new = _normalize(x + step * d)
            grad_new = _gradient(x_new)
            grad_new_tan = _project_to_sphere(x_new, grad_new)
            
            s = x_new - x
            y = grad_new_tan - grad_tan
            sy = np.sum(s * y)
            if sy > 1e-15:
                s_history.append(s)
                y_history.append(y)
                if len(s_history) > m:
                    s_history.pop(0)
                    y_history.pop(0)
            
            x = x_new
        else:
            # First iteration: steepest descent
            step = 1.0
            f_curr = _energy_fast(x)
            d = -grad_tan
            for ls in range(20):
                x_new = x + step * d
                x_new = _normalize(x_new)
                f_new = _energy_fast(x_new)
                if f_new < f_curr - 1e-4 * step * np.sum(grad_tan * d):
                    break
                step *= 0.5
            
            x = _normalize(x + step * d)
        
        prev_grad = grad_tan
        prev_x = x
    
    return x


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Start with Fibonacci sphere
    best_pts = _fibonacci_sphere(N)
    best_energy = _energy_fast(best_pts)
    
    # Run L-BFGS optimization
    try:
        pts_opt = _lbfgs_sphere(best_pts, max_iter=3000, tol=1e-12)
        e_opt = _energy_fast(pts_opt)
        if e_opt < best_energy:
            best_pts = pts_opt
            best_energy = e_opt
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline:
        return _normalize(best_pts)
    
    # Try a few random restarts if time permits
    rng = np.random.default_rng(42)
    for restart in range(5):
        if time.time() > deadline - 5.0:
            break
        
        # Random start
        pts_rand = rng.standard_normal((N, 3))
        pts_rand = _normalize(pts_rand)
        
        # Optimize
        try:
            pts_opt = _lbfgs_sphere(pts_rand, max_iter=1500, tol=1e-10)
            e_opt = _energy_fast(pts_opt)
            if e_opt < best_energy:
                best_pts = pts_opt
                best_energy = e_opt
        except Exception:
            pass
    
    # Final polish with higher precision
    if time.time() < deadline - 2.0:
        try:
            pts_final = _lbfgs_sphere(best_pts, max_iter=1000, tol=1e-13)
            e_final = _energy_fast(pts_final)
            if e_final < best_energy:
                best_pts = pts_final
                best_energy = e_final
        except Exception:
            pass
    
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
