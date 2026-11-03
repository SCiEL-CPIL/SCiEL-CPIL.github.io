"""Logarithmic energy minimization for 100 points on the unit sphere.
Multi-stage Riemannian optimization with diverse initializations,
L-BFGS-B polish, and final high-precision refinement."""
import time
import math
import numpy as np

N = 100
_DEADLINE = 150.0
_SAFETY = 18.0
_START = time.time()


def _remaining():
    return _DEADLINE - _SAFETY - (time.time() - _START)


def _project(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return X / n


def _energy(pts):
    """Fast energy computation using matrix operations."""
    n = pts.shape[0]
    sq_norms = np.einsum('ij,ij->i', pts, pts)
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.maximum(dist_sq, 1e-16)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(n, k=1)
    return float(-np.sum(np.log(dist[iu])))


def _energy_grad(pts):
    """Compute energy and gradient (projected onto tangent space)."""
    n = pts.shape[0]
    sq_norms = np.einsum('ij,ij->i', pts, pts)
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.maximum(dist_sq, 1e-16)
    dist = np.sqrt(dist_sq)
    
    iu = np.triu_indices(n, k=1)
    energy = float(-np.sum(np.log(dist[iu])))
    
    # Gradient of -sum log(d_ij) with respect to points
    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)
    
    # grad_i = -sum_j W_ij * x_i + sum_j W_ij * x_j
    row_sums = W.sum(axis=1, keepdims=True)
    grad = -row_sums * pts + (W @ pts)
    
    # Project onto tangent space
    dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - dot * pts
    
    return energy, grad


def _precise_energy(pts):
    """Compute energy using math.fsum for accuracy."""
    n = pts.shape[0]
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-9:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)


def _fibonacci_spiral(n, offset=0.5):
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(rng):
    v = rng.standard_normal((N, 3))
    return _project(v)


def _sobol_like(n, seed=0):
    """Generate well-distributed points using a quasi-random approach."""
    rng = np.random.default_rng(seed)
    points = []
    golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0
    for i in range(n):
        t = (i + 0.5) / n
        z = 1.0 - 2.0 * t
        angle = 2.0 * np.pi * i * (1.0 + 1.0 / golden_ratio) + rng.standard_normal() * 0.1
        r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
        points.append([r * np.cos(angle), r * np.sin(angle), z])
    return _project(np.array(points))


def _gradient_descent(pts, deadline, max_iter=5000, lr_init=0.01, lr_min=1e-12):
    """Adaptive gradient descent with backtracking line search and Riemannian exponential map."""
    current = pts.copy()
    best_e = _energy(current)
    best_pts = current.copy()
    lr = lr_init
    n = N
    c1 = 0.1
    rho = 0.5
    no_improve_count = 0
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        e, g = _energy_grad(current)
        g_norm = np.linalg.norm(g)
        if g_norm < 1e-13:
            break
        
        # Backtracking line search
        step = lr
        found = False
        for bt in range(20):
            v = -step * g
            v_norms = np.linalg.norm(v, axis=1, keepdims=True)
            v_norms_safe = np.where(v_norms < 1e-12, 1.0, v_norms)
            v_unit = v / v_norms_safe
            cos_t = np.cos(v_norms)
            sin_t = np.sin(v_norms)
            new_pts = current * cos_t + v_unit * sin_t
            new_pts = _project(new_pts)
            
            new_e = _energy(new_pts)
            
            if new_e <= e + c1 * step * g_norm * g_norm:
                found = True
                break
            step *= rho
        
        if not found:
            step = max(lr * rho**20, 1e-15)
            v = -step * g
            v_norms = np.linalg.norm(v, axis=1, keepdims=True)
            v_norms_safe = np.where(v_norms < 1e-12, 1.0, v_norms)
            v_unit = v / v_norms_safe
            cos_t = np.cos(v_norms)
            sin_t = np.sin(v_norms)
            new_pts = current * cos_t + v_unit * sin_t
            new_pts = _project(new_pts)
            new_e = _energy(new_pts)
        
        if new_e < best_e - 1e-15:
            current = new_pts
            best_e = new_e
            best_pts = current.copy()
            no_improve_count = 0
            lr = min(step * 1.5, 0.1)
        elif new_e < best_e:
            current = new_pts
            no_improve_count += 1
            lr = step
        else:
            lr = step * 0.5
            no_improve_count += 1
            if lr < lr_min or no_improve_count > 50:
                break
    
    return best_pts, best_e


def _momentum_gd(pts, deadline, max_iter=3000, lr_init=0.03, beta=0.9):
    """Momentum-based gradient descent with Riemannian updates."""
    current = pts.copy()
    best_e = _energy(current)
    best_pts = current.copy()
    
    velocity = np.zeros_like(current)
    lr = lr_init
    c1 = 0.1
    rho = 0.5
    no_improve_count = 0
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        e, g = _energy_grad(current)
        g_norm = np.linalg.norm(g)
        
        if g_norm < 1e-12:
            break
        
        # Project velocity to tangent space
        dot_v = np.sum(velocity * current, axis=1, keepdims=True)
        velocity = velocity - dot_v * current
        
        # Backtracking line search
        step = lr
        found = False
        for bt in range(15):
            v = velocity - step * g
            v_norms = np.linalg.norm(v, axis=1, keepdims=True)
            v_norms_safe = np.where(v_norms < 1e-12, 1.0, v_norms)
            v_unit = v / v_norms_safe
            cos_t = np.cos(v_norms)
            sin_t = np.sin(v_norms)
            new_pts = current * cos_t + v_unit * sin_t
            new_pts = _project(new_pts)
            
            new_e = _energy(new_pts)
            
            if new_e <= e + c1 * step * g_norm * g_norm:
                found = True
                break
            step *= rho
        
        if not found:
            step = max(lr * rho**15, 1e-15)
            v = velocity - step * g
            v_norms = np.linalg.norm(v, axis=1, keepdims=True)
            v_norms_safe = np.where(v_norms < 1e-12, 1.0, v_norms)
            v_unit = v / v_norms_safe
            cos_t = np.cos(v_norms)
            sin_t = np.sin(v_norms)
            new_pts = current * cos_t + v_unit * sin_t
            new_pts = _project(new_pts)
            new_e = _energy(new_pts)
        
        if new_e < best_e - 1e-15:
            best_e = new_e
            best_pts = new_pts.copy()
            current = new_pts
            velocity = beta * velocity - step * g
            no_improve_count = 0
            lr = min(step * 1.5, 0.1)
        elif new_e < best_e:
            current = new_pts
            velocity = beta * velocity - step * g
            no_improve_count += 1
            lr = step
        else:
            current = best_pts.copy()
            velocity *= 0.5
            lr = step * 0.5
            no_improve_count += 1
            if no_improve_count > 50:
                break
    
    return best_pts, best_e


def _lbfgsb_optimize(pts, deadline, max_iter=20000):
    """L-BFGS-B optimization on flattened coordinates with projection."""
    from scipy.optimize import minimize
    
    x0 = pts.flatten().copy()
    best_e = _energy(pts)
    best_x = x0.copy()
    
    def obj(x):
        p = x.reshape(N, 3)
        p = _project(p)
        return _energy(p)
    
    def grad(x):
        p = x.reshape(N, 3)
        p = _project(p)
        e, g = _energy_grad(p)
        return g.flatten()
    
    try:
        res = minimize(obj, x0, jac=grad, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': 1e-16, 'gtol': 1e-12,
                                'maxls': 200, 'maxfun': 100000})
        p = _project(res.x.reshape(N, 3))
        e = _energy(p)
        if e < best_e:
            best_e = e
            best_x = res.x.copy()
    except Exception:
        pass
    
    return _project(best_x.reshape(N, 3)), best_e


def _valid(X):
    if X.shape != (N, 3) or not np.all(np.isfinite(X)):
        return False
    if np.any(np.linalg.norm(X, axis=1) < 1e-12):
        return False
    d2 = np.einsum('ij,ij->i', X, X)[:, None] + np.einsum('ij,ij->i', X, X)[None, :] \
         - 2.0 * (X @ X.T)
    np.fill_diagonal(d2, np.inf)
    if np.min(d2) < (1e-9) ** 2:
        return False
    return True


def _spherical_harmonic_field(pts, deadline, max_iter=5000, lr=0.001):
    """
    Optimize using a smooth potential field approach:
    Model points as minima of a Gaussian-like field on the sphere,
    parameterized by spherical harmonic coefficients.
    This provides a smooth surrogate for the discrete point placement.
    """
    current = pts.copy()
    best_e = _energy(current)
    best_pts = current.copy()
    
    # Use low-order spherical harmonics as a smooth potential
    # We'll use a simple approach: smooth the gradient with a kernel
    kernel_width = 0.3  # angular width of smoothing kernel
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        e, g = _energy_grad(current)
        g_norm = np.linalg.norm(g)
        if g_norm < 1e-12:
            break
        
        # Smooth the gradient by averaging with neighbors within kernel_width
        # This acts as a low-pass filter on the potential field
        smooth_g = np.zeros_like(g)
        for i in range(N):
            # Find neighbors within kernel_width radians
            dots = current @ current[i]
            mask = dots > np.cos(kernel_width)
            if np.sum(mask) > 0:
                smooth_g[i] = g[mask].mean(axis=0)
            else:
                smooth_g[i] = g[i]
        
        # Project smoothed gradient to tangent space
        dot = np.sum(smooth_g * current, axis=1, keepdims=True)
        smooth_g = smooth_g - dot * current
        
        # Take step
        step = lr
        v = -step * smooth_g
        v_norms = np.linalg.norm(v, axis=1, keepdims=True)
        v_norms_safe = np.where(v_norms < 1e-12, 1.0, v_norms)
        v_unit = v / v_norms_safe
        cos_t = np.cos(v_norms)
        sin_t = np.sin(v_norms)
        new_pts = current * cos_t + v_unit * sin_t
        new_pts = _project(new_pts)
        
        new_e = _energy(new_pts)
        
        if new_e < best_e:
            best_e = new_e
            best_pts = new_pts.copy()
            current = new_pts
        else:
            # Reduce learning rate
            lr *= 0.95
            if lr < 1e-10:
                break
    
    return best_pts, best_e


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    
    # Fibonacci spirals with various offsets
    for offset in [0.1, 0.3, 0.5, 0.7, 0.9]:
        initializations.append(_fibonacci_spiral(N, offset=offset))
    
    # Jittered Fibonacci spirals
    base = _fibonacci_spiral(N, offset=0.5)
    for s in range(15):
        jitter_scale = 0.01 + 0.005 * s
        jittered = base + jitter_scale * rng.standard_normal((N, 3))
        initializations.append(_project(jittered))
    
    # Random points with different seeds
    for s in range(15):
        initializations.append(_random_sphere(np.random.default_rng(100 + s)))
    
    # Sobol-like points
    for s in range(8):
        initializations.append(_sobol_like(N, seed=s))
    
    best_pts = None
    best_energy = float('inf')
    
    # Quick evaluation of all initializations
    scored_inits = []
    for init_pts in initializations:
        if time.time() > deadline - 60.0:
            break
        e_quick = _energy(init_pts)
        scored_inits.append((e_quick, init_pts))
    
    scored_inits.sort(key=lambda x: x[0])
    
    # Optimize top candidates with multiple methods
    top_k = min(10, len(scored_inits))
    for idx in range(top_k):
        if time.time() > deadline - 20.0:
            break
        
        e_quick, init_pts = scored_inits[idx]
        
        # Method 1: Gradient descent
        gd_pts, gd_e = _gradient_descent(init_pts, deadline, max_iter=3000, lr_init=0.01)
        if gd_e < best_energy:
            best_energy = gd_e
            best_pts = gd_pts
        
        # Method 2: Momentum gradient descent
        if time.time() < deadline - 15.0:
            mg_pts, mg_e = _momentum_gd(init_pts, deadline, max_iter=2000, lr_init=0.03)
            if mg_e < best_energy:
                best_energy = mg_e
                best_pts = mg_pts
        
        # Method 3: Spherical harmonic field smoothing
        if time.time() < deadline - 10.0:
            sh_pts, sh_e = _spherical_harmonic_field(init_pts, deadline, max_iter=2000, lr=0.005)
            if sh_e < best_energy:
                best_energy = sh_e
                best_pts = sh_pts
    
    # Final polish passes on the global best
    if best_pts is not None:
        # L-BFGS-B polish
        if time.time() < deadline - 15.0:
            lb_pts, lb_e = _lbfgsb_optimize(best_pts, deadline, max_iter=20000)
            if lb_e < best_energy:
                best_energy = lb_e
                best_pts = lb_pts
        
        # Final gradient descent with tiny steps
        if time.time() < deadline - 10.0:
            gd_pts, gd_e = _gradient_descent(best_pts, deadline, max_iter=5000, lr_init=0.005, lr_min=1e-14)
            if gd_e < best_energy:
                best_energy = gd_e
                best_pts = gd_pts
        
        # Another L-BFGS-B pass
        if time.time() < deadline - 8.0:
            lb_pts, lb_e = _lbfgsb_optimize(best_pts, deadline, max_iter=30000)
            if lb_e < best_energy:
                best_energy = lb_e
                best_pts = lb_pts
        
        # Ultra-precise final gradient descent
        if time.time() < deadline - 3.0:
            gd_pts, gd_e = _gradient_descent(best_pts, deadline, max_iter=3000, lr_init=0.001, lr_min=1e-15)
            if gd_e < best_energy:
                best_energy = gd_e
                best_pts = gd_pts
    
    if best_pts is None:
        best_pts = _fibonacci_spiral(N, offset=0.5)
    
    # Final normalization
    best_pts = _project(best_pts)
    
    # Final validation
    final_energy = _precise_energy(best_pts)
    if not np.isfinite(final_energy) or not _valid(best_pts):
        best_pts = _fibonacci_spiral(N, offset=0.5)
        best_pts = _project(best_pts)
    
    return best_pts
