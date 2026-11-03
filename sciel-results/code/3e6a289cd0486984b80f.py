"""Minimize logarithmic energy of 150 points on the unit sphere."""
import numpy as np
import time
import math

N = 150
TIME_LIMIT = 110.0  # safety margin before 120s kill


def _log_energy(pts):
    """Compute sum_{i<j} log(1/|x_i - x_j|) using math.fsum."""
    n = len(pts)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-9:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)


def _fibonacci_sphere(n):
    """Generate n points via Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts


def _project(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return pts / norms


def _gradient(pts):
    """Compute gradient of log energy w.r.t. points on sphere (tangential)."""
    n = len(pts)
    grad = np.zeros_like(pts)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = pts[i] - pts[j]
            dist = np.linalg.norm(d)
            if dist < 1e-12:
                continue
            # d/dx_i log(1/|x_i - x_j|) = -1/|x_i-x_j| * (x_i-x_j)/|x_i-x_j| = -d/dist^2
            grad[i] += -d / (dist * dist)
    # Project to tangent space: remove radial component
    for i in range(n):
        radial = np.dot(grad[i], pts[i])
        grad[i] -= radial * pts[i]
    return grad


def _lbfgs_step(pts, grad, m=10):
    """Simple L-BFGS step on the sphere."""
    n = len(pts)
    # Use a simple momentum + adaptive step approach for robustness
    # We'll implement a basic L-BFGS
    pass


def _optimize_lbfgs(pts, max_iters=5000, tol=1e-10):
    """L-BFGS optimization on sphere."""
    n = len(pts)
    m = 10  # memory for L-BFGS

    s_hist = []  # list of displacement vectors
    y_hist = []  # list of gradient differences
    alpha_hist = []
    
    # Flatten for L-BFGS
    x = pts.copy()
    g = _gradient(x)
    
    # L-BFGS two-loop recursion
    def lbfgs_direction(g, x, s_hist, y_hist):
        r = len(s_hist)
        if r == 0:
            return -g
        qs = [None] * r
        beta = [None] * r
        q = g.copy()
        for i in range(r - 1, -1, -1):
            si = s_hist[i]
            yi = y_hist[i]
            alpha_i = np.dot(si, q)
            rho_i = 1.0 / np.dot(yi, si)
            beta[i] = rho_i * alpha_i
            q = q - beta[i] * yi
            qs[i] = q
        # Initial Hessian approximation
        gamma = 1.0
        if r > 0:
            last_s = s_hist[-1]
            last_y = y_hist[-1]
            yys = np.dot(last_y, last_y)
            if yys > 1e-12:
                gamma = np.dot(last_s, last_y) / yys
        r_vec = gamma * q
        for i in range(r):
            si = s_hist[i]
            yi = y_hist[i]
            rho_i = 1.0 / np.dot(yi, si)
            alpha_i = rho_i * np.dot(yi, r_vec)
            r_vec = r_vec + si * (beta[i] - alpha_i)
        return -r_vec
    
    for it in range(max_iters):
        g = _gradient(x)
        # Check convergence
        g_norm = np.linalg.norm(g)
        if g_norm < tol:
            break
        
        # L-BFGS direction
        d = lbfgs_direction(g, x, s_hist, y_hist)
        
        # Project direction to tangent space
        for i in range(n):
            radial = np.dot(d[i], x[i])
            d[i] -= radial * x[i]
        
        # Line search (backtracking)
        step = 1.0
        f_curr = _log_energy(x)
        found = False
        for _ in range(20):
            x_new = _project(x + step * d)
            f_new = _log_energy(x_new)
            if f_new < f_curr - 1e-4 * step * np.dot(d, g):
                found = True
                break
            step *= 0.5
        
        if not found:
            # Try smaller step
            step = 1e-4
            x_new = _project(x + step * d)
            f_new = _log_energy(x_new)
            if f_new >= f_curr:
                break
        
        # Update history
        s = x_new - x
        g_new = _gradient(x_new)
        y = g_new - g
        
        # Curvature condition
        if np.dot(s, y) > 1e-10:
            s_hist.append(s)
            y_hist.append(y)
            if len(s_hist) > m:
                s_hist.pop(0)
                y_hist.pop(0)
        
        x = x_new
    
    return x


def _optimize_momentum(pts, max_iters=10000, lr_init=0.01):
    """Gradient descent with momentum on sphere."""
    x = pts.copy()
    v = np.zeros_like(x)
    lr = lr_init
    best_f = _log_energy(x)
    best_x = x.copy()
    
    for it in range(max_iters):
        g = _gradient(x)
        v = 0.9 * v - lr * g
        # Project velocity to tangent space
        for i in range(len(x)):
            radial = np.dot(v[i], x[i])
            v[i] -= radial * x[i]
        
        x_new = _project(x + v)
        f_new = _log_energy(x_new)
        
        if f_new < best_f:
            best_f = f_new
            best_x = x_new.copy()
        
        # Adaptive learning rate
        if f_new > best_f + 0.1:
            lr *= 0.5
        elif it % 100 == 0 and lr < 0.001:
            lr *= 1.1
        
        x = x_new
    
    return best_x


def _restart_optimize(pts, seed, max_iters):
    """Run optimization with a given seed."""
    np.random.seed(seed)
    # Add small perturbation
    noise = np.random.randn(*pts.shape) * 0.01
    x = _project(pts + noise)
    
    # Try momentum first (faster), then L-BFGS if time allows
    x = _optimize_momentum(x, max_iters=max_iters)
    return x


def construct():
    start_time = time.time()
    
    # Initial Fibonacci spiral
    pts0 = _fibonacci_sphere(N)
    best_pts = pts0.copy()
    best_energy = _log_energy(pts0)
    
    # Phase 1: Optimize the initial configuration
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT - 30:
        x = _optimize_momentum(pts0, max_iters=5000)
        f = _log_energy(x)
        if f < best_energy:
            best_energy = f
            best_pts = x.copy()
    
    # Phase 2: Multiple restarts with perturbations
    np.random.seed(42)
    for restart in range(20):
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT - 15:
            break
        
        # Create perturbed starting point
        noise_scale = 0.1 * (1 + restart * 0.1)
        noise = np.random.randn(N, 3) * noise_scale
        x_start = _project(best_pts + noise)
        
        # Optimize
        x = _optimize_momentum(x_start, max_iters=3000)
        f = _log_energy(x)
        
        if f < best_energy:
            best_energy = f
            best_pts = x.copy()
    
    # Phase 3: Final polish with L-BFGS if time allows
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT - 10:
        x = _optimize_lbfgs(best_pts, max_iters=2000)
        f = _log_energy(x)
        if f < best_energy:
            best_energy = f
            best_pts = x.copy()
    
    # Final check
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT - 5:
        # One more momentum polish
        x = _optimize_momentum(best_pts, max_iters=2000, lr_init=0.001)
        f = _log_energy(x)
        if f < best_energy:
            best_energy = f
            best_pts = x.copy()
    
    # Ensure valid output
    best_pts = _project(best_pts)
    
    # Verify no points too close
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                # Nudge apart
                best_pts[j] += 1e-8 * np.random.randn(3)
                best_pts = _project(best_pts)
    
    return best_pts
