"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint
from scipy.spatial import Voronoi, qhull
from scipy.spatial.distance import cdist

N = 26
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0


def _voronoi_initialization(rng, n):
    """Generate initial circle positions using Voronoi-based approach."""
    # Start with a grid perturbation
    side = int(np.ceil(np.sqrt(n)))
    xs = np.array([(i + 0.5) / side for i in range(side)] * side)
    ys = np.array([(j + 0.5) / side for j in range(side) for i in range(side)])
    pts = np.column_stack([xs, ys])
    # Perturb
    pts += rng.uniform(-0.1 / side, 0.1 / side, size=pts.shape)
    pts = np.clip(pts, 0.01, 0.99)
    
    # Compute Voronoi to get better radii estimates
    try:
        vor = Voronoi(pts[:n])
        radii = []
        for i in range(n):
            # Find closest vertex to point i
            region = vor.regions[vor.point_region[i]]
            if len(region) == 0:
                radii.append(0.5 / side)
                continue
            verts = vor.vertices[region]
            if len(verts) == 0:
                radii.append(0.5 / side)
                continue
            # Distance to nearest vertex
            d = np.min(np.linalg.norm(verts - pts[i], axis=1))
            radii.append(min(d, 0.5 / side) * 0.95)
        radii = np.array(radii)
    except Exception:
        radii = np.full(n, 0.5 / side - 0.01)
    
    return pts[:n].copy(), radii


def _soft_plus(x, beta=50.0):
    """Smooth approximation of max(0, x) using log-sum-exp."""
    # softplus(x) = log(1 + exp(x)) / beta approximates max(0, x)
    # For large beta, this is close to max(0, x)
    # Use stable computation
    ax = np.abs(x)
    return np.where(x > 0, x + np.log1p(np.exp(-x)), np.log1p(np.exp(x))) / beta


def _cost_and_grad(theta, beta=50.0, mu=100.0):
    """
    Objective: minimize -sum(radii) + penalty for constraint violations.
    
    theta = [x_0, ..., x_{n-1}, y_0, ..., y_{n-1}, r_0, ..., r_{n-1}]
    
    Constraints:
    1. x_i >= r_i, y_i >= r_i, x_i <= 1-r_i, y_i <= 1-r_i (inside square)
    2. (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2 (non-overlapping)
    3. r_i > 0
    """
    n = N
    x = theta[:n]
    y = theta[n:2*n]
    r = theta[2*n:3*n]
    
    # Objective: maximize sum(r) => minimize -sum(r)
    obj = -np.sum(r)
    grad_obj = np.zeros(3*n)
    grad_obj[2*n:3*n] = -1.0
    
    # Penalty terms
    penalty = 0.0
    grad_pen = np.zeros(3*n)
    
    # Boundary constraints: x_i >= r_i => x_i - r_i >= 0
    # Violation: max(0, r_i - x_i)
    viol_x_left = np.maximum(0, r - x)
    viol_x_right = np.maximum(0, r - (1 - x))
    viol_y_bottom = np.maximum(0, r - y)
    viol_y_top = np.maximum(0, r - (1 - y))
    
    # Soft penalties for boundaries
    sp_xl = _soft_plus(r - x, beta)
    sp_xr = _soft_plus(r - (1 - x), beta)
    sp_yb = _soft_plus(r - y, beta)
    sp_yt = _soft_plus(r - (1 - y), beta)
    
    penalty += mu * (np.sum(sp_xl) + np.sum(sp_xr) + np.sum(sp_yb) + np.sum(sp_yt))
    
    # Gradients of softplus: sigmoid-like
    # d/dz softplus(z) = sigmoid(z) where z is the argument
    sig_xl = 1.0 / (1.0 + np.exp(-beta * (r - x)))
    sig_xr = 1.0 / (1.0 + np.exp(-beta * (r - (1 - x))))
    sig_yb = 1.0 / (1.0 + np.exp(-beta * (r - y)))
    sig_yt = 1.0 / (1.0 + np.exp(-beta * (r - (1 - y))))
    
    grad_pen[:n] += mu * (-sig_xl + sig_xr)  # d/dx
    grad_pen[n:2*n] += mu * (-sig_yb + sig_yt)  # d/dy
    grad_pen[2*n:3*n] += mu * (sig_xl + sig_xr + sig_yb + sig_yt)  # d/dr
    
    # Pairwise overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d2 = dx * dx + dy * dy
            d = np.sqrt(d2 + 1e-12)
            rsum = r[i] + r[j]
            # Violation: max(0, rsum - d)
            z = rsum - d
            sp = _soft_plus(z, beta)
            penalty += mu * sp
            
            # Gradient: d/dz softplus = sigmoid(beta*z)
            sig = 1.0 / (1.0 + np.exp(-beta * z))
            
            # d/dx_i = -sig * dx/d, d/dx_j = sig * dx/d
            # d/dy_i = -sig * dy/d, d/dy_j = sig * dy/d
            # d/dr_i = sig, d/dr_j = sig
            inv_d = 1.0 / d
            grad_pen[i] += mu * (-sig * dx * inv_d)
            grad_pen[j] += mu * (sig * dx * inv_d)
            grad_pen[n + i] += mu * (-sig * dy * inv_d)
            grad_pen[n + j] += mu * (sig * dy * inv_d)
            grad_pen[2*n + i] += mu * sig
            grad_pen[2*n + j] += mu * sig
    
    total = obj + penalty
    grad = grad_obj + grad_pen
    return total, grad


def _is_valid(x, y, r, tol=1e-7):
    """Check if the packing is valid (disjoint interiors, inside square)."""
    n = len(r)
    # Check boundaries
    if np.any(x < -tol) or np.any(y < -tol) or np.any(x > 1 + tol) or np.any(y > 1 + tol):
        return False
    if np.any(r < -tol):
        return False
    # Check circles stay in square
    if np.any(x - r < -tol) or np.any(y - r < -tol):
        return False
    if np.any(x + r > 1 + tol) or np.any(y + r > 1 + tol):
        return False
    # Check pairwise distances
    for i in range(n):
        for j in range(i + 1, n):
            d2 = (x[i] - x[j])**2 + (y[i] - y[j])**2
            rsum = r[i] + r[j]
            if d2 < (rsum - tol)**2:
                return False
    return True


def _project_to_feasible(x, y, r, max_iter=100):
    """Simple projection to make configuration feasible."""
    x = x.copy()
    y = y.copy()
    r = r.copy()
    
    for _ in range(max_iter):
        # Clamp to square
        for i in range(N):
            r[i] = max(r[i], 1e-6)
            r[i] = min(r[i], 0.5)
            x[i] = np.clip(x[i], r[i], 1 - r[i])
            y[i] = np.clip(y[i], r[i], 1 - r[i])
        
        # Resolve overlaps by shrinking radii
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                rsum = r[i] + r[j]
                if d < rsum:
                    # Shrink the smaller radius
                    overlap = rsum - d
                    if r[i] < r[j]:
                        r[i] = max(r[i] - overlap * 0.5, 1e-6)
                    else:
                        r[j] = max(r[j] - overlap * 0.5, 1e-6)
                    changed = True
        if not changed:
            break
    
    return x, y, r


def _local_polish(x, y, r, time_budget=5.0):
    """High-precision polish using L-BFGS-B with tight tolerance."""
    theta = np.concatenate([x, y, r])
    
    def obj_fn(t):
        val, _ = _cost_and_grad(t, beta=200.0, mu=500.0)
        return val
    
    def grad_fn(t):
        _, g = _cost_and_grad(t, beta=200.0, mu=500.0)
        return g
    
    bounds = []
    for i in range(N):
        bounds.append((1e-6, 0.5))  # x
    for i in range(N):
        bounds.append((1e-6, 0.5))  # y
    for i in range(N):
        bounds.append((1e-6, 0.5))  # r
    
    start = time.time()
    try:
        res = minimize(obj_fn, theta, jac=grad_fn, method='L-BFGS-B',
                       bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-12,
                                               'gtol': 1e-10, 'maxls': 50})
        if res.success or res.fun < obj_fn(theta):
            theta_new = res.x
            x_new = theta_new[:N]
            y_new = theta_new[N:2*N]
            r_new = theta_new[2*N:3*N]
            x_new, y_new, r_new = _project_to_feasible(x_new, y_new, r_new)
            if _is_valid(x_new, y_new, r_new):
                return x_new, y_new, r_new
    except Exception:
        pass
    
    return x, y, r


def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    best_score = -np.inf
    best_result = None
    
    # Initial grid solution
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            cells.append(((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3))
    best_result = np.array(cells[:N])
    best_score = np.sum(best_result[:, 2])
    
    # Multi-start optimization
    n_starts = 0
    max_starts = 50
    
    for start_idx in range(max_starts):
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT - SAFETY_MARGIN - 10.0:
            break
        
        # Generate initial configuration
        if start_idx == 0:
            # Use Voronoi-based initialization
            x0, r0 = _voronoi_initialization(rng, N)
            y0 = x0.copy()  # Start with same x and y, will be adjusted
            # Actually, let's use the grid as base
            x0 = best_result[:, 0].copy()
            y0 = best_result[:, 1].copy()
            r0 = best_result[:, 2].copy()
        else:
            # Random perturbation of best or random
            if start_idx % 2 == 0 and best_result is not None:
                x0 = best_result[:, 0] + rng.normal(0, 0.05, N)
                y0 = best_result[:, 1] + rng.normal(0, 0.05, N)
                r0 = best_result[:, 2] + rng.normal(0, 0.01, N)
            else:
                x0 = rng.uniform(0.1, 0.9, N)
                y0 = rng.uniform(0.1, 0.9, N)
                r0 = rng.uniform(0.02, 0.12, N)
            
            x0 = np.clip(x0, 0.01, 0.99)
            y0 = np.clip(y0, 0.01, 0.99)
            r0 = np.clip(r0, 0.01, 0.3)
        
        # Make feasible
        x0, y0, r0 = _project_to_feasible(x0, y0, r0)
        
        if not _is_valid(x0, y0, r0):
            continue
        
        theta0 = np.concatenate([x0, y0, r0])
        
        # Optimization with increasing beta
        current_x, current_y, current_r = x0.copy(), y0.copy(), r0.copy()
        
        for beta in [20.0, 50.0, 100.0, 200.0]:
            elapsed = time.time() - start_time
            if elapsed > TIME_LIMIT - SAFETY_MARGIN - 5.0:
                break
            
            theta = np.concatenate([current_x, current_y, current_r])
            
            def obj_fn(t, b=beta):
                val, _ = _cost_and_grad(t, beta=b, mu=100.0)
                return val
            
            def grad_fn(t, b=beta):
                _, g = _cost_and_grad(t, beta=b, mu=100.0)
                return g
            
            bounds = []
            for i in range(N):
                bounds.append((1e-6, 0.5))
            for i in range(N):
                bounds.append((1e-6, 0.5))
            for i in range(N):
                bounds.append((1e-6, 0.5))
            
            try:
                res = minimize(obj_fn, theta, jac=grad_fn, method='L-BFGS-B',
                               bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-10,
                                                       'gtol': 1e-8, 'maxls': 40})
                if res.fun < obj_fn(theta):
                    theta_new = res.x
                    current_x = theta_new[:N]
                    current_y = theta_new[N:2*N]
                    current_r = theta_new[2*N:3*N]
                    current_x, current_y, current_r = _project_to_feasible(current_x, current_y, current_r)
            except Exception:
                pass
        
        # Final polish
        elapsed = time.time() - start_time
        if elapsed < TIME_LIMIT - SAFETY_MARGIN - 3.0:
            current_x, current_y, current_r = _local_polish(
                current_x, current_y, current_r,
                time_budget=min(5.0, TIME_LIMIT - SAFETY_MARGIN - time.time() + start_time - 3.0))
        
        if _is_valid(current_x, current_y, current_r):
            score = np.sum(current_r)
            if score > best_score:
                best_score = score
                best_result = np.column_stack([current_x, current_y, current_r])
        
        n_starts += 1
    
    # Final deadline check
    elapsed = time.time() - start_time
    if elapsed > TIME_LIMIT - SAFETY_MARGIN:
        # Return best result immediately
        return best_result
    
    # If we have time, do one more polish on the best
    if best_result is not None and elapsed < TIME_LIMIT - SAFETY_MARGIN - 2.0:
        bx, by, br = best_result[:, 0], best_result[:, 1], best_result[:, 2]
        bx, by, br = _local_polish(bx, by, br, time_budget=min(3.0, TIME_LIMIT - SAFETY_MARGIN - (time.time() - start_time) - 1.0))
        if _is_valid(bx, by, br) and np.sum(br) > best_score:
            best_result = np.column_stack([bx, by, br])
            best_score = np.sum(br)
    
    return best_result
