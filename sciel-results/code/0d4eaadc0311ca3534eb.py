"""Packing 200 equal circles in a unit square using Delaunay-based optimization."""
import numpy as np
import time
from scipy.spatial import Delaunay, cKDTree
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

N = 200


def _compute_radius(c):
    """Compute the largest common radius for given centres."""
    n = c.shape[0]
    if n == 0:
        return 0.0
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    if n < 2:
        return min_boundary
    dists = pdist(c, metric='euclidean')
    min_pair_dist = np.min(dists)
    return min(min_pair_dist / 2.0, min_boundary)


def _get_delaunay_edges(c):
    """Get edges from Delaunay triangulation."""
    try:
        tri = Delaunay(c)
        edges = set()
        for simplex in tri.simplices:
            for i in range(len(simplex)):
                for j in range(i + 1, len(simplex)):
                    a, b = simplex[i], simplex[j]
                    edges.add((min(a, b), max(a, b)))
        return list(edges)
    except Exception:
        # Fallback: use KDTree nearest neighbors
        tree = cKDTree(c)
        edges = set()
        for i in range(N):
            _, nn = tree.query(c[i], k=8)
            for j in nn:
                if j != i:
                    edges.add((min(i, j), max(i, j)))
        return list(edges)


def _soft_min_energy_and_grad(c, temperature=0.01):
    """
    Compute soft-min energy and its gradient.
    Energy = -soft_min(min_pair_dist/2, min_boundary)
    """
    c = np.clip(c, -0.4999, 0.4999)
    n = c.shape[0]
    
    # Boundary distances
    boundary_dists = 0.5 - np.abs(c)
    
    # Pairwise distances - use Delaunay edges for efficiency
    edges = _get_delaunay_edges(c)
    if not edges:
        # Fallback to all pairs if Delaunay fails
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    pair_dists = np.zeros(len(edges))
    for k, (i, j) in enumerate(edges):
        diff = c[i] - c[j]
        pair_dists[k] = np.linalg.norm(diff)
    
    # Soft min of pair distances / 2
    T = temperature
    p = 20
    
    # For pairs
    mask_p = pair_dists < 0.15
    if np.any(mask_p):
        soft_min_pair = np.sum(pair_dists[mask_p] ** p) ** (-1.0 / p)
    else:
        soft_min_pair = np.min(pair_dists) if len(pair_dists) > 0 else 1.0
    
    # For boundary
    mask_b = boundary_dists < 0.15
    if np.any(mask_b):
        soft_min_bound = np.sum(boundary_dists[mask_b] ** p) ** (-1.0 / p)
    else:
        soft_min_bound = np.min(boundary_dists)
    
    r_pair = soft_min_pair / 2.0
    r_bound = soft_min_bound
    
    # Combined soft min
    log_soft = T * np.logaddexp(-r_pair / T, -r_bound / T)
    energy = -log_soft
    
    # Gradient computation
    grad = np.zeros((n, 2))
    
    # Weight for pair vs boundary
    w_pair = np.exp(-r_pair / T) / (np.exp(-r_pair / T) + np.exp(-r_bound / T))
    w_bound = 1.0 - w_pair
    
    # Gradient from pair distances
    if np.any(mask_p):
        weights_p = (pair_dists[mask_p] ** (p - 1)) / np.sum(pair_dists[mask_p] ** p)
        for k, (i, j) in enumerate(edges):
            if not mask_p[k]:
                continue
            diff = c[i] - c[j]
            dist = pair_dists[k]
            if dist < 1e-10:
                continue
            direction = diff / dist
            w = w_pair * weights_p[k] / (2.0 * p)
            grad[i] += w * direction
            grad[j] -= w * direction
    
    # Gradient from boundary
    if np.any(mask_b):
        weights_b = (boundary_dists[mask_b] ** (p - 1)) / np.sum(boundary_dists[mask_b] ** p)
        flat_idx = 0
        for i in range(n):
            for axis in range(2):
                if mask_b[i, axis]:
                    w = w_bound * weights_b[flat_idx] / p
                    if c[i, axis] >= 0:
                        grad[i, axis] -= w
                    else:
                        grad[i, axis] += w
                flat_idx += 1
    
    return energy, grad


def _optimize_delaunay(c0, max_time=30.0):
    """
    Gradient-based optimization using Delaunay edges for efficiency.
    Uses custom gradient descent with adaptive step size.
    """
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    c_current = c0.copy()
    
    # Parameters
    lr = 0.005
    lr_decay = 0.999
    momentum = 0.9
    velocity = np.zeros_like(c_current)
    patience = 0
    max_patience = 500
    
    while time.time() < start_time + max_time - 2:
        energy, grad = _soft_min_energy_and_grad(c_current, temperature=0.01)
        
        # Momentum update
        velocity = momentum * velocity - lr * grad
        c_trial = c_current + velocity
        c_trial = np.clip(c_trial, -0.4999, 0.4999)
        
        r_trial = _compute_radius(c_trial)
        
        if r_trial > best_r:
            best_r = r_trial
            best_c = c_trial.copy()
            patience = 0
        else:
            patience += 1
            if patience > max_patience:
                break
        
        c_current = c_trial
        lr *= lr_decay
        
        if lr < 1e-7:
            break
    
    return best_c, best_r


def _high_precision_polish(c0, max_time=20.0):
    """
    Final high-precision polish using L-BFGS-B with very tight tolerances.
    """
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    def obj_and_grad(params):
        c = params.reshape(-1, 2)
        energy, grad = _soft_min_energy_and_grad(c, temperature=0.005)
        return energy, grad.ravel()
    
    try:
        result = minimize(
            obj_and_grad,
            best_c.flatten(),
            method='L-BFGS-B',
            jac=True,
            bounds=[(-0.4999, 0.4999)] * (2 * N),
            options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 50}
        )
        c_new = np.clip(result.x.reshape(-1, 2), -0.4999, 0.4999)
        r_new = _compute_radius(c_new)
        if r_new > best_r:
            best_r = r_new
            best_c = c_new.copy()
    except Exception:
        pass
    
    # Second pass with even tighter temperature
    if time.time() < start_time + max_time - 3:
        def obj_and_grad2(params):
            c = params.reshape(-1, 2)
            energy, grad = _soft_min_energy_and_grad(c, temperature=0.002)
            return energy, grad.ravel()
        
        try:
            result = minimize(
                obj_and_grad2,
                best_c.flatten(),
                method='L-BFGS-B',
                jac=True,
                bounds=[(-0.4999, 0.4999)] * (2 * N),
                options={'maxiter': 300, 'ftol': 1e-16, 'gtol': 1e-13, 'maxls': 50}
            )
            c_new = np.clip(result.x.reshape(-1, 2), -0.4999, 0.4999)
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_r = r_new
                best_c = c_new.copy()
        except Exception:
            pass
    
    return best_c, best_r


def _initialize_hexagonal():
    """Initialize with a hexagonal pattern."""
    best_c = None
    best_r = 0.0
    
    for d in np.linspace(0.055, 0.075, 30):
        points = []
        row_spacing = d * np.sqrt(3) / 2
        x_offset = d / 2
        
        y = -0.5 + d / 2
        row = 0
        while y <= 0.5 - d / 2 + 1e-9:
            x = -0.5 + d / 2
            if row % 2 == 1:
                x += x_offset
            while x <= 0.5 - d / 2 + 1e-9:
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    points.append([x, y])
                x += d
            y += row_spacing
            row += 1
        
        if len(points) >= N:
            c = np.array(points[:N])
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
        elif len(points) > 0 and best_c is None:
            c = np.array(points)
            if len(c) < N:
                extra = np.random.uniform(-0.45, 0.45, size=(N - len(c), 2))
                c = np.vstack([c, extra])
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
    
    if best_c is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_c = grid[:N].copy()
    
    return best_c


def _jittered_grid_init(seed=0):
    """Create a jittered grid initialization."""
    rng = np.random.RandomState(seed)
    m = int(np.ceil(np.sqrt(N)))
    grid_x, grid_y = np.meshgrid(
        np.linspace(-0.45, 0.45, m),
        np.linspace(-0.45, 0.45, m)
    )
    grid = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    rng.shuffle(grid)
    c = grid[:N].copy()
    c += rng.normal(0, 0.01, size=(N, 2))
    c = np.clip(c, -0.4999, 0.4999)
    return c


def _lbfgs_optimize(c0, max_time=20.0):
    """Local optimization using L-BFGS-B with soft-min energy."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    temperatures = [0.01, 0.015, 0.02, 0.03]
    
    for T in temperatures:
        if time.time() > start_time + max_time - 2:
            break
        
        def obj(params):
            c = params.reshape(-1, 2)
            c = np.clip(c, -0.4999, 0.4999)
            
            boundary_dists = 0.5 - np.abs(c)
            edges = _get_delaunay_edges(c)
            if not edges:
                edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
            pair_dists = np.array([np.linalg.norm(c[i] - c[j]) for i, j in edges])
            
            p = 30
            mask = pair_dists < 0.15
            if np.any(mask):
                soft_min_dist = np.sum(pair_dists[mask] ** p) ** (-1.0 / p)
            else:
                soft_min_dist = np.min(pair_dists)
            
            mask_b = boundary_dists < 0.15
            if np.any(mask_b):
                soft_boundary = np.sum(boundary_dists[mask_b] ** p) ** (-1.0 / p)
            else:
                soft_boundary = np.min(boundary_dists)
            
            r_pair = soft_min_dist / 2.0
            r_bound = soft_boundary
            
            soft_min_combined = -T * np.logaddexp(-r_pair / T, -r_bound / T)
            return -soft_min_combined
        
        try:
            result = minimize(
                obj,
                c0.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.4999, 0.4999)] * (2 * N),
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            
            c_new = np.clip(result.x.reshape(-1, 2), -0.4999, 0.4999)
            r_new = _compute_radius(c_new)
            
            if r_new > best_r:
                best_c = c_new
                best_r = r_new
                c0 = c_new.copy()
        except Exception:
            continue
    
    return best_c, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_radius = 0.0
    best_c = None
    
    # Step 1: Hexagonal initialization
    c_hex = _initialize_hexagonal()
    r_hex = _compute_radius(c_hex)
    best_radius = r_hex
    best_c = c_hex.copy()
    
    # Step 2: Delaunay-based gradient optimization
    if time.time() < deadline - 40:
        max_opt_time = min(30.0, deadline - time.time() - 10)
        c_opt, r_opt = _optimize_delaunay(best_c, max_time=max_opt_time)
        if r_opt > best_radius:
            best_radius = r_opt
            best_c = c_opt.copy()
    
    # Step 3: L-BFGS optimization
    if time.time() < deadline - 30:
        remaining = deadline - time.time() - 10
        if remaining > 10:
            c_cp, r_cp = _lbfgs_optimize(best_c, max_time=min(25.0, remaining))
            if r_cp > best_radius:
                best_radius = r_cp
                best_c = c_cp.copy()
    
    # Step 4: Multiple random restarts
    n_restarts = 3
    for trial in range(n_restarts):
        if time.time() > deadline - 25:
            break
        
        c_init = _jittered_grid_init(seed=42 + trial)
        r_init = _compute_radius(c_init)
        if r_init > best_radius:
            best_radius = r_init
            best_c = c_init.copy()
        
        remaining = deadline - time.time() - 10
        if remaining > 8:
            max_opt_time = min(12.0, remaining)
            c_opt, r_opt = _optimize_delaunay(c_init, max_time=max_opt_time)
            if r_opt > best_radius:
                best_radius = r_opt
                best_c = c_opt.copy()
    
    # Step 5: High precision polish
    if time.time() < deadline - 10:
        remaining = deadline - time.time() - 3
        if remaining > 8:
            c_pol, r_pol = _high_precision_polish(best_c, max_time=min(15.0, remaining))
            if r_pol > best_radius:
                best_radius = r_pol
                best_c = c_pol.copy()
    
    # Final clip to ensure validity
    best_c = np.clip(best_c, -0.5, 0.5)
    
    # Verify the result is valid
    if best_c.shape != (N, 2) or not np.all(np.isfinite(best_c)):
        best_c = _initialize_hexagonal()
        best_c = np.clip(best_c, -0.5, 0.5)
    
    return best_c
