"""Physics-based relaxation packing: Coulomb repulsion + soft wall potential."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
_DEADLINE = 140.0
_start_time = None


def _elapsed():
    return time.time() - _start_time if _start_time is not None else 0.0


def _remaining():
    return _DEADLINE - _elapsed()


def _compute_radius(points):
    """Compute the maximum valid packing radius for given points."""
    n = len(points)
    if n < 2:
        return 0.0
    # Min distance between pairs
    d = np.sqrt(np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=-1))
    np.fill_diagonal(d, np.inf)
    min_pair = d.min()
    # Min distance to boundary
    dists = 1.0 - np.sqrt(np.sum(points ** 2, axis=1))
    min_bound = dists.min()
    return min(min_pair / 2.0, min_bound)


def _energy(points, beta=100.0):
    """
    Compute a smooth energy function that penalizes overlaps and boundary violations.
    We want to minimize this energy.
    """
    n = len(points)
    if n < 2:
        return 0.0
    
    # Pairwise distances
    d = np.sqrt(np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=-1))
    np.fill_diagonal(d, np.inf)
    
    # Soft min of pairwise distances (approximates the minimum distance)
    # We use a log-sum-exp trick for stability
    m = np.min(d)
    soft_min_pair = m - np.log(np.sum(np.exp(-beta * (d - m)))) / beta
    
    # Boundary distances
    dists = 1.0 - np.sqrt(np.sum(points ** 2, axis=1))
    m_b = np.min(dists)
    soft_min_bound = m_b - np.log(np.sum(np.exp(-beta * (dists - m_b)))) / beta
    
    # The radius is min(soft_min_pair / 2, soft_min_bound)
    # We want to maximize this radius, so we minimize the negative.
    # However, the soft min is a lower bound on the true min.
    # A better approach for gradient descent is to use a potential that blows up
    # when constraints are violated.
    
    # Let's try a different energy:
    # E = sum(1/d^2) for pairs + sum(1/(1-r)^2) for boundary
    # This is expensive (O(N^2)) but smooth.
    
    # Actually, let's stick to the soft-min objective but ensure it's differentiable.
    # The previous program used L-BFGS-B on a soft-min objective.
    # Let's try a direct gradient descent on a physics-inspired potential.
    
    # Potential:
    # V_pair = sum_{i<j} 1/d_ij^alpha
    # V_wall = sum_i 1/(1-r_i)^alpha
    # Minimize V_total.
    
    # To avoid division by zero, we clip distances.
    eps = 1e-6
    d_safe = np.maximum(d, eps)
    v_pair = np.sum(1.0 / d_safe**4)
    
    r_safe = np.maximum(1.0 - np.sqrt(np.sum(points ** 2, axis=1)), eps)
    v_wall = np.sum(1.0 / r_safe**4)
    
    return v_pair + v_wall


def _objective(params):
    """Objective function for optimizer: minimize negative radius proxy."""
    pts = params.reshape(-1, 2)
    return _energy(pts)


def _grad(params):
    """
    Compute gradient of the energy function.
    E = sum_{i<j} 1/d_ij^4 + sum_i 1/(1-r_i)^4
    """
    pts = params.reshape(-1, 2)
    n = len(pts)
    grad = np.zeros_like(pts)
    
    # Pairwise gradients
    # dE/dd_ij = -4/d_ij^5
    # dd_ij/dx_i = -(x_i - x_j)/d_ij
    # So dE/dx_i = sum_{j!=i} (-4/d_ij^5) * (-(x_i - x_j)/d_ij) = sum_{j!=i} 4/d_ij^6 * (x_i - x_j)
    
    d = np.sqrt(np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=-1))
    np.fill_diagonal(d, 1.0) # Avoid div by zero, we mask diagonal later
    
    # d^6
    d6 = d**6
    d6_safe = np.maximum(d6, 1e-12)
    
    # Vector difference
    diff = pts[:, None, :] - pts[None, :, :]
    
    # Gradient contribution from pairs
    # grad_i = sum_j 4/d_ij^6 * (x_i - x_j)
    coeff = 4.0 / d6_safe
    np.fill_diagonal(coeff, 0.0)
    
    grad_pair = np.sum(coeff[:, :, None] * diff, axis=1)
    
    # Wall gradients
    # E_wall = sum_i 1/(1-r_i)^4
    # r_i = sqrt(x_i^2 + y_i^2)
    # dE/dr_i = -4/(1-r_i)^5
    # dr_i/dx_i = x_i/r_i
    # dE/dx_i = -4/(1-r_i)^5 * x_i/r_i
    
    r = np.sqrt(np.sum(pts ** 2, axis=1))
    r_safe = np.maximum(r, 1e-12)
    one_minus_r = 1.0 - r
    one_minus_r_safe = np.maximum(one_minus_r, 1e-12)
    
    coeff_wall = -4.0 / one_minus_r_safe**5
    grad_wall = coeff_wall[:, None] * (pts / r_safe[:, None])
    
    return (grad_pair + grad_wall).ravel()


def _relax(points, max_iters=500, step_size=0.01):
    """
    Perform gradient descent to relax the configuration.
    """
    pts = points.copy()
    current_energy = _energy(pts)
    
    for i in range(max_iters):
        if _remaining() < 1.0:
            break
        
        g = _grad(pts.ravel())
        
        # Adaptive step size? Or fixed?
        # Let's try a simple step.
        pts_new = pts - step_size * g.reshape(-1, 2)
        
        new_energy = _energy(pts_new)
        
        if new_energy < current_energy:
            pts = pts_new
            current_energy = new_energy
        else:
            # Reduce step size
            step_size *= 0.5
            if step_size < 1e-8:
                break
                
    return pts


def _refine_continuous(points, max_iters=200):
    """
    Use L-BFGS-B to refine the configuration, maximizing the minimum radius.
    """
    if _remaining() < 3.0:
        return points
    
    n = len(points)
    x0 = points.ravel()
    
    def obj(x):
        pts = x.reshape(n, 2)
        # Use soft min to maximize the minimum radius
        # We minimize the negative of the soft min radius.
        
        d = np.sqrt(np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=-1))
        np.fill_diagonal(d, np.inf)
        
        dists = 1.0 - np.sqrt(np.sum(pts ** 2, axis=1))
        
        # Soft min with high beta for sharpness
        beta = 50.0
        m_p = np.min(d)
        soft_min_pair = m_p - np.log(np.sum(np.exp(-beta * (d - m_p)))) / beta
        m_b = np.min(dists)
        soft_min_bound = m_b - np.log(np.sum(np.exp(-beta * (dists - m_b)))) / beta
        
        r = min(soft_min_pair / 2.0, soft_min_bound)
        return -r
    
    try:
        res = minimize(obj, x0, method='L-BFGS-B', options={'maxiter': max_iters, 'gtol': 1e-9})
        if _remaining() < 0.5:
            return points
        new_pts = res.x.reshape(n, 2)
        if _compute_radius(new_pts) > _compute_radius(points):
            return new_pts
    except Exception:
        pass
    return points


def construct():
    global _start_time
    _start_time = time.time()
    
    best_radius = 0.0
    best_points = None
    
    # Generate multiple initial configurations
    rng = np.random.default_rng(42)
    
    configs = []
    
    # 1. Random points
    for _ in range(5):
        pts = rng.uniform(-1, 1, size=(N, 2))
        # Scale to be inside
        r = np.sqrt(np.sum(pts**2, axis=1))
        max_r = np.max(r)
        if max_r > 0:
            pts = pts * (0.9 / max_r)
        configs.append(pts)
        
    # 2. Hexagonal lattice seed
    spacing = 0.2
    pts = []
    for i in range(-10, 11):
        for j in range(-10, 11):
            x = i * spacing + 0.5 * spacing * (j % 2)
            y = j * spacing * np.sqrt(3) / 2
            if x*x + y*y <= 0.9**2:
                pts.append([x, y])
    if len(pts) >= N:
        # Shuffle and take N
        idx = rng.choice(len(pts), N, replace=False)
        configs.append(np.array(pts)[idx])
        
    # 3. Concentric rings
    radii = np.linspace(0.1, 0.9, 5)
    counts = [1, 6, 12, 18, 13]
    pts = []
    for r, c in zip(radii, counts):
        if c == 1:
            pts.append([r, 0])
        else:
            angles = 2 * np.pi * np.arange(c) / c
            pts.append(np.column_stack([r * np.cos(angles), r * np.sin(angles)]))
    configs.append(np.vstack(pts))
    
    for init_pts in configs:
        if _remaining() < 10.0:
            break
            
        # Relax using physics-based gradient descent
        relaxed_pts = _relax(init_pts, max_iters=200, step_size=0.05)
        
        # Refine using L-BFGS-B
        refined_pts = _refine_continuous(relaxed_pts, max_iters=100)
        
        r = _compute_radius(refined_pts)
        if r > best_radius:
            best_radius = r
            best_points = refined_pts.copy()
            
        # Also try refining the relaxed points directly with a different beta
        # And maybe try a second relaxation pass with smaller step size
        if _remaining() > 20.0:
            fine_relaxed = _relax(relaxed_pts, max_iters=100, step_size=0.005)
            fine_refined = _refine_continuous(fine_relaxed, max_iters=100)
            r2 = _compute_radius(fine_refined)
            if r2 > best_radius:
                best_radius = r2
                best_points = fine_refined.copy()

    if best_points is None or best_radius <= 0:
        # Fallback: simple grid
        spacing = 0.2
        pts = []
        for i in range(-5, 6):
            for j in range(-5, 6):
                pts.append([i*spacing, j*spacing])
        pts = np.array(pts)
        mask = np.sum(pts**2, axis=1) <= 1.0
        pts = pts[mask]
        if len(pts) >= N:
            best_points = pts[:N]
        else:
            # Just return random valid points
            best_points = np.zeros((N, 2))
            for i in range(N):
                while True:
                    p = rng.uniform(-1, 1, size=2)
                    if np.sum(p**2) < 0.8:
                        best_points[i] = p
                        break
            best_points = best_points * 0.5 # Scale down to avoid overlap roughly

    # Final check
    final_r = _compute_radius(best_points)
    if final_r <= 0 or not np.all(np.isfinite(best_points)):
        # Emergency fallback
        spacing = 0.2
        pts = []
        for i in range(-5, 6):
            for j in range(-5, 6):
                pts.append([i*spacing, j*spacing])
        pts = np.array(pts)
        mask = np.sum(pts**2, axis=1) <= 1.0
        pts = pts[mask]
        if len(pts) >= N:
            return pts[:N]
        else:
            # Random
            return rng.uniform(-0.5, 0.5, size=(N, 2))

    return best_points
