import numpy as np
from scipy.optimize import minimize
import time

N = 300
_DEADLINE = time.time() + 145.0  # Safety margin before 150s kill

def _hex_lattice_init(n_points, rotation=0.0, seed=42):
    """Generate a hexagonal lattice of points inside the unit circle."""
    rng = np.random.RandomState(seed)
    s = np.sqrt(2 * np.pi / (n_points * np.sqrt(3)))
    # Slightly shrink spacing to ensure we can fit n points easily
    s *= 0.98
    k = int(1.0 / s) + 2
    
    points = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3) / 2
            if np.hypot(x, y) <= 1.0:
                points.append([x, y])
                
    points = np.array(points)
    
    if rotation != 0.0:
        cos_r = np.cos(rotation)
        sin_r = np.sin(rotation)
        R = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
        points = points @ R.T
    
    dists = np.linalg.norm(points, axis=1)
    indices = np.argsort(dists)
    points = points[indices[:n_points]]
    
    # Add small random noise to break symmetry
    points += rng.normal(0, s * 0.02, size=points.shape)
    
    return points

def _radial_ring_init(n_points, seed=42):
    """Generate points on concentric rings."""
    rng = np.random.RandomState(seed)
    points = []
    s = np.sqrt(2 * np.pi / (n_points * np.sqrt(3)))
    n_rings = int(1.0 / s) + 1
    
    for r_idx in range(n_rings):
        r = (r_idx + 0.5) * s
        if r >= 1.0:
            break
        n_on_ring = max(1, int(2 * np.pi * r / s))
        angles = np.linspace(0, 2 * np.pi, n_on_ring, endpoint=False)
        for a in angles:
            x = r * np.cos(a)
            y = r * np.sin(a)
            if np.hypot(x, y) < 1.0:
                points.append([x, y])
    
    points = np.array(points)
    if len(points) < n_points:
        while len(points) < n_points:
            r = np.sqrt(rng.random())
            a = rng.random() * 2 * np.pi
            points = np.vstack([points, [[r * np.cos(a), r * np.sin(a)]]])
    elif len(points) > n_points:
        dists = np.linalg.norm(points, axis=1)
        indices = np.argsort(dists)
        points = points[indices[:n_points]]
    
    points += rng.normal(0, s * 0.02, size=points.shape)
    return points

def _compute_true_score(points):
    """Compute the true packing radius."""
    D = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    L = np.linalg.norm(D, axis=2)
    
    np.fill_diagonal(L, np.inf)
    min_pair = np.min(L) / 2.0
    
    L_wall = np.linalg.norm(points, axis=1)
    min_wall = 1.0 - np.max(L_wall)
    
    return min(min_pair, min_wall)

def _soft_min_obj_and_grad(x, beta):
    """
    Compute the negative soft-min value and its gradient.
    """
    C = x.reshape(N, 2)
    
    D = C[:, np.newaxis, :] - C[np.newaxis, :, :]
    L = np.linalg.norm(D, axis=2)
    
    L_safe = np.where(L > 1e-12, L, 1.0)
    x_pairs = L_safe / 2.0
    
    L_wall = np.linalg.norm(C, axis=1)
    x_walls = 1.0 - L_wall
    
    i_idx, j_idx = np.triu_indices(N, k=1)
    x_pairs_flat = x_pairs[i_idx, j_idx]
    
    x_all = np.concatenate([x_pairs_flat, x_walls])
    
    # Log-sum-exp trick for stability
    min_x = np.min(x_all)
    shifted_x = x_all - min_x
    exp_terms = np.exp(-beta * shifted_x)
    Z = np.sum(exp_terms)
    P = exp_terms / Z
    
    val = min_x - (1.0 / beta) * np.log(Z)
    
    grad = np.zeros_like(C)
    
    P_pairs = P[:len(i_idx)]
    
    D_flat = D[i_idx, j_idx]
    L_flat = L[i_idx, j_idx]
    L_flat_safe = np.where(L_flat > 1e-12, L_flat, 1.0)
    
    U = D_flat / L_flat_safe[:, np.newaxis]
    
    W = P_pairs * 0.5
    
    np.add.at(grad, i_idx, W[:, np.newaxis] * U)
    np.add.at(grad, j_idx, -W[:, np.newaxis] * U)
    
    P_walls = P[len(i_idx):]
    
    L_wall_safe = np.where(L_wall > 1e-12, L_wall, 1.0)
    U_walls = -C / L_wall_safe[:, np.newaxis]
    
    grad += P_walls[:, np.newaxis] * U_walls
    
    return -val, -grad

def _optimize_lbfgs(points, beta, max_iter=300, time_limit=10.0):
    """Run L-BFGS-B optimization with a time limit."""
    start_time = time.time()
    
    def obj_and_grad(x):
        if time.time() - start_time > time_limit:
            raise TimeoutError()
        return _soft_min_obj_and_grad(x, beta)
    
    try:
        res = minimize(obj_and_grad, points.flatten(), method='L-BFGS-B', jac=True,
                       options={'maxiter': max_iter, 'gtol': 1e-7, 'ftol': 1e-13})
        if res.success or res.status in [1, 2, 4]:
            return res.x.reshape(N, 2)
    except Exception:
        pass
    return points

def _exact_subgradient_polish(points, max_iters=200, tol=1e-8, time_limit=5.0):
    """
    Local optimization using exact subgradient of the minimum constraint.
    This is a structural change: instead of smooth approximation, we use the 
    subgradient of the active constraints to move the points.
    """
    pts = points.copy()
    n = len(pts)
    start_time = time.time()
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit or time.time() > _DEADLINE - 2:
            break
        
        # Compute all constraint values
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        radii = np.sqrt(np.sum(pts**2, axis=1))
        boundary_dists = 1.0 - radii
        
        # Pair constraints: dists/2
        ii, jj = np.triu_indices(n, k=1)
        pair_vals = dists[ii, jj] / 2.0
        bound_vals = boundary_dists
        
        all_vals = np.concatenate([pair_vals, bound_vals])
        min_val = np.min(all_vals)
        
        if min_val < tol:
            break
        
        # Find active constraints (within tolerance of minimum)
        # Use a small tolerance to define the active set
        active_mask = np.abs(all_vals - min_val) < 1e-5
        active_pairs = np.where(active_mask[:len(ii)])[0]
        active_bounds = np.where(active_mask[len(ii):])[0]
        
        if len(active_pairs) == 0 and len(active_bounds) == 0:
            break
        
        # Build subgradient for the minimum constraint
        # The subgradient of min(v) is the average of gradients of active v's
        grad = np.zeros_like(pts)
        
        # Pairwise active constraints
        if len(active_pairs) > 0:
            for idx in active_pairs:
                i, j = ii[idx], jj[idx]
                d = dists[i, j]
                if d < 1e-10:
                    continue
                # grad_i = (pts[i] - pts[j]) / (2*d)
                # grad_j = (pts[j] - pts[i]) / (2*d)
                direction = (pts[i] - pts[j]) / d
                grad[i] += direction / (2.0 * len(active_pairs))
                grad[j] -= direction / (2.0 * len(active_pairs))
        
        # Boundary active constraints
        if len(active_bounds) > 0:
            for idx in active_bounds:
                i = active_bounds[idx]
                r = radii[i]
                if r < 1e-10:
                    continue
                # grad_i = -pts[i] / r
                direction = -pts[i] / r
                grad[i] += direction / (2.0 * len(active_bounds))
        
        # Normalize gradient
        grad_norm = np.sqrt(np.sum(grad**2))
        if grad_norm < 1e-10:
            break
        grad = grad / grad_norm
        
        # Line search for step size
        best_step = 0.0
        best_obj = min_val
        # Try a range of step sizes
        step_sizes = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.05]
        
        for step in step_sizes:
            new_pts = pts + step * grad
            # Ensure points stay inside the circle during line search
            new_radii = np.linalg.norm(new_pts, axis=1)
            if np.any(new_radii > 1.0):
                # Scale back points that are outside
                new_pts[new_radii > 1.0] *= (0.999 / new_radii[new_radii > 1.0])[:, np.newaxis]
            
            new_score = _compute_true_score(new_pts)
            if new_score > best_obj:
                best_obj = new_score
                best_step = step
        
        if best_step == 0.0:
            # Try even smaller steps if none worked
            for step in [1e-5, 5e-5]:
                new_pts = pts + step * grad
                new_radii = np.linalg.norm(new_pts, axis=1)
                if np.any(new_radii > 1.0):
                    new_pts[new_radii > 1.0] *= (0.999 / new_radii[new_radii > 1.0])[:, np.newaxis]
                new_score = _compute_true_score(new_pts)
                if new_score > best_obj:
                    best_obj = new_score
                    best_step = step
        
        if best_step == 0.0:
            break
        
        pts = pts + best_step * grad
    
    return pts

def construct():
    start_time = time.time()
    time_limit = 145.0
    
    # Multi-start with different initializations
    inits = []
    
    # Hex lattice with different rotations
    for rot, seed in [(0.0, 42), (np.pi / 6, 43), (np.pi / 3, 44), (np.pi / 4, 45), (np.pi/12, 46)]:
        inits.append(_hex_lattice_init(N, rotation=rot, seed=seed))
    
    # Radial ring init
    inits.append(_radial_ring_init(N, seed=47))
    
    # Random init
    rng = np.random.RandomState(48)
    r = np.sqrt(rng.random(N))
    a = rng.random(N) * 2 * np.pi
    inits.append(np.column_stack([r * np.cos(a), r * np.sin(a)]))
    
    best_points = None
    best_score = -np.inf
    
    # First pass: optimize each init with annealed beta (Soft-Min)
    for idx, init in enumerate(inits):
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        
        current = init.copy()
        
        # Progressive beta annealing
        # We use a coarser annealing schedule to save time for the exact polish
        for beta in [50.0, 200.0, 500.0]:
            remaining = time_limit - (time.time() - start_time)
            if remaining < 5:
                break
            current = _optimize_lbfgs(current, beta, max_iter=200, 
                                       time_limit=min(10.0, remaining - 2))
        
        # Intermediate score check
        score = _compute_true_score(current)
        if score > best_score:
            best_score = score
            best_points = current.copy()
        
        # If we have time, do a quick exact polish on this candidate
        # to see if it improves significantly
        remaining = time_limit - (time.time() - start_time)
        if remaining > 10:
            polished = _exact_subgradient_polish(current, max_iters=100, time_limit=min(5.0, remaining - 2))
            polished_score = _compute_true_score(polished)
            if polished_score > best_score:
                best_score = polished_score
                best_points = polished.copy()
    
    # Second pass: refine the best solution further with Exact Subgradient Polish
    if best_points is not None and time.time() - start_time < time_limit:
        current = best_points.copy()
        
        # Alternate between low-beta soft-min (to escape local minima) and exact polish
        for _ in range(3):
            remaining = time.time() - start_time
            if remaining > time_limit - 5:
                break
            
            # 1. Small soft-min step to adjust configuration
            current = _optimize_lbfgs(current, 100.0, max_iter=100, 
                                       time_limit=min(5.0, time_limit - remaining - 5))
            
            # 2. Exact subgradient polish
            current = _exact_subgradient_polish(current, max_iters=200, time_limit=min(10.0, time_limit - (time.time() - start_time) - 2))
            
            score = _compute_true_score(current)
            if score > best_score:
                best_score = score
                best_points = current.copy()
    
    # Final high-precision polish
    if best_points is not None and time.time() - start_time < time_limit - 5:
        current = best_points.copy()
        current = _exact_subgradient_polish(current, max_iters=500, time_limit=min(15.0, time_limit - (time.time() - start_time) - 2))
        score = _compute_true_score(current)
        if score > best_score:
            best_score = score
            best_points = current.copy()
    
    if best_points is None:
        # Fallback: return hex lattice init
        best_points = _hex_lattice_init(N)
    
    # Final validation: ensure all points are inside the unit circle
    norms = np.linalg.norm(best_points, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        best_points[mask] = best_points[mask] * (0.9999 / norms[mask])[:, np.newaxis]
    
    return best_points
