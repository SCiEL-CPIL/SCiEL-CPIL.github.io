import numpy as np
from scipy.optimize import minimize
import time

N = 300

def _hex_lattice_init(n_points, rotation=0.0, seed=42):
    """Generate a hexagonal lattice of points inside the unit circle."""
    rng = np.random.RandomState(seed)
    s = np.sqrt(2 * np.pi / (n_points * np.sqrt(3)))
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
    
    points += rng.normal(0, s * 0.01, size=points.shape)
    
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
    
    points += rng.normal(0, s * 0.01, size=points.shape)
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

def _gd_polish(points, lr=0.001, n_iter=100, beta=500.0):
    """Gradient descent polish pass."""
    current = points.copy()
    for i in range(n_iter):
        val, grad = _soft_min_obj_and_grad(current.flatten(), beta)
        current = current - lr * grad.reshape(N, 2)
        norms = np.linalg.norm(current, axis=1)
        mask = norms > 0.999
        if np.any(mask):
            current[mask] = current[mask] * (0.999 / norms[mask])[:, np.newaxis]
    
    return current

def _anneal_beta(points, beta_start, beta_end, n_steps, time_limit, start_time):
    """Anneal beta from beta_start to beta_end over n_steps."""
    current = points.copy()
    for i in range(n_steps):
        if time.time() - start_time > time_limit:
            break
        frac = i / max(n_steps - 1, 1)
        beta = beta_start * (beta_end / beta_start) ** frac
        remaining = time_limit - (time.time() - start_time)
        if remaining < 2:
            break
        current = _optimize_lbfgs(current, beta, max_iter=200, 
                                   time_limit=min(8.0, remaining - 1))
    return current

def construct():
    start_time = time.time()
    time_limit = 145.0  # Leave 5s margin before 150s kill
    
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
    
    # First pass: optimize each init with annealed beta
    for idx, init in enumerate(inits):
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        
        current = init.copy()
        
        # Progressive beta annealing
        current = _anneal_beta(current, 50.0, 500.0, 4, 
                               time_limit - elapsed, start_time)
        
        # Final polish with high beta
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        current = _optimize_lbfgs(current, 1000.0, max_iter=300, 
                                   time_limit=min(10.0, time_limit - elapsed))
        current = _gd_polish(current, lr=0.0005, n_iter=50, beta=2000.0)
        score = _compute_true_score(current)
        if score > best_score:
            best_score = score
            best_points = current.copy()
    
    # Second pass: refine the best solution further
    if best_points is not None and time.time() - start_time < time_limit:
        current = best_points.copy()
        for beta in [2000.0, 5000.0, 10000.0]:
            elapsed = time.time() - start_time
            if elapsed > time_limit:
                break
            current = _optimize_lbfgs(current, beta, max_iter=500, 
                                       time_limit=min(15.0, time_limit - elapsed))
            current = _gd_polish(current, lr=0.0002, n_iter=100, beta=beta * 2)
            score = _compute_true_score(current)
            if score > best_score:
                best_score = score
                best_points = current.copy()
    
    # Final high-precision polish
    if best_points is not None and time.time() - start_time < time_limit - 5:
        current = best_points.copy()
        current = _optimize_lbfgs(current, 20000.0, max_iter=1000, 
                                   time_limit=min(10.0, time_limit - (time.time() - start_time)))
        current = _gd_polish(current, lr=0.0001, n_iter=200, beta=40000.0)
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
