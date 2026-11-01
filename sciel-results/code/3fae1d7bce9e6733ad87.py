import numpy as np
from scipy.optimize import minimize
import time

N = 300
SEED = 12345
T_START = time.time()
DEADLINE = 120.0  # Safety margin before 150s limit

def _init_dist(points):
    """Compute initial pairwise distance matrix."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    return dists

def _score(points):
    """Compute the true packing radius."""
    if points.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    
    # Distance to container boundary
    norm = np.linalg.norm(points, axis=1)
    r_bound = 1.0 - np.max(norm)
    if r_bound < 0.0:
        return 0.0
    
    # Minimum pairwise half-distance
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    r_pair = 0.5 * np.min(dists)
    
    r = min(r_bound, r_pair)
    return float(r) if r > 0.0 else 0.0

def _soft_min_obj_and_grad(x, beta):
    """
    Compute the negative soft-min value and its gradient.
    x: shape (2N,)
    beta: softness parameter
    """
    C = x.reshape(N, 2)
    
    # Pairwise distances
    diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
    L = np.linalg.norm(diff, axis=2)
    
    # Handle zero distances (should not happen if init is good, but just in case)
    L_safe = np.where(L > 1e-12, L, 1.0)
    
    # Margins
    x_pairs = L_safe / 2.0
    L_wall = np.linalg.norm(C, axis=1)
    x_walls = 1.0 - L_wall
    
    # Weights for soft-min: w_k = exp(-beta * x_k)
    # To avoid underflow/overflow, we can shift by min, but for gradient we need raw exps relative to Z
    # Actually, for stability, we should compute Z = sum(exp(-beta * x_k))
    # If beta is large, exp(-beta * x_k) can underflow for non-minimal x_k.
    # This is actually good: it focuses on the minimum.
    # But if all x_k are large (bad packing), exp(-beta * x_k) might be 0.
    # Let's shift by min to keep numbers stable, then adjust.
    # soft_min = min_x - (1/beta) * log(sum(exp(-beta * (x_k - min_x))))
    # This is numerically stable.
    
    x_pairs_flat = x_pairs[np.triu_indices(N, k=1)]
    x_all = np.concatenate([x_pairs_flat, x_walls])
    
    min_x = np.min(x_all)
    shifted_x = x_all - min_x
    exp_terms = np.exp(-beta * shifted_x)
    Z = np.sum(exp_terms)
    
    # Value: -soft_min
    val = -(min_x - (1.0 / beta) * np.log(Z))
    
    # Gradient
    # d(val)/dx_k = - d(soft_min)/dx_k
    # d(soft_min)/dx_k = exp(-beta*(x_k - min_x)) / Z
    # So d(val)/dx_k = - exp(-beta*(x_k - min_x)) / Z
    
    P = exp_terms / Z  # Probabilities
    
    grad = np.zeros_like(C)
    
    # Pairwise part
    # P_pairs = P[:len(i_idx)]
    i_idx, j_idx = np.triu_indices(N, k=1)
    P_pairs = P[:len(i_idx)]
    
    D_flat = diff[i_idx, j_idx]
    L_flat = L[i_idx, j_idx]
    L_flat_safe = np.where(L_flat > 1e-12, L_flat, 1.0)
    
    # Unit vectors
    U = D_flat / L_flat_safe[:, np.newaxis]
    
    # Contribution to grad:
    # d(val)/dp_i = - sum_j P_{ij} * 0.5 * U_{ij}
    # d(val)/dp_j = - sum_i P_{ji} * 0.5 * (-U_{ji}) = + sum_i P_{ji} * 0.5 * U_{ji}
    # Note U_{ji} = -U_{ij}
    # So d(val)/dp_j = - sum_i P_{ji} * 0.5 * U_{ij}
    
    W = P_pairs * 0.5
    
    # For each pair (i, j):
    # grad[i] -= W * U
    # grad[j] -= W * U  (Wait, let's re-verify signs)
    
    # Let's re-derive carefully.
    # S = -1/beta log(sum exp(-beta x_k))
    # dS/dp_i = sum_k (dS/dx_k) (dx_k/dp_i)
    # dS/dx_k = P_k
    # x_{ij} = 0.5 ||p_i - p_j||
    # dx_{ij}/dp_i = 0.5 (p_i - p_j) / ||p_i - p_j|| = 0.5 U_{ij}
    # dx_{ij}/dp_j = 0.5 (p_j - p_i) / ||p_j - p_i|| = -0.5 U_{ij}
    
    # dS/dp_i = P_{ij} * 0.5 U_{ij} + ... (boundary)
    # dS/dp_j = P_{ij} * (-0.5 U_{ij}) + ...
    
    # We minimize f = -S.
    # df/dp_i = -dS/dp_i = - P_{ij} * 0.5 U_{ij}
    # df/dp_j = -dS/dp_j = + P_{ij} * 0.5 U_{ij}
    
    # So:
    # grad[i] -= W * U
    # grad[j] += W * U
    
    np.add.at(grad, i_idx, -W[:, np.newaxis] * U)
    np.add.at(grad, j_idx, W[:, np.newaxis] * U)
    
    # Boundary part
    P_walls = P[len(i_idx):]
    
    L_wall_safe = np.where(L_wall > 1e-12, L_wall, 1.0)
    U_walls = -C / L_wall_safe[:, np.newaxis]
    
    # x_i = 1 - ||p_i||
    # dx_i/dp_i = -p_i/||p_i|| = U_walls
    # dS/dp_i = P_i * U_walls
    # df/dp_i = -P_i * U_walls
    
    grad += -P_walls[:, np.newaxis] * U_walls
    
    return val, grad.flatten()

def _lattice_seed(spacing):
    """Generate a hexagonal lattice and pick N points inside the disk."""
    while True:
        k = int(1.0 / spacing) + 3
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                rows.append((x, y))
        pts = np.array(rows)
        rad = np.hypot(pts[:, 0], pts[:, 1])
        keep = rad <= 1.0 - spacing / 2.0
        pts = pts[keep]
        if len(pts) >= N:
            order = np.argsort(np.hypot(pts[:, 0], pts[:, 1]), kind="stable")
            return pts[order[:N]].copy()
        spacing *= 0.97

def _optimize_lbfgs(points, beta, max_iter=200, time_limit=5.0):
    """Run L-BFGS-B optimization with a time limit."""
    start_time = time.time()
    
    def obj_and_grad(x):
        if time.time() - start_time > time_limit:
            raise TimeoutError()
        return _soft_min_obj_and_grad(x, beta)
    
    try:
        res = minimize(obj_and_grad, points.flatten(), method='L-BFGS-B', jac=True,
                       options={'maxiter': max_iter, 'gtol': 1e-8, 'ftol': 1e-14})
        if res.success or res.status in [1, 2, 4]:
            return res.x.reshape(N, 2)
    except Exception:
        pass
    return points

def construct():
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    # Initializations
    inits = []
    
    # 1. Hex lattice
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    inits.append(_lattice_seed(spacing))
    
    # 2. Hex lattice rotated
    pts_rot = _lattice_seed(spacing)
    angle = np.pi / 6
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    inits.append(pts_rot @ R.T)
    
    # 3. Radial rings
    # Estimate ring radius
    s = np.sqrt(2 * np.pi / (N * np.sqrt(3)))
    n_rings = int(1.0 / s) + 1
    points = []
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
    pts_ring = np.array(points)
    if len(pts_ring) < N:
        r_rand = np.sqrt(rng.random(N - len(pts_ring)))
        a_rand = rng.random(N - len(pts_ring)) * 2 * np.pi
        extra = np.column_stack([r_rand * np.cos(a_rand), r_rand * np.sin(a_rand)])
        pts_ring = np.vstack([pts_ring, extra])
    elif len(pts_ring) > N:
        dists = np.linalg.norm(pts_ring, axis=1)
        indices = np.argsort(dists)
        pts_ring = pts_ring[indices[:N]]
    inits.append(pts_ring)

    best_points = None
    best_score = -np.inf
    
    # Progressive Beta Schedule
    betas = [10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0]
    
    for init in inits:
        if time.time() - t0 > DEADLINE:
            break
        
        current = init.copy()
        
        for beta in betas:
            if time.time() - t0 > DEADLINE:
                break
            
            # Allocate time based on beta (higher beta needs more care/iterations?)
            # Or just fixed small chunks.
            t_alloc = 4.0
            if beta > 1000:
                t_alloc = 6.0
                
            current = _optimize_lbfgs(current, beta, max_iter=100, time_limit=t_alloc)
            
            # Project back inside if needed (L-BFGS might drift)
            norms = np.linalg.norm(current, axis=1)
            mask = norms > 1.0
            if np.any(mask):
                current[mask] = current[mask] / norms[mask, None] * 0.999
            
            score = _score(current)
            if score > best_score:
                best_score = score
                best_points = current.copy()
                
    # Final high-precision polish on the best found
    if best_points is not None and time.time() - t0 < DEADLINE - 5:
        current = best_points.copy()
        # One more pass with very high beta and tight tolerance
        current = _optimize_lbfgs(current, 20000.0, max_iter=500, time_limit=10.0)
        norms = np.linalg.norm(current, axis=1)
        mask = norms > 1.0
        if np.any(mask):
            current[mask] = current[mask] / norms[mask, None] * 0.999
            
        score = _score(current)
        if score > best_score:
            best_score = score
            best_points = current

    # Fallback
    if best_points is None:
        best_points = _lattice_seed(spacing)

    return best_points
