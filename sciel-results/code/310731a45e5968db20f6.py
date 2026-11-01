import numpy as np
from scipy.optimize import minimize
import time

N = 300
SEED = 12345
T_START = time.time()
DEADLINE = 120.0  # Safety margin before 150s limit

# Precompute upper triangle indices (constant, computed once)
_I_IDX, _J_IDX = np.triu_indices(N, k=1)
_N_PAIRS = len(_I_IDX)

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
    
    # Minimum pairwise half-distance (only upper triangle)
    diff = points[_I_IDX] - points[_J_IDX]
    dists = np.sqrt(np.sum(diff * diff, axis=1))
    r_pair = 0.5 * np.min(dists)
    
    r = min(r_bound, r_pair)
    return float(r) if r > 0.0 else 0.0

def _soft_min_obj_and_grad(x, beta):
    """
    Compute the negative soft-min value and its gradient.
    x: shape (2N,)
    beta: softness parameter
    Uses precomputed upper-triangle indices for efficiency.
    """
    C = x.reshape(N, 2)
    
    # Pairwise distances (upper triangle only)
    D = C[_I_IDX] - C[_J_IDX]
    L = np.sqrt(np.sum(D * D, axis=1))
    L_safe = np.where(L > 1e-12, L, 1.0)
    
    # Margins
    x_pairs = L_safe / 2.0
    L_wall = np.linalg.norm(C, axis=1)
    x_walls = 1.0 - L_wall
    
    x_all = np.concatenate([x_pairs, x_walls])
    
    min_x = np.min(x_all)
    shifted_x = x_all - min_x
    exp_terms = np.exp(-beta * shifted_x)
    Z = np.sum(exp_terms)
    
    # Value: -soft_min
    val = -(min_x - (1.0 / beta) * np.log(Z))
    
    # Probabilities
    P = exp_terms / Z
    
    grad = np.zeros((N, 2))
    
    # Pairwise part
    P_pairs = P[:_N_PAIRS]
    W = P_pairs * 0.5
    
    # Unit vectors from i to j
    U = D / L_safe[:, np.newaxis]
    
    # df/dp_i = -W * U for each pair (i,j)
    # df/dp_j = +W * U for each pair (i,j)
    np.add.at(grad, _I_IDX, -W[:, np.newaxis] * U)
    np.add.at(grad, _J_IDX, W[:, np.newaxis] * U)
    
    # Boundary part
    P_walls = P[_N_PAIRS:]
    
    L_wall_safe = np.where(L_wall > 1e-12, L_wall, 1.0)
    U_walls = -C / L_wall_safe[:, np.newaxis]
    
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

def _local_polish(points, max_iter=2000, tol=1e-14, time_limit=15.0):
    """
    Direct local optimization of the true min objective using
    subgradient / projected gradient on binding constraints.
    This is a final polish that works on the exact objective.
    """
    start_time = time.time()
    C = points.copy()
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Compute all constraints
        D = C[_I_IDX] - C[_J_IDX]
        L = np.sqrt(np.sum(D * D, axis=1))
        x_pairs = L / 2.0
        L_wall = np.linalg.norm(C, axis=1)
        x_walls = 1.0 - L_wall
        
        x_all = np.concatenate([x_pairs, x_walls])
        min_x = np.min(x_all)
        
        # Find binding constraints (within tolerance of min)
        tol_bind = max(tol, min_x * 1e-10)
        bind_mask = x_all < (min_x + tol_bind)
        
        if not np.any(bind_mask):
            break
        
        # Build subgradient: sum of unit vectors for binding constraints
        subgrad = np.zeros((N, 2))
        
        n_bind_pairs = np.sum(bind_mask[:_N_PAIRS])
        if n_bind_pairs > 0:
            bind_pair_idx = np.where(bind_mask[:_N_PAIRS])[0]
            D_bind = D[bind_pair_idx]
            L_bind = L[bind_pair_idx]
            L_bind_safe = np.where(L_bind > 1e-12, L_bind, 1.0)
            U_bind = D_bind / L_bind_safe[:, np.newaxis]
            
            # For each binding pair (i,j): push i away from j, j away from i
            # subgrad[i] += -U (push i away from j means moving in -U direction... wait)
            # x_{ij} = 0.5*||p_i - p_j||, d(x_{ij})/dp_i = 0.5*U_{ij}
            # To increase x_{ij}, move p_i in +U direction, p_j in -U direction
            # But we want to maximize min, so we move in direction that increases the binding constraints
            # subgrad[i] += 0.5 * U_{ij}
            # subgrad[j] += -0.5 * U_{ij}
            np.add.at(subgrad, _I_IDX[bind_pair_idx], 0.5 * U_bind)
            np.add.at(subgrad, _J_IDX[bind_pair_idx], -0.5 * U_bind)
        
        n_bind_walls = np.sum(bind_mask[_N_PAIRS:])
        if n_bind_walls > 0:
            bind_wall_idx = np.where(bind_mask[_N_PAIRS:])[0]
            C_bind = C[bind_wall_idx]
            L_bind = L_wall[bind_wall_idx]
            L_bind_safe = np.where(L_bind > 1e-12, L_bind, 1.0)
            U_wall = -C_bind / L_bind_safe[:, np.newaxis]
            # x_i = 1 - ||p_i||, d(x_i)/dp_i = -p_i/||p_i|| = U_wall
            # To increase x_i, move in +U_wall direction (toward center)
            subgrad[bind_wall_idx] += U_wall
        
        # Normalize subgradient
        sg_norm = np.linalg.norm(subgrad)
        if sg_norm < 1e-15:
            break
        direction = subgrad / sg_norm
        
        # Adaptive step size
        step = 0.001
        for _ in range(50):
            C_new = C + step * direction
            # Project inside
            norms = np.linalg.norm(C_new, axis=1)
            mask = norms > 1.0
            if np.any(mask):
                C_new[mask] = C_new[mask] / norms[mask, None] * (1.0 - 1e-10)
            
            new_score = _score(C_new)
            if new_score > min_x - 1e-15:
                C = C_new
                break
            step *= 0.5
        
        if step < 1e-15:
            break
    
    return C

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

def _perturb(points, rng, magnitude=0.01):
    """Add small random perturbation."""
    return points + rng.normal(0, magnitude, size=points.shape)

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
    
    # 4. Perturbed hex lattice
    inits.append(_perturb(_lattice_seed(spacing), rng, 0.005))
    
    # 5. Another rotated lattice with different angle
    angle2 = np.pi / 12
    R2 = np.array([[np.cos(angle2), -np.sin(angle2)], [np.sin(angle2), np.cos(angle2)]])
    inits.append(_lattice_seed(spacing) @ R2.T)

    best_points = None
    best_score = -np.inf
    
    # Progressive Beta Schedule
    betas = [10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0]
    
    for init in inits:
        if time.time() - t0 > DEADLINE - 20:
            break
        
        current = init.copy()
        
        for beta in betas:
            if time.time() - t0 > DEADLINE - 20:
                break
            
            # Allocate time based on beta
            t_alloc = 3.0
            if beta > 1000:
                t_alloc = 5.0
                
            current = _optimize_lbfgs(current, beta, max_iter=150, time_limit=t_alloc)
            
            # Project back inside if needed
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
        # First: very high beta soft-min polish
        current = best_points.copy()
        current = _optimize_lbfgs(current, 50000.0, max_iter=300, time_limit=8.0)
        norms = np.linalg.norm(current, axis=1)
        mask = norms > 1.0
        if np.any(mask):
            current[mask] = current[mask] / norms[mask, None] * 0.999
        
        score = _score(current)
        if score > best_score:
            best_score = score
            best_points = current.copy()
        
        # Then: direct local polish on exact objective
        if time.time() - t0 < DEADLINE - 3:
            current = best_points.copy()
            current = _local_polish(current, max_iter=3000, tol=1e-14, time_limit=10.0)
            score = _score(current)
            if score > best_score:
                best_score = score
                best_points = current.copy()

    # Fallback
    if best_points is None:
        best_points = _lattice_seed(spacing)

    return best_points
