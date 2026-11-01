import time
import numpy as np
from scipy.optimize import minimize

N = 50
_START_TIME = time.time()
_DEADLINE = _START_TIME + 135.0  # 15s safety margin before 150s kill

def _score(pts):
    """Compute the valid packing radius (or 0 if invalid)."""
    if len(pts) != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    
    # Check if inside unit circle
    dists_origin = np.linalg.norm(pts, axis=1)
    if np.any(dists_origin > 1.0 + 1e-9):
        return 0.0
    
    # Calculate min distance to boundary
    r_cont = 1.0 - np.max(dists_origin)
    if r_cont < 0:
        return 0.0
        
    # Calculate min pairwise distance
    if N < 2:
        return r_cont
        
    # Efficient pairwise distance
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_mat = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dist_mat, np.inf)
    r_pair = 0.5 * np.min(dist_mat)
    
    return min(r_cont, r_pair)

def _soft_min_obj_and_grad(flat, beta):
    """
    Compute the soft-min objective and gradient for L-BFGS-B.
    Objective: minimize f(x) = (1/beta) * ln(sum(exp(-beta * x_k)))
    where x_k are the critical distances (pairwise and boundary).
    Minimizing this is equivalent to maximizing the soft-min of x_k.
    """
    pts = flat.reshape(N, 2)
    
    # Calculate boundary distances: 1 - |c_i|
    dists_origin = np.linalg.norm(pts, axis=1)
    boundary_dists = 1.0 - dists_origin
    
    # Calculate pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_mat = np.linalg.norm(diff, axis=-1)
    
    # We only need upper triangle for unique pairs
    iu = np.triu_indices(N, k=1)
    pair_dists = dist_mat[iu]
    
    # Combine all critical distances
    all_dists = np.concatenate([boundary_dists, pair_dists / 2.0])
    
    # Soft min calculation with log-sum-exp trick for numerical stability
    y = -beta * all_dists
    max_y = np.max(y)
    y_shifted = y - max_y
    sum_exp = np.sum(np.exp(y_shifted))
    log_sum_exp = max_y + np.log(sum_exp)
    obj_val = log_sum_exp / beta
    
    # Gradient
    weights = np.exp(y_shifted) / sum_exp
    
    grad_pts = np.zeros_like(pts)
    
    # 1. Boundary terms: d/dpts_i of (1 - |c_i|) = -pts_i / |c_i|
    w_boundary = weights[0:N]
    safe_dists = np.where(dists_origin < 1e-8, 1.0, dists_origin)
    boundary_grad = (w_boundary / safe_dists)[:, np.newaxis] * pts
    grad_pts += boundary_grad
    
    # 2. Pairwise terms: d/dpts_i of (|c_i - c_j|/2) = 0.5 * (c_i - c_j)/|c_i - c_j|
    w_pairs = weights[N:]
    i_idx, m_idx = iu
    d_pairs = dist_mat[i_idx, m_idx]
    safe_d_pairs = np.where(d_pairs < 1e-8, 1.0, d_pairs)
    vecs = pts[i_idx] - pts[m_idx]
    unit_vecs = vecs / safe_d_pairs[:, np.newaxis]
    
    grad_contrib_i = -0.5 * w_pairs[:, np.newaxis] * unit_vecs
    grad_contrib_m = 0.5 * w_pairs[:, np.newaxis] * unit_vecs
    
    np.add.at(grad_pts, i_idx, grad_contrib_i)
    np.add.at(grad_pts, m_idx, grad_contrib_m)
    
    grad_flat = grad_pts.flatten()
    
    return obj_val, grad_flat

def _hex_lattice_init():
    """Generate a hexagonal lattice packing as initial guess."""
    best_pts = None
    best_score = -1.0
    
    for spacing in [0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29]:
        k = int(1.0 / spacing) + 3
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        
        mask = radius <= 1.0
        points = points[mask]
        radius = radius[mask]
        
        if len(points) >= N:
            idx = np.argsort(radius, kind="stable")[:N]
            pts = points[idx]
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts
    
    if best_pts is not None:
        return best_pts
        
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 0.9, N))
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

def _ring_init():
    """Generate a ring-based initialization matching known optimal structure."""
    best_pts = None
    best_score = -1.0
    
    # Known optimal structure: 1 center + 4 rings: 1, 6, 12, 18, 13 = 50
    rings = [1, 6, 12, 18, 13]
    ring_radii = [0.0, 0.25, 0.50, 0.75, 0.95]
    
    for seed in range(20):
        rng = np.random.default_rng(42 + seed)
        pts_list = []
        
        for ring_idx, count in enumerate(rings):
            if ring_idx == 0:
                pts_list.append([0.0, 0.0])
            else:
                radius = ring_radii[ring_idx]
                angle_offset = rng.uniform(0, 2 * np.pi)
                for k in range(count):
                    angle = angle_offset + 2 * np.pi * k / count
                    x = radius * np.cos(angle)
                    y = radius * np.sin(angle)
                    r_jitter = rng.normal(0, 0.003)
                    x += r_jitter * np.cos(angle)
                    y += r_jitter * np.sin(angle)
                    pts_list.append([x, y])
        
        pts = np.array(pts_list[:N])
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts
    
    if best_pts is not None:
        return best_pts
    return _hex_lattice_init()

def _random_init():
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 0.95, N))
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

def _perturb(pts, sigma=0.01):
    """Add small random perturbation to escape local minima."""
    noise = np.random.normal(0, sigma, pts.shape)
    return pts + noise

def _project_to_disk(pts):
    """Project points back into the unit disk."""
    dists = np.linalg.norm(pts, axis=1)
    mask = dists > 1.0
    if np.any(mask):
        pts = pts.copy()
        pts[mask] = pts[mask] / dists[mask, np.newaxis]
    return pts

def _force_directed_relax(centers, max_steps=3000):
    """
    Force-directed relaxation using repulsive springs and boundary constraint.
    Acts as a local search / escape mechanism from local minima.
    """
    current = centers.copy()
    n = len(current)
    best_score = _score(current)
    best_positions = current.copy()
    
    for step in range(max_steps):
        if time.time() > _DEADLINE - 15.0:
            break
        
        forces = np.zeros_like(current)
        
        # Repulsive forces between all pairs
        i_idx, j_idx = np.triu_indices(n, k=1)
        diff = current[i_idx, :] - current[j_idx, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=1))
        safe_dists = np.where(dists > 1e-8, dists, 1.0)
        
        # Strong repulsion when close, using 1/d^2 potential
        repulsion_strength = 1.0 / (safe_dists ** 2 + 1e-6)
        unit_vecs = diff / safe_dists[:, np.newaxis]
        weighted_forces = repulsion_strength[:, np.newaxis] * unit_vecs
        
        np.add.at(forces, i_idx, weighted_forces)
        np.add.at(forces, j_idx, -weighted_forces)
        
        # Boundary constraint: push inward if too close to edge
        norms = np.linalg.norm(current, axis=1)
        # Soft boundary: quadratic penalty outside radius 0.9
        boundary_force = np.where(norms > 0.9, (norms - 0.9) ** 2 * 10.0, 0.0)
        safe_norms = np.where(norms > 1e-8, norms, 1.0)
        forces -= boundary_force[:, np.newaxis] * current / safe_norms[:, np.newaxis]
        
        # Adaptive time step based on max force
        max_force = np.max(np.linalg.norm(forces, axis=1))
        dt = 0.0005 / (1.0 + max_force * 5.0)
        
        current += dt * forces
        current = _project_to_disk(current)
        
        # Check improvement
        s = _score(current)
        if s > best_score:
            best_score = s
            best_positions = current.copy()
    
    return best_positions

def _optimize_lbfgs(pts, beta, maxiter=500, gtol=1e-8):
    """Run L-BFGS-B optimization with given beta."""
    try:
        res = minimize(
            _soft_min_obj_and_grad,
            pts.flatten(),
            args=(beta,),
            method='L-BFGS-B',
            jac=True,
            bounds=[(-1.0, 1.0)] * (2 * N),
            options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': gtol, 'disp': False}
        )
        new_pts = res.x.reshape(N, 2)
        new_pts = _project_to_disk(new_pts)
        return new_pts
    except Exception:
        return pts

def construct():
    start_time = time.time()
    
    # Generate diverse initializations
    inits = []
    inits.append(_hex_lattice_init())
    inits.append(_ring_init())
    for _ in range(8):
        inits.append(_random_init())
        
    best_pts = inits[0]
    best_score = _score(best_pts)
    
    # Phase 1: For each init, do force-directed relaxation then L-BFGS-B with adaptive beta
    betas = [10, 50, 200, 1000]
    
    for init_pts in inits:
        if time.time() > _DEADLINE - 25.0:
            break
            
        current_pts = init_pts.copy()
        
        # Force-directed relaxation to get a better starting point
        current_pts = _force_directed_relax(current_pts, max_steps=500)
        s = _score(current_pts)
        if s > best_score:
            best_score = s
            best_pts = current_pts.copy()
            
        # L-BFGS-B with increasing beta
        for beta in betas:
            if time.time() > _DEADLINE - 20.0:
                break
                
            new_pts = _optimize_lbfgs(current_pts, beta, maxiter=300, gtol=1e-8)
            new_score = _score(new_pts)
            
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
                current_pts = new_pts.copy()
                
            # If we got a good improvement, do a short force-directed pass to escape
            if new_score > best_score - 1e-6 and time.time() < _DEADLINE - 25.0:
                escaped = _force_directed_relax(current_pts, max_steps=200)
                esc_score = _score(escaped)
                if esc_score > best_score:
                    best_score = esc_score
                    best_pts = escaped.copy()
                    current_pts = escaped.copy()
        
        # Check time
        if time.time() > _DEADLINE - 20.0:
            break
            
    # Phase 2: Multiple final polish passes with very high beta on the best found
    polish_betas = [2000, 5000, 10000, 20000, 50000, 100000]
    for beta in polish_betas:
        if time.time() > _DEADLINE - 5.0:
            break
        new_pts = _optimize_lbfgs(best_pts, beta, maxiter=500, gtol=1e-12)
        new_score = _score(new_pts)
        if new_score > best_score:
            best_score = new_score
            best_pts = new_pts
    
    # Phase 3: Perturbation-based local search near the best solution
    if time.time() < _DEADLINE - 10.0:
        for sigma in [0.02, 0.01, 0.005, 0.002, 0.001]:
            if time.time() > _DEADLINE - 5.0:
                break
            for attempt in range(6):
                if time.time() > _DEADLINE - 5.0:
                    break
                perturbed = _perturb(best_pts, sigma)
                perturbed = _project_to_disk(perturbed)
                
                # Force-directed escape
                perturbed = _force_directed_relax(perturbed, max_steps=300)
                
                # Quick polish with moderate beta
                new_pts = _optimize_lbfgs(perturbed, 500, maxiter=300, gtol=1e-10)
                new_score = _score(new_pts)
                if new_score > best_score:
                    best_score = new_score
                    best_pts = new_pts
    
    return best_pts
