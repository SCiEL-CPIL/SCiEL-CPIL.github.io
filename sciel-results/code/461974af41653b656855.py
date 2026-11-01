import time
import numpy as np
from scipy.optimize import minimize

N = 50
# 150s limit, keep 10s safety margin
_DEADLINE = time.time() + 140.0 

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
    
    # Soft min calculation
    y = -beta * all_dists
    max_y = np.max(y)
    y_shifted = y - max_y
    sum_exp = np.sum(np.exp(y_shifted))
    log_sum_exp = max_y + np.log(sum_exp)
    obj_val = log_sum_exp / beta
    
    weights = np.exp(y_shifted) / sum_exp
    
    grad_pts = np.zeros_like(pts)
    
    # 1. Boundary terms
    w_boundary = weights[0:N]
    safe_dists = np.where(dists_origin < 1e-8, 1.0, dists_origin)
    boundary_grad = (w_boundary / safe_dists)[:, np.newaxis] * pts
    grad_pts += boundary_grad
    
    # 2. Pairwise terms
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
    
    return obj_val, grad_pts.flatten()

def _hex_lattice_init():
    """
    Generate a hexagonal lattice initialization that fits 50 circles.
    This mimics the known optimal structure more closely.
    """
    pts = []
    
    # Center
    pts.append([0.0, 0.0])
    
    # Hexagonal rings: 6, 12, 18, ... points
    # For 50 circles: 1 + 6 + 12 + 18 + 13 = 50 (last ring truncated)
    ring_counts = [6, 12, 18, 13]
    
    # Optimal spacing for hex packing: d = 2*r where r is circle radius
    # We want the outermost ring to fit in unit circle
    # If we have 4 rings, the outermost radius is approximately 4*d
    # So 4*d + r <= 1, and d >= 2*r => 4*2*r + r <= 1 => 9r <= 1 => r <= 1/9
    # But we also need to account for the actual geometry
    
    # Use a scaling factor that makes the packing fit
    # Start with unit spacing and scale
    base_d = 0.25  # initial guess for spacing
    
    for ring_idx, count in enumerate(ring_counts, 1):
        # Ring radius for hexagonal packing
        # For hex lattice, ring k has radius k * d * sqrt(3)/2 approximately
        # But for simplicity, use linear scaling
        r = ring_idx * base_d
        
        # Generate angles for this ring
        angles = np.linspace(0, 2*np.pi, count, endpoint=False)
        
        # Offset alternating rings by pi/count for hexagonal packing
        if ring_idx % 2 == 0:
            angles += np.pi / count
        
        x = r * np.cos(angles)
        y = r * np.sin(angles)
        pts.extend(np.column_stack([x, y]))
    
    pts = np.array(pts[:N])
    
    # Scale to fit in unit circle
    max_r = np.max(np.linalg.norm(pts, axis=1))
    if max_r > 0:
        pts = pts * (0.95 / max_r)
    
    return pts

def _ring_init(ring_counts):
    """
    Generate a ring-based initial configuration.
    """
    pts = []
    
    if ring_counts[0] > 0:
        pts.append([0.0, 0.0])
        
    num_rings = len(ring_counts) - 1
    if num_rings == 0:
        pass
        
    if num_rings > 0:
        radii = np.linspace(0.2, 0.8, num_rings) if num_rings > 1 else [0.5]
        
        for k in range(num_rings):
            count = ring_counts[k+1]
            if count <= 0:
                continue
            r = radii[k]
            angles = np.linspace(0, 2*np.pi, count, endpoint=False)
            angles += np.random.uniform(-0.1, 0.1, count)
            x = r * np.cos(angles)
            y = r * np.sin(angles)
            pts.extend(np.column_stack([x, y]))
            
    pts = np.array(pts)
    return pts[:N]

def _generate_inits():
    """
    Generate multiple diverse initial configurations.
    """
    inits = []
    
    # 1. Hexagonal lattice
    hex_pts = _hex_lattice_init()
    hex_pts = hex_pts + np.random.normal(0, 0.005, (N, 2))
    radii = np.linalg.norm(hex_pts, axis=1)
    mask = radii > 1.0
    if np.any(mask):
        hex_pts[mask] = hex_pts[mask] / radii[mask, np.newaxis]
    inits.append(hex_pts)
    
    # 2. Ring configurations
    ring_configs = [
        [1, 6, 12, 15, 16],
        [1, 6, 12, 12, 19],
        [1, 6, 12, 10, 21],
        [1, 6, 10, 14, 19],
        [1, 6, 12, 18, 13],
        [1, 6, 14, 14, 15],
        [1, 6, 12, 16, 15],
    ]
    
    for cfg in ring_configs:
        pts = _ring_init(cfg)
        pts = pts + np.random.normal(0, 0.01, (N, 2))
        radii = np.linalg.norm(pts, axis=1)
        mask = radii > 1.0
        if np.any(mask):
            pts[mask] = pts[mask] / radii[mask, np.newaxis]
        inits.append(pts)
    
    # 3. Random initialization
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 0.9, N))
    random_pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    inits.append(random_pts)
    
    return inits

def _optimize_with_lbfgsb(pts0, beta_schedule, max_time=10.0):
    """
    Run L-BFGS-B optimization with a schedule of beta values.
    """
    start_time = time.time()
    current_pts = pts0.copy()
    
    for beta in beta_schedule:
        if time.time() - start_time > max_time:
            break
            
        res = minimize(
            _soft_min_obj_and_grad,
            current_pts.flatten(),
            args=(beta,),
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 800, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        
        current_pts = res.x.reshape(N, 2)
        
        # Project to unit disk
        radii = np.linalg.norm(current_pts, axis=1)
        mask = radii > 1.0
        if np.any(mask):
            current_pts[mask] = current_pts[mask] / radii[mask, np.newaxis]
            
    return current_pts

def construct():
    start_time = time.time()
    
    # 1. Generate initial configurations
    inits = _generate_inits()
    
    best_pts = inits[0]
    best_score = _score(best_pts)
    
    # 2. Optimization Loop
    # More aggressive beta schedule with more iterations
    beta_schedule = [5, 20, 50, 100, 200, 500]
    
    # Time budget: ~140s total, leave 15s for polish
    time_per_run = 12.0
    
    for i, init_pts in enumerate(inits):
        if time.time() > _DEADLINE - 15.0:
            break
            
        try:
            opt_pts = _optimize_with_lbfgsb(init_pts, beta_schedule, max_time=time_per_run)
            opt_score = _score(opt_pts)
            
            if opt_score > best_score:
                best_score = opt_score
                best_pts = opt_pts.copy()
        except Exception:
            pass
            
    # 3. Final Polish with very high beta
    if time.time() < _DEADLINE - 3.0:
        remaining_time = _DEADLINE - time.time() - 3.0
        polish_beta = [1000, 2000, 5000, 10000]
        try:
            polished_pts = _optimize_with_lbfgsb(best_pts, polish_beta, max_time=remaining_time)
            polished_score = _score(polished_pts)
            if polished_score > best_score:
                best_pts = polished_pts
                best_score = polished_score
        except Exception:
            pass
            
    # Final safety check
    final_score = _score(best_pts)
    if final_score <= 0:
        return inits[0]
        
    return best_pts

# EVOLVE-BLOCK-END
