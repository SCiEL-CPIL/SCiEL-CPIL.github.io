import time
import numpy as np

N = 50
DEADLINE = time.time() + 145.0

def _score(pts):
    """
    Compute the valid packing radius (or 0 if invalid).
    """
    if len(pts) != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    
    dists_origin = np.linalg.norm(pts, axis=1)
    if np.any(dists_origin > 1.0):
        return 0.0
    r_cont = 1.0 - np.max(dists_origin)
    if r_cont < 0:
        return 0.0
    
    if N < 2:
        return r_cont
        
    dx = pts[:, 0:1] - pts[0:1, 0]
    dy = pts[:, 1:2] - pts[0:1, 1]
    dist_mat = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(dist_mat, np.inf)
    r_pair = 0.5 * np.min(dist_mat)
    
    return min(r_cont, r_pair)

def _hex_init(spacing):
    """
    Generate a hexagonal lattice of points.
    """
    points = []
    k = int(2.0 / spacing) + 2
    for j in range(-k, k + 1):
        y = j * spacing * np.sqrt(3.0) / 2.0
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            r = np.hypot(x, y)
            if r + spacing / 2.0 <= 1.0:
                points.append((x, y))
    
    pts = np.array(points)
    if len(pts) < N:
        return None
    
    radii = np.linalg.norm(pts, axis=1)
    idx = np.argsort(radii, kind='stable')[:N]
    return pts[idx]

def _random_init():
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 1.0, N))
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

def _optimize(pts_init, max_time=10.0):
    """
    Optimize the packing using a force-directed method with simulated annealing.
    """
    pts = pts_init.copy().astype(np.float64)
    best_pts = pts.copy()
    best_score = _score(pts)
    
    T = 0.1
    cooling = 0.99
    step_scale = 0.01
    
    start_time = time.time()
    
    while time.time() < min(start_time + max_time, DEADLINE):
        forces = np.zeros_like(pts)
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff * diff, axis=-1)
        np.fill_diagonal(dists_sq, 1.0)
        
        weights = 2.0 / (dists_sq * dists_sq)
        np.fill_diagonal(weights, 0.0)
        
        forces_rep = np.sum(diff * weights[..., np.newaxis], axis=1)
        
        dists_origin = np.linalg.norm(pts, axis=1)
        dists_origin_safe = np.where(dists_origin < 1e-6, 1e-6, dists_origin)
        
        forces_boundary = np.zeros_like(pts)
        mask = dists_origin > 0.1
        denom = (1.0 - dists_origin_safe[mask]) ** 3
        denom = np.where(denom < 1e-9, 1e-9, denom)
        forces_boundary[mask] = -2.0 * (pts[mask] / dists_origin_safe[mask, np.newaxis]) / denom[:, np.newaxis]
        
        total_forces = forces_rep + forces_boundary
        
        f_norms = np.linalg.norm(total_forces, axis=1, keepdims=True)
        f_norms[f_norms == 0] = 1.0
        total_forces = total_forces / f_norms
        
        noise = np.random.normal(0, T, (N, 2))
        
        new_pts = pts + step_scale * (total_forces + noise)
        
        new_radii = np.linalg.norm(new_pts, axis=1)
        mask_out = new_radii > 1.0
        if np.any(mask_out):
            new_pts[mask_out] = new_pts[mask_out] / new_radii[mask_out, np.newaxis]
            
        new_score = _score(new_pts)
        
        if new_score > best_score:
            best_score = new_score
            best_pts = new_pts.copy()
            pts = new_pts.copy()
            T *= cooling
        else:
            pts = new_pts.copy()
            T *= cooling
            
        if T < 1e-6:
            break
            
    return best_pts, best_score

def construct():
    start_time = time.time()
    best_pts = None
    best_score = 0.0
    
    base_spacing = 0.2516
    for spacing_factor in [0.9, 0.95, 1.0, 1.05, 1.1]:
        spacing = base_spacing * spacing_factor
        pts_init = _hex_init(spacing)
        if pts_init is not None:
            score_init = _score(pts_init)
            if score_init > best_score:
                best_score = score_init
                best_pts = pts_init.copy()
                
            if time.time() < DEADLINE - 5.0:
                opt_pts, opt_score = _optimize(pts_init, max_time=10.0)
                if opt_score > best_score:
                    best_score = opt_score
                    best_pts = opt_pts.copy()
                    
    for _ in range(3):
        if time.time() > DEADLINE - 5.0:
            break
        pts_init = _random_init()
        radii = np.linalg.norm(pts_init, axis=1)
        mask = radii > 1.0
        if np.any(mask):
            pts_init[mask] = pts_init[mask] / radii[mask, np.newaxis]
            
        score_init = _score(pts_init)
        if score_init > best_score:
            best_score = score_init
            best_pts = pts_init.copy()
            
        opt_pts, opt_score = _optimize(pts_init, max_time=10.0)
        if opt_score > best_score:
            best_score = opt_score
            best_pts = opt_pts.copy()

    if best_pts is not None and time.time() < DEADLINE - 2.0:
        opt_pts, opt_score = _optimize(best_pts, max_time=5.0)
        if opt_score > best_score:
            best_score = opt_score
            best_pts = opt_pts.copy()

    if best_pts is None or _score(best_pts) <= 0:
        pts_fallback = []
        for i in range(10):
            for j in range(5):
                x = (i - 4.5) * 0.2
                y = (j - 2.0) * 0.2
                if np.hypot(x, y) < 0.9:
                    pts_fallback.append([x, y])
        if len(pts_fallback) < N:
             while len(pts_fallback) < N:
                 x, y = np.random.uniform(-0.8, 0.8, 2)
                 if np.hypot(x,y) < 0.8:
                     pts_fallback.append([x,y])
        best_pts = np.array(pts_fallback[:N])
        
    return best_pts
