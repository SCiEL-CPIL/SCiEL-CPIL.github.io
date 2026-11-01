import numpy as np
import time
from scipy.optimize import minimize

N = 50
START_TIME = time.time()
TIME_LIMIT = 115.0  # Safety margin for 120s limit

def compute_radius(pts):
    """
    Compute the maximum radius for the given centers.
    Returns -1e6 if invalid (outside cube).
    """
    if np.any(np.abs(pts) > 0.5 + 1e-9):
        return -1e6
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    wall_dists = 0.5 - np.abs(pts)
    min_wall_dist = np.min(wall_dists)
    return min(min_pair_dist / 2.0, min_wall_dist)

def neg_radius(pts):
    r = compute_radius(pts)
    return -r

def soft_min_radius(pts, beta=50.0):
    """
    Smooth surrogate for the negative radius using log-sum-exp soft-min.
    """
    if np.any(np.abs(pts) > 0.5 + 1e-9):
        return 1e6
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    
    pair_dists = dists / 2.0
    wall_dists = 0.5 - np.abs(pts)
    wall_flat = wall_dists.flatten()
    
    pair_flat = pair_dists[np.triu_indices(N, k=1)]
    all_dists = np.concatenate([pair_flat, wall_flat])
    
    d_min = np.min(all_dists)
    shifted = -beta * (all_dists - d_min)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min_val = d_min - log_sum / beta
    
    return -soft_min_val

def get_grid_init(nx=5, ny=5, nz=2):
    """
    Generate a grid initialization.
    """
    xs = (np.arange(nx) + 0.5) / nx - 0.5
    ys = (np.arange(ny) + 0.5) / ny - 0.5
    zs = (np.arange(nz) + 0.5) / nz - 0.5
    
    pts = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    pts += np.random.normal(0, 0.001, size=(N, 3))
    pts = np.clip(pts, -0.49, 0.49)
    return pts

def get_fcc_init():
    """
    Generate an FCC lattice initialization.
    """
    fcc_pts = []
    cell_size = 1.0 / 3.0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                x0, y0, z0 = i * cell_size, j * cell_size, k * cell_size
                fcc_pts.append([x0, y0, z0])
                fcc_pts.append([x0 + 0.5 * cell_size, y0 + 0.5 * cell_size, z0])
                fcc_pts.append([x0 + 0.5 * cell_size, y0, z0 + 0.5 * cell_size])
                fcc_pts.append([x0, y0 + 0.5 * cell_size, z0 + 0.5 * cell_size])
    
    fcc_pts = np.array(fcc_pts)
    fcc_pts -= 0.5
    
    mask = np.all(np.abs(fcc_pts) <= 0.5, axis=1)
    fcc_pts = fcc_pts[mask]
    
    if len(fcc_pts) > N:
        selected = []
        remaining = fcc_pts.copy()
        for _ in range(N):
            if not remaining.size:
                break
            pt = remaining[0]
            selected.append(pt)
            d = np.linalg.norm(remaining - pt, axis=1)
            remaining = remaining[d > 0.12]
        fcc_pts = np.array(selected)
        
    if len(fcc_pts) < N:
        n_pad = N - len(fcc_pts)
        pad = np.random.uniform(-0.45, 0.45, size=(n_pad, 3))
        fcc_pts = np.vstack([fcc_pts, pad])
        
    return fcc_pts[:N]

def get_bcc_init():
    """
    Generate a body-centered cubic lattice initialization.
    """
    bcc_pts = []
    cell_size = 1.0 / 3.0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                x0, y0, z0 = i * cell_size, j * cell_size, k * cell_size
                bcc_pts.append([x0, y0, z0])
                bcc_pts.append([x0 + 0.5 * cell_size, y0 + 0.5 * cell_size, z0 + 0.5 * cell_size])
    
    bcc_pts = np.array(bcc_pts)
    bcc_pts -= 0.5
    
    mask = np.all(np.abs(bcc_pts) <= 0.5, axis=1)
    bcc_pts = bcc_pts[mask]
    
    if len(bcc_pts) > N:
        selected = []
        remaining = bcc_pts.copy()
        for _ in range(N):
            if not remaining.size:
                break
            pt = remaining[0]
            selected.append(pt)
            d = np.linalg.norm(remaining - pt, axis=1)
            remaining = remaining[d > 0.12]
        bcc_pts = np.array(selected)
        
    if len(bcc_pts) < N:
        n_pad = N - len(bcc_pts)
        pad = np.random.uniform(-0.45, 0.45, size=(n_pad, 3))
        bcc_pts = np.vstack([bcc_pts, pad])
        
    return bcc_pts[:N]

def local_refine_smooth(pts):
    """
    Refine using smooth surrogate (L-BFGS-B) followed by exact polish (Nelder-Mead).
    """
    x = pts.flatten().copy()
    bounds = [(-0.499, 0.499)] * (N * 3)
    
    betas = [20.0, 50.0, 100.0, 200.0]
    for beta in betas:
        if time.time() - START_TIME > TIME_LIMIT - 10.0:
            break
        try:
            def obj(x):
                p = x.reshape(N, 3)
                return soft_min_radius(p, beta)
            
            res = minimize(
                obj, x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            x = res.x
        except Exception:
            pass
    
    x_2d = x.reshape(N, 3)
    x_2d = np.clip(x_2d, -0.499, 0.499)
    
    try:
        res = minimize(
            neg_radius, x_2d,
            method='Nelder-Mead',
            options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        x_2d = res.x
    except Exception:
        pass
    
    x_2d = np.clip(x_2d, -0.499, 0.499)
    return x_2d

def local_refine_exact(pts):
    """
    Refine using exact Nelder-Mead only.
    """
    x = pts.flatten().copy()
    
    try:
        res = minimize(
            neg_radius, x,
            method='Nelder-Mead',
            options={'maxiter': 200, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
        )
        x = res.x
    except Exception:
        pass
    
    x = np.clip(x, -0.499, 0.499)
    return x.reshape(N, 3)

def construct():
    best_pts = None
    best_r = -np.inf
    
    # Generate diverse initializations
    inits = [
        get_grid_init(5, 5, 2),
        get_fcc_init(),
        get_bcc_init(),
        np.random.uniform(-0.45, 0.45, size=(N, 3))
    ]
    
    # Add more random restarts if time permits
    while time.time() - START_TIME < TIME_LIMIT - 30.0:
        inits.append(np.random.uniform(-0.45, 0.45, size=(N, 3)))
    
    # Try each initialization
    for i, init_pts in enumerate(inits):
        if time.time() - START_TIME > TIME_LIMIT - 15.0:
            break
        
        init_pts = np.clip(init_pts, -0.49, 0.49)
        
        # Alternate between smooth and exact refinement strategies
        if i % 2 == 0:
            refined = local_refine_smooth(init_pts)
        else:
            refined = local_refine_exact(init_pts)
        
        refined = np.clip(refined, -0.499, 0.499)
        r = compute_radius(refined)
        
        if r > best_r:
            best_r = r
            best_pts = refined.copy()
        
        # If we have a good result, do a final polish
        if best_r > 0.135 and time.time() - START_TIME < TIME_LIMIT - 5.0:
            try:
                res = minimize(
                    neg_radius, best_pts,
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                )
                r2 = compute_radius(res.x)
                if r2 > best_r:
                    best_r = r2
                    best_pts = res.x.copy()
            except Exception:
                pass
    
    # If no valid points found, return grid
    if best_pts is None:
        grid = get_grid_init(5, 5, 2)
        return grid[:N]
    
    return best_pts
