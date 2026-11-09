# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import sys

def _cubic_grid(m):
    """Return (m**3, 3) array of cubic grid points in [-0.5, 0.5]^3."""
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid

def _grid_score(points, r):
    """Compute the largest radius that fits for the given points."""
    if len(points) < 2:
        return 0.5
    # min pairwise distance
    # Use broadcasting for N=100
    d = points[:, None, :] - points[None, :, :]
    dists = np.sqrt(np.sum(d**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    # distance to boundary
    margins = CUBE_HALF - np.abs(points)
    min_boundary = np.min(margins)
    return min(min_pair, min_boundary) / 2.0

def _gradient_refine(points, max_iter=5000, start_temp=0.05, end_temp=0.001, lr=0.001):
    """
    Refine points using gradient ascent on a smooth surrogate (Log-Sum-Exp of negative distances).
    We minimize g(x) = sum exp(-d_ij/tau) + sum exp(-m_i/tau).
    """
    N = len(points)
    best_pts = points.copy()
    best_score = _grid_score(best_pts, 0.1)
    
    current_temp = start_temp
    for it in range(max_iter):
        # Compute distances and margins
        d = best_pts[:, None, :] - best_pts[None, :, :]
        dists = np.sqrt(np.sum(d**2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        
        margins = CUBE_HALF - np.abs(best_pts)
        
        # Weights for log-sum-exp approximation
        w_pair = np.exp(-dists / current_temp)
        np.fill_diagonal(w_pair, 0)
        
        w_bound = np.exp(-margins / current_temp)
        
        # Gradient accumulation
        unit_vecs = d / dists[:, :, None]
        
        grad = np.zeros_like(best_pts)
        # Pairwise gradient contribution
        grad += np.sum(w_pair[:, :, None] * unit_vecs, axis=1) / current_temp
        
        # Boundary gradient contribution
        sign_x = np.sign(best_pts)
        sign_x[best_pts == 0] = 1.0 
        grad += (w_bound * sign_x) / current_temp[:, None]
        
        # Update
        best_pts -= lr * grad
        
        # Project back to cube
        best_pts = np.clip(best_pts, -CUBE_HALF, CUBE_HALF)
        
        # Check score
        score = _grid_score(best_pts, 0.1)
        if score > best_score:
            best_score = score
            
        # Anneal temperature
        current_temp = max(end_temp, current_temp * 0.995)
        
    return best_pts, best_score

def _body_centered_lattice():
    """
    Construct a BCC-like arrangement for 100 points in a unit cube.
    """
    nx, ny, nz = 5, 4, 5
    xs = np.linspace(-0.5 + 0.5/nx, 0.5 - 0.5/nx, nx)
    ys = np.linspace(-0.5 + 0.5/ny, 0.5 - 0.5/ny, ny)
    zs = np.linspace(-0.5 + 0.5/nz, 0.5 - 0.5/nz, nz)
    grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    return grid

def _fcc_lattice():
    """
    Approximate FCC packing in a cube.
    """
    nx, ny, nz = 4, 5, 5
    xs = np.linspace(-0.5, 0.5, nx, endpoint=True)
    ys = np.linspace(-0.5, 0.5, ny, endpoint=True)
    zs = np.linspace(-0.5, 0.5, nz, endpoint=True)
    
    points = []
    for k, z in enumerate(zs):
        for j, y in enumerate(ys):
            for i, x in enumerate(xs):
                offset = 0.5 / nx if (i + j + k) % 2 == 0 else 0.0
                px = x + offset
                py = y + offset
                pz = z + offset
                px = max(-0.5, min(0.5, px))
                py = max(-0.5, min(0.5, py))
                pz = max(-0.5, min(0.5, pz))
                points.append((px, py, pz))
    return np.array(points[:100])

def construct():
    start_time = time.time()
    
    rng = np.random.default_rng(42)
    
    configs = [
        ("cubic_5x4x5", _body_centered_lattice()),
    ]
    
    m5 = 5
    ticks5 = (np.arange(m5) + 0.5) / m5 - 0.5
    grid5 = np.array([(x, y, z) for z in ticks5 for y in ticks5 for x in ticks5])
    configs.append(("cubic_5x5x4", grid5[:100]))
    
    try:
        fcc = _fcc_lattice()
        if len(fcc) >= 100:
            configs.append(("fcc_like", fcc[:100]))
    except:
        pass
    
    best_pts = None
    best_score = -1.0
    
    for name, pts in configs:
        if time.time() - start_time > DEADLINE_S - 30:
            break
        if len(pts) != 100:
            continue
        pts = np.clip(pts, -0.5, 0.5)
        score = _grid_score(pts, 0.1)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
    
    if best_pts is None:
        best_pts = _body_centered_lattice()
        best_score = _grid_score(best_pts, 0.1)
    
    elapsed = time.time() - start_time
    remaining = DEADLINE_S - 5 - elapsed
    if remaining > 10:
        n_restarts = max(1, int(remaining / 1.5))
        for restart in range(min(n_restarts, 15)):
            if time.time() - start_time > DEADLINE_S - 5:
                break
            rng = np.random.default_rng(1000 + restart)
            if restart == 0:
                pts_to_refine = best_pts.copy()
            else:
                pts_to_refine = best_pts + rng.normal(0, 0.02, size=(100, 3))
                pts_to_refine = np.clip(pts_to_refine, -0.5, 0.5)
            
            refined_pts, refined_score = _gradient_refine(pts_to_refine, max_iter=2000, start_temp=0.05, end_temp=0.005, lr=0.002)
            if refined_score > best_score:
                best_score = refined_score
                best_pts = refined_pts.copy()
    
    best_pts = np.clip(best_pts, -0.5, 0.5)
    best_score = _grid_score(best_pts, best_score)
    
    return best_pts
# EVOLVE-BLOCK-END
