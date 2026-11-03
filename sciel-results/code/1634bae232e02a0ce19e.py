"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 50
DEADLINE_SECONDS = 130  # safety margin before 150s kill

# EVOLVE-BLOCK-START
def _score(centers):
    """Compute the maximum radius for given centers."""
    if len(centers) != N:
        return -1.0
    centers = np.asarray(centers, dtype=np.float64)
    if not np.all(np.isfinite(centers)):
        return -1.0
    # Wall distances
    wall_dists = np.min(0.5 - np.abs(centers), axis=1)
    min_wall = np.min(wall_dists)
    # Pairwise distances
    from scipy.spatial.distance import cdist
    d = cdist(centers, centers)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0
    return min(min_wall, min_pair)


def _bcc_lattice(n_points):
    """Generate a BCC lattice scaled to fit in [-0.5, 0.5]^3, take first n_points."""
    # BCC: points at corners and body center of each cubic cell
    # We'll generate a 3D grid of BCC unit cells
    # For N=50, we need roughly cube root of ~50/2 = 25 -> ~3 per axis for corners
    # Let's try a 4x4x4 BCC lattice: 4^3 = 64 corner points + 64 body centers = 128 total
    # But we only need 50. Let's use a smaller grid.
    
    # Try m x m x m BCC cells
    for m in range(2, 8):
        # Corner points: (i, j, k) for i,j,k in 0..m-1
        # Body center points: (i+0.5, j+0.5, k+0.5) for i,j,k in 0..m-1
        corners = np.array([(i, j, k) for k in range(m) for j in range(m) for i in range(m)], dtype=float)
        body_centers = np.array([(i+0.5, j+0.5, k+0.5) for k in range(m) for j in range(m) for i in range(m)], dtype=float)
        all_points = np.vstack([corners, body_centers])
        if len(all_points) >= n_points:
            break
    
    # Scale to fit in [-0.5, 0.5]^3
    # The lattice spans [0, m-1] in each dimension
    # We need to map [0, m-1] to [-0.5, 0.5]
    # But we should account for the radius, so let's just scale and then optimize
    span = m - 1
    if span > 0:
        all_points = all_points / span * 0.9 - 0.45  # Scale to fit with some margin
    else:
        all_points = np.zeros_like(all_points)
    
    # Shuffle and take first n_points for diversity
    idx = np.random.default_rng(42).permutation(len(all_points))[:n_points]
    return all_points[idx]


def _optimize_local(centers, max_iter=1000, tol=1e-10):
    """Local optimization using L-BFGS-B to maximize the minimum distance."""
    from scipy.optimize import minimize
    
    def negative_radius(x):
        c = x.reshape(-1, 3)
        # Wall distances
        wall_dists = 0.5 - np.abs(c)
        min_wall = np.min(wall_dists)
        # Pairwise distances
        from scipy.spatial.distance import cdist
        d = cdist(c, c)
        np.fill_diagonal(d, np.inf)
        min_pair = np.min(d) / 2.0
        return -min(min_wall, min_pair)
    
    def negative_radius_grad(x):
        c = x.reshape(-1, 3)
        wall_dists = 0.5 - np.abs(c)
        min_wall = np.min(wall_dists)
        
        from scipy.spatial.distance import cdist
        d = cdist(c, c)
        np.fill_diagonal(d, np.inf)
        min_pair = np.min(d) / 2.0
        
        r = min(min_wall, min_pair)
        grad = np.zeros_like(x)
        
        if min_wall <= min_pair:
            # Gradient from wall constraint
            for i in range(N):
                for k in range(3):
                    if wall_dists[i, k] == min_wall:
                        grad[i*3+k] = -np.sign(c[i, k]) if c[i, k] != 0 else 0
        else:
            # Gradient from pair constraint
            for i in range(N):
                for j in range(i+1, N):
                    if d[i, j] == 2 * min_pair:
                        diff = c[i] - c[j]
                        norm = np.linalg.norm(diff)
                        if norm > 1e-12:
                            grad[i*3:i*3+3] += diff / norm
                            grad[j*3:j*3+3] -= diff / norm
        
        return grad
    
    x0 = centers.flatten()
    
    # Bounds: each coordinate in [-0.5, 0.5]
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    result = minimize(
        negative_radius,
        x0,
        method='L-BFGS-B',
        bounds=bounds,
        maxiter=max_iter,
        ftol=tol,
        gtol=tol,
        options={'maxls': 20, 'maxcor': 10}
    )
    
    return result.x.reshape(-1, 3)


def _perturb(centers, magnitude=0.05):
    """Add random perturbation to escape local minima."""
    rng = np.random.default_rng()
    return centers + rng.normal(0, magnitude, size=centers.shape)


def construct():
    start_time = time.time()
    
    # Initialize with BCC lattice
    centers = _bcc_lattice(N)
    best_centers = centers.copy()
    best_score = _score(best_centers)
    
    # Multi-stage optimization with multiple restarts
    rng = np.random.default_rng(42)
    
    for restart in range(10):
        if time.time() - start_time > DEADLINE_SECONDS - 30:
            break
        
        if restart == 0:
            current = best_centers.copy()
        else:
            # Perturb from best or random restart
            if restart % 3 == 0:
                current = _bcc_lattice(N)
            else:
                current = _perturb(best_centers, magnitude=0.08)
        
        # Clip to valid range
        current = np.clip(current, -0.5, 0.5)
        
        # Local optimization
        optimized = _optimize_local(current, max_iter=500, tol=1e-9)
        optimized = np.clip(optimized, -0.5, 0.5)
        
        score = _score(optimized)
        if score > best_score:
            best_score = score
            best_centers = optimized
        
        # Additional polish with tighter tolerance
        if time.time() - start_time < DEADLINE_SECONDS - 15:
            polished = _optimize_local(best_centers, max_iter=1000, tol=1e-12)
            polished = np.clip(polished, -0.5, 0.5)
            polish_score = _score(polished)
            if polish_score > best_score:
                best_score = polish_score
                best_centers = polished
    
    # Final check and return
    final_score = _score(best_centers)
    if final_score < 0:
        # Fallback to simple grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]
    
    return best_centers
# EVOLVE-BLOCK-END
