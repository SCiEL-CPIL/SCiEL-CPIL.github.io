"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_radius(c):
    """Compute the common radius for centres c (N,3) in cube [-0.5,0.5]^3."""
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    # Wall distances
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    return min(min_pair / 2.0, min_wall)


def _initial_fcc_like(n):
    """Generate an initial configuration inspired by FCC lattice."""
    # FCC lattice: points at (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0) in unit cell
    # Scale to fit in [-0.5, 0.5]^3
    pts = []
    # Generate FCC points in a larger region, then select the best n
    a = 0.5  # lattice constant candidate
    offsets = np.array([
        [0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]
    ])
    # Generate points in a 3x3x3 supercell
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                for off in offsets:
                    p = np.array([i, j, k]) * a + off - 0.5
                    pts.append(p)
    pts = np.array(pts)
    # Keep only points inside the cube
    mask = np.all(np.abs(pts) <= 0.5, axis=1)
    pts = pts[mask]
    
    # If we have more than n points, select the best subset greedily
    if len(pts) > n:
        # Sort by distance from center, prefer central points
        d_center = np.sum(pts * pts, axis=1)
        order = np.argsort(d_center)
        pts = pts[order]
    
    # If still not enough, add random points
    if len(pts) < n:
        extra = np.random.uniform(-0.45, 0.45, size=(n - len(pts), 3))
        pts = np.vstack([pts, extra])
    
    return pts[:n].copy()


def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s
    
    np.random.seed(42)
    
    # Start with FCC-like initial configuration
    c = _initial_fcc_like(N)
    # Small random perturbation to break symmetry
    c += np.random.uniform(-0.01, 0.01, size=c.shape)
    # Clamp to cube
    c = np.clip(c, -0.499, 0.499)
    
    best_c = c.copy()
    best_r = _compute_radius(best_c)
    
    # Simulated annealing
    T = 0.05
    T_min = 1e-6
    cooling = 0.999
    
    # Precompute for speed
    n = N
    
    step_count = 0
    while T > T_min:
        if time.time() > deadline - 5.0:
            break
        
        # Move one or a few random points
        n_moves = 1
        if np.random.random() < 0.3:
            n_moves = 2
        if np.random.random() < 0.1:
            n_moves = 3
        
        idx = np.random.choice(n, size=n_moves, replace=False)
        old = c[idx].copy()
        
        # Random displacement
        scale = T * 2.0
        c[idx] += np.random.normal(0, scale, size=(n_moves, 3))
        # Clamp to cube
        c = np.clip(c, -0.499, 0.499)
        
        r_new = _compute_radius(c)
        r_old = _compute_radius(old) if n_moves == 1 else best_r
        
        # Accept if better or with probability exp((r_new - r_old)/T)
        if r_new >= r_old:
            if r_new > best_r:
                best_r = r_new
                best_c = c.copy()
        else:
            delta = r_new - r_old
            if delta / T > np.log(np.random.random()):
                pass  # accept
            else:
                c[idx] = old
        
        T *= cooling
        step_count += 1
        
        # Periodic check
        if step_count % 1000 == 0:
            if time.time() > deadline - 10.0:
                break
    
    return best_c
# EVOLVE-BLOCK-END
