"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import scipy.optimize
import time

N = 300
START_TIME = time.time()
TIME_LIMIT = 140.0 # 10s safety margin

def compute_radius(pts):
    """
    Computes the maximum radius r for the given centers pts.
    r = min( min_{i<j} ||p_i - p_j|| / 2, min_i (1 - ||p_i||) )
    """
    # Boundary constraint
    dists_to_center = np.linalg.norm(pts, axis=1)
    r_boundary = 1.0 - np.min(dists_to_center)
    
    # Pairwise distance constraint
    # Vectorized distance calculation
    # Shape (N, N)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    r_pair = min_pair_dist / 2.0
    
    return min(r_boundary, r_pair)

def objective(pts_flat):
    """
    Objective function for scipy.optimize.minimize.
    Minimizes -radius.
    """
    pts = pts_flat.reshape(N, 2)
    r = compute_radius(pts)
    return -r

def local_optimize(initial_pts):
    """
    Runs Nelder-Mead optimization on the initial points.
    Returns the optimized points and the best radius found.
    """
    initial_flat = initial_pts.flatten()
    
    # Nelder-Mead options
    options = {
        'maxiter': 2000,
        'xatol': 1e-8,
        'fatol': 1e-8,
        'adaptive': True
    }
    
    try:
        res = scipy.optimize.minimize(objective, initial_flat, method='Nelder-Mead', options=options)
        best_pts = res.x.reshape(N, 2)
        best_r = compute_radius(best_pts)
        return best_pts, best_r
    except Exception:
        return initial_pts, compute_radius(initial_pts)

def generate_jittered_lattice(jitter_scale):
    """
    Generates a jittered hexagonal lattice as an initial guess.
    """
    # Estimate spacing for N points in unit circle
    # Area of unit circle = pi. Area of 1 circle = pi*r^2.
    # N * pi * r^2 <= pi  => r <= 1/sqrt(N)
    # r ~ 1/sqrt(300) ~ 0.057
    # Spacing d = 2r ~ 0.114
    # Let's start with a slightly larger spacing to ensure validity, then let optimizer shrink/expand?
    # No, we want to start with valid points.
    # Let's use a spacing that fits N points comfortably.
    
    # Hexagonal lattice generation
    # We need to fill the unit circle.
    # Let's try a range of spacings and pick the one that yields ~N points, then jitter.
    
    best_points = None
    best_count_diff = float('inf')
    
    for spacing in np.linspace(0.05, 0.15, 10):
        k = int(1.0 / spacing) + 2
        rows = []
        for i in range(-k, k + 1):
            for j in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x*x + y*y <= 1.0:
                    rows.append((x, y))
        
        if len(rows) >= N:
            # If we have more than N, we can just take a subset or use all and let optimizer prune?
            # The task requires exactly N circles.
            # If we have more, we can randomly select N? Or keep the ones closest to center?
            # Keeping closest to center is safer for validity initially.
            pts = np.array(rows)
            r = np.linalg.norm(pts, axis=1)
            pts = pts[np.argsort(r)[:N]]
            
            # Apply jitter
            jitter = np.random.normal(0, jitter_scale, pts.shape)
            pts += jitter
            
            # Ensure validity (inside circle)
            # If jitter pushed it out, project back? Or just clip?
            # Clipping might cause overlaps.
            # Let's just check if valid. If not, try smaller jitter or different spacing.
            # For now, assume small jitter keeps it valid if spacing was safe.
            # To be safe, we can scale down if any point is outside.
            norms = np.linalg.norm(pts, axis=1)
            max_norm = np.max(norms)
            if max_norm > 1.0:
                pts *= (1.0 / max_norm) * 0.99 # Scale down to be inside
            
            return pts

    # Fallback: Random points
    return np.random.uniform(-1, 1, (N, 2)) * 0.5

def construct():
    global START_TIME, TIME_LIMIT
    
    best_r = 0.0
    best_pts = None
    
    # Initial best guess from seed logic (hex lattice)
    # Let's create a few diverse initial guesses
    initial_guesses = []
    
    # 1. Jittered lattice with small jitter
    initial_guesses.append(generate_jittered_lattice(0.01))
    
    # 2. Jittered lattice with larger jitter
    initial_guesses.append(generate_jittered_lattice(0.03))
    
    # 3. Random points scaled to fit
    pts_rand = np.random.uniform(-1, 1, (N, 2))
    norms = np.linalg.norm(pts_rand, axis=1)
    pts_rand = pts_rand / np.max(norms) * 0.9
    initial_guesses.append(pts_rand)

    for i, init_pts in enumerate(initial_guesses):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
            
        # Run local optimization
        opt_pts, opt_r = local_optimize(init_pts)
        
        if opt_r > best_r:
            best_r = opt_r
            best_pts = opt_pts
            
        # Check time
        if time.time() - START_TIME > TIME_LIMIT - 5:
            break

    if best_pts is None:
        # Fallback to seed behavior
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            best_pts = points[np.argsort(radius, kind="stable")[:N]]
        else:
            # Should not happen
            best_pts = np.zeros((N, 2))
            best_pts[:, 0] = np.linspace(-0.9, 0.9, N)
            best_pts[:, 1] = 0

    return best_pts

# EVOLVE-BLOCK-START
# (The code above is the implementation. The block markers are for the evolution system.)
# EVOLVE-BLOCK-END
