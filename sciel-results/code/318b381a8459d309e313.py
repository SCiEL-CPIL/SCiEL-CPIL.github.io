"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import cdist


def _compute_radius(centers):
    """Compute the largest common radius for given centers."""
    n = len(centers)
    if n < 2:
        return 0.5
    # Pairwise distances
    d = cdist(centers, centers)
    # Mask diagonal
    np.fill_diagonal(d, np.inf)
    min_pair_dist = np.min(d)
    # Distance to boundary
    min_boundary = np.min(0.5 - np.abs(centers), axis=1)
    return min(min_pair_dist / 2.0, min_boundary)


def _fcc_lattice_init():
    """Generate FCC lattice points in the cube and select best 100."""
    # FCC lattice: points at (i, j, k) where i+j+k is even, scaled by spacing
    # For a unit cube, we want spacing such that we get ~100 points
    # FCC has density pi/(3*sqrt(2)) ~ 0.7405
    # For N=100 in unit cube, typical spacing ~ 100^(-1/3) ~ 0.215
    # But FCC nearest neighbor distance = spacing * sqrt(2)/2 = spacing * 0.7071
    # So radius = spacing * 0.7071 / 2 = spacing * 0.3536
    
    # Let's try different spacings
    best_centers = None
    best_radius = 0
    
    for spacing in np.arange(0.15, 0.35, 0.01):
        # Generate FCC points: (i*s, j*s, k*s) where i+j+k is even
        max_coord = int(np.ceil(0.5 / spacing)) + 1
        points = []
        for i in range(-max_coord, max_coord + 1):
            for j in range(-max_coord, max_coord + 1):
                for k in range(-max_coord, max_coord + 1):
                    if (i + j + k) % 2 == 0:
                        pt = np.array([i * spacing, j * spacing, k * spacing])
                        if np.all(np.abs(pt) <= 0.5):
                            points.append(pt)
        
        if len(points) < N:
            continue
        
        points = np.array(points)
        
        # For each point, compute minimum distance to boundary
        boundary_dists = np.min(0.5 - np.abs(points), axis=1)
        
        # Sort by boundary distance descending, take top N
        # But we also need to ensure pairwise distances are good
        # Let's just take the ones closest to center (max boundary distance)
        idx = np.argsort(boundary_dists)[::-1][:N]
        centers = points[idx]
        
        r = _compute_radius(centers)
        if r > best_radius:
            best_radius = r
            best_centers = centers.copy()
    
    return best_centers, best_radius


def _kmc_refine(centers, max_iters=5000, temp=0.01, deadline=None):
    """Kinetic Monte Carlo refinement with constraint-targeted push-apart."""
    n = len(centers)
    centers = centers.copy()
    best_radius = _compute_radius(centers)
    best_centers = centers.copy()
    
    t0 = time.time()
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute current radius
        d = cdist(centers, centers)
        np.fill_diagonal(d, np.inf)
        min_pair_dist = np.min(d)
        boundary_dists = 0.5 - np.abs(centers)
        min_boundary = np.min(boundary_dists, axis=1)
        current_radius = min(min_pair_dist / 2.0, np.min(min_boundary))
        
        if current_radius > best_radius:
            best_radius = current_radius
            best_centers = centers.copy()
        
        # Find the tightest constraints
        # Get the pair with minimum distance
        min_idx = np.unravel_index(np.argmin(d), d.shape)
        i, j = min_idx
        
        # Also consider boundary constraints
        boundary_min_idx = np.argmin(min_boundary)
        
        # Decide which constraint to target
        if min_pair_dist / 2.0 < np.min(min_boundary):
            # Pair constraint is tighter
            # Push i and j apart
            diff = centers[i] - centers[j]
            dist = np.linalg.norm(diff)
            if dist > 1e-12:
                direction = diff / dist
            else:
                direction = np.random.randn(3)
                direction /= np.linalg.norm(direction)
            
            # Target distance slightly larger than current
            target_dist = min_pair_dist + temp * np.random.exponential(1.0)
            move = (target_dist - dist) * direction * 0.5
            
            new_i = centers[i] + move
            new_j = centers[j] - move
            
            # Check validity
            valid = True
            for pt in [new_i, new_j]:
                if np.any(np.abs(pt) > 0.5):
                    valid = False
                    break
            
            if valid:
                # Check if this improves things (accept with probability)
                centers[i] = new_i
                centers[j] = new_j
                
                # Compute new radius
                d_new = cdist(centers, centers)
                np.fill_diagonal(d_new, np.inf)
                new_min_pair = np.min(d_new)
                new_boundary = 0.5 - np.abs(centers)
                new_min_boundary = np.min(new_boundary, axis=1)
                new_radius = min(new_min_pair / 2.0, np.min(new_min_boundary))
                
                # Metropolis acceptance
                delta_r = new_radius - current_radius
                if delta_r < 0 and np.random.random() < np.exp(delta_r / (temp * 0.1)):
                    # Reject
                    centers[i] = centers[i] - move
                    centers[j] = centers[j] + move
        else:
            # Boundary constraint is tighter
            k = boundary_min_idx
            # Push k towards center
            direction = -centers[k]
            norm = np.linalg.norm(direction)
            if norm > 1e-12:
                direction /= norm
            
            move_mag = min_boundary[k] + temp * np.random.exponential(1.0) - min_boundary[k]
            move = direction * move_mag * 0.5
            
            new_k = centers[k] + move
            if np.all(np.abs(new_k) <= 0.5):
                centers[k] = new_k
                
                # Compute new radius
                d_new = cdist(centers, centers)
                np.fill_diagonal(d_new, np.inf)
                new_min_pair = np.min(d_new)
                new_boundary = 0.5 - np.abs(centers)
                new_min_boundary = np.min(new_boundary, axis=1)
                new_radius = min(new_min_pair / 2.0, np.min(new_min_boundary))
                
                delta_r = new_radius - current_radius
                if delta_r < 0 and np.random.random() < np.exp(delta_r / (temp * 0.1)):
                    centers[k] = new_k - move
        
        # Anneal temperature
        temp = max(0.001, temp * 0.999)
    
    return best_centers, best_radius


def construct():
    t_start = time.time()
    deadline = t_start + 145.0  # Safety margin before 150s limit
    
    # Start with FCC lattice initialization
    centers, radius = _fcc_lattice_init()
    if centers is None:
        # Fallback to simple cubic
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        centers = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])[:N]
        radius = _compute_radius(centers)
    
    # KMC refinement
    centers, radius = _kmc_refine(centers, max_iters=20000, temp=0.02, deadline=deadline)
    
    # Check time and return
    if time.time() > deadline - 5.0:
        pass  # Already at deadline
    
    return centers
# EVOLVE-BLOCK-END
