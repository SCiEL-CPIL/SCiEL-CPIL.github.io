import numpy as np
import time
import math

N = 50

def _compute_radius(centers):
    """Compute the valid radius for a set of centers inside the unit circle."""
    if len(centers) != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Distance to boundary
    dists_boundary = 1.0 - np.hypot(centers[:, 0], centers[:, 1])
    r_boundary = np.min(dists_boundary)
    if r_boundary <= 0:
        return 0.0
    # Pairwise distances
    dx = centers[:, 0:1] - centers[:, 0:1].T
    dy = centers[:, 1:2] - centers[:, 1:2].T
    dists = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(dists, np.inf)
    r_pair = np.min(dists) / 2.0
    r = min(r_boundary, r_pair)
    if r <= 0:
        return 0.0
    return r

def _hexagonal_packing():
    """Generate initial hexagonal lattice packing."""
    # Target radius is approx 0.1258. Spacing approx 0.2516.
    # We generate a large hex grid and pick the best N points.
    spacing = 0.25
    k = int(1.0 / spacing) + 2
    rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
            for i in range(-k, k + 1) for j in range(-k, k + 1)]
    points = np.array(rows)
    radius = np.hypot(points[:, 0], points[:, 1])
    # Keep points that fit comfortably
    mask = radius <= 1.0 - spacing / 2.0
    points = points[mask]
    if len(points) < N:
        return _random_packing()
    # Sort by distance from center to prioritize central positions? 
    # Or just take the first N. Sorting by radius ensures a compact core.
    points = points[np.argsort(radius, kind="stable")[:N]]
    return points

def _concentric_packing():
    """Generate initial concentric ring packing."""
    centers = []
    centers.append([0.0, 0.0])
    # 1 + 6 + 12 + 18 + 13 = 50
    rings = [
        (6, 0.25),
        (12, 0.50),
        (18, 0.75),
        (13, 0.95)
    ]
    for count, r in rings:
        angles = np.linspace(0, 2*np.pi, count, endpoint=False)
        # Add a phase shift for even rings to interlock better? 
        # Simple: no shift for now.
        for a in angles:
            centers.append([r * np.cos(a), r * np.sin(a)])
            
    centers = np.array(centers[:N])
    return centers

def _spiral_packing():
    """Generate initial Fibonacci spiral packing."""
    centers = []
    # Golden angle
    ga = math.pi * (3.0 - math.sqrt(5.0))
    # Scale factor so that the N-th point is near the boundary
    # r_n = c * sqrt(n)
    # We want r_N approx 0.9
    c = 0.9 / math.sqrt(N)
    for i in range(N):
        r = c * math.sqrt(i + 0.5)
        theta = ga * i
        centers.append([r * math.cos(theta), r * math.sin(theta)])
    return np.array(centers)

def _random_packing():
    """Generate random initial packing."""
    centers = np.random.uniform(-0.8, 0.8, size=(N, 2))
    return centers

def _optimize_annealing(centers_init, deadline):
    """
    Simulated Annealing with hard constraint projection.
    We maximize the minimum distance (radius).
    Objective: Minimize -min_dist.
    """
    centers = centers_init.copy()
    best_centers = centers.copy()
    best_r = _compute_radius(centers)
    
    # Precompute pairs for efficiency
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    # Initial temperature
    T = 0.05
    T_min = 1e-6
    cooling_rate = 0.995
    
    # Step size
    step_size = 0.05
    
    current_r = best_r
    
    # Track time
    start_time = time.time()
    
    while time.time() < deadline and T > T_min:
        # 1. Propose move: perturb all points or a subset?
        # Perturbing all points with small noise helps escape local minima but is slow.
        # Perturbing a random subset is faster.
        # Let's perturb all points with Gaussian noise.
        noise = np.random.normal(0, step_size, centers.shape)
        new_centers = centers + noise
        
        # 2. Repair/Project new_centers to be feasible (non-overlapping and inside)
        # This is the key "hard constraint" step.
        # We iterate a few times to push overlapping circles apart.
        
        feasible = True
        max_iter_repair = 50
        
        for _ in range(max_iter_repair):
            # Check boundary
            norms = np.linalg.norm(new_centers, axis=1)
            # If outside, project back to radius 1 - epsilon
            # But we don't know epsilon (radius) yet. 
            # Strategy: Project to boundary 1.0 first, then handle overlaps.
            # Actually, to maximize radius, we want them as close to boundary as possible?
            # No, we want to maximize the MINIMUM slack.
            
            # Simple projection:
            # 1. If any point is outside unit circle, move it to the circle.
            out_mask = norms > 1.0
            if np.any(out_mask):
                new_centers[out_mask] /= norms[out_mask, None]
                # Slightly pull in to avoid numerical issues? 
                # new_centers[out_mask] *= 0.999
            
            # 2. Resolve overlaps.
            # Calculate pairwise distances
            dx = new_centers[i_idx, 0] - new_centers[j_idx, 0]
            dy = new_centers[i_idx, 1] - new_centers[j_idx, 1]
            dists = np.sqrt(dx**2 + dy**2)
            
            # We want dists > 2*r_current_estimate?
            # Since we don't have a fixed r, we just want to separate them.
            # Let's assume a target separation based on current best_r.
            # If dist < 2 * best_r, they overlap in the "best" configuration.
            # But we are exploring. 
            # Let's just push them apart if they are very close (e.g., < 0.1) to avoid singularities.
            # Better: Use a repulsive force for all pairs closer than 2*current_min_dist.
            
            min_dist = np.min(dists)
            
            # If we have a valid configuration, we accept/reject based on energy.
            # For repair, we just want a valid configuration to evaluate.
            # Let's do a simple relaxation:
            # If dist < 0.1, push apart.
            close_mask = dists < 0.1
            if not np.any(close_mask):
                break
            
            # Push apart
            for k in np.where(close_mask)[0]:
                i = i_idx[k]
                j = j_idx[k]
                d = dists[k]
                if d < 1e-6:
                    # Random push
                    vec = np.random.normal(size=2)
                    vec /= np.linalg.norm(vec)
                else:
                    vec = (new_centers[i] - new_centers[j]) / d
                
                # Push each by half the needed distance
                # Need 0.1 - d. Push by 0.05 each.
                push = 0.05
                new_centers[i] += vec * push
                new_centers[j] -= vec * push
            
            # Re-check boundary after push
            norms = np.linalg.norm(new_centers, axis=1)
            out_mask = norms > 1.0
            if np.any(out_mask):
                new_centers[out_mask] /= norms[out_mask, None]
                
        # Evaluate new configuration
        new_r = _compute_radius(new_centers)
        
        if new_r > 0:
            # Acceptance criterion
            if new_r > current_r:
                # Always accept improvement
                centers = new_centers
                current_r = new_r
                if current_r > best_r:
                    best_r = current_r
                    best_centers = centers.copy()
            else:
                # Accept worse with probability exp((new_r - current_r) / T)
                # Note: new_r and current_r are radii. We want to maximize.
                # So delta = new_r - current_r is negative.
                # Prob = exp(delta / T)
                delta = new_r - current_r
                if delta < 0:
                    prob = np.exp(delta / T)
                    if np.random.random() < prob:
                        centers = new_centers
                        current_r = new_r
        
        # Cooling
        T *= cooling_rate
        # Adapt step size
        # If we are stuck, increase step size?
        # If we are improving, decrease step size?
        # Simple: keep constant or decay slowly.
        step_size *= 0.999
        
    return best_centers, best_r

def construct():
    deadline = time.time() + 110.0
    
    # Multi-start with diverse initializations
    inits = [
        _hexagonal_packing(),
        _concentric_packing(),
        _spiral_packing(),
        _random_packing(),
        _random_packing()
    ]
    
    best_centers = inits[0]
    best_r = 0.0
    
    # Allocate time to each start
    # Reserve 10s for final checks/return
    time_budget = deadline - time.time()
    time_per_init = time_budget / len(inits)
    
    for i, init in enumerate(inits):
        if time.time() > deadline - 5:
            break
            
        # Ensure at least 5 seconds per init
        t_start = time.time()
        c, r = _optimize_annealing(init, deadline - 5)
        
        if r > best_r:
            best_r = r
            best_centers = c
            
    return best_centers

if __name__ == "__main__":
    centers = construct()
    print(_compute_radius(centers))
