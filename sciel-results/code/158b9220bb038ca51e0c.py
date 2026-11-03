import numpy as np
import time
from scipy.optimize import minimize
import sys

N = 300

def _get_hex_lattice(n_points):
    """Generate a hexagonal lattice clipped to the unit circle."""
    # Estimate spacing
    # Area of unit circle is pi.
    # Area per hex point is sqrt(3)/2 * s^2.
    # n * sqrt(3)/2 * s^2 ~ pi
    # s ~ sqrt(2*pi / (n*sqrt(3)))
    s = np.sqrt(2 * np.pi / (n_points * np.sqrt(3)))
    
    # Generate grid
    # We need to generate enough points to cover the circle
    # Radius of circle is 1.
    # Max i, j such that point is within 1.
    # Point (i, j) in hex coords:
    # x = s * (i + 0.5 * (j % 2))
    # y = s * (j * sqrt(3) / 2)
    
    # Let's just generate a large square grid and filter
    # s is approx 0.1. 1/s is approx 10.
    # So range -15 to 15 should be safe.
    max_k = int(1.0 / s) + 3
    points = []
    for i in range(-max_k, max_k + 1):
        for j in range(-max_k, max_k + 1):
            x = s * (i + 0.5 * (j % 2))
            y = s * (j * np.sqrt(3.0) / 2.0)
            if x*x + y*y <= 1.0:
                points.append((x, y))
                
    points = np.array(points)
    
    # If we have more than n_points, we need to select the best n_points.
    # The best points are those closest to the center? Or those that form the densest packing?
    # For a lattice, the "best" subset of size N is usually the one with the smallest max radius 
    # or simply the N points closest to the origin, as the boundary effect is worst there.
    # However, simply taking the N closest to origin might leave "holes" near the boundary 
    # if the lattice is cut.
    # A better heuristic: Sort by radius, take N closest.
    radii = np.hypot(points[:, 0], points[:, 1])
    idx = np.argsort(radii)
    points = points[idx[:n_points]]
    
    return points

def _calc_radius(points):
    """Calculate the packing radius for a set of points."""
    if len(points) < 2:
        return 1.0 - np.max(np.hypot(points[:, 0], points[:, 1]))
    
    # Pairwise distances
    # Using broadcasting: points[:, np.newaxis, :] - points[np.newaxis, :, :]
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Boundary distance
    norms = np.hypot(points[:, 0], points[:, 1])
    min_boundary_dist = 1.0 - np.max(norms)
    
    r = min(min_pair_dist / 2.0, min_boundary_dist)
    return r

def construct():
    start_time = time.time()
    time_limit = 110.0 # Safety margin
    
    # 1. Initialization
    points = _get_hex_lattice(N)
    current_radius = _calc_radius(points)
    best_points = points.copy()
    best_radius = current_radius
    
    # 2. Simulated Annealing
    # Temperature
    T = 0.01
    T_min = 1e-6
    cooling_rate = 0.995
    
    # Move parameters
    move_size = 0.05
    
    # We will perform moves until time limit
    # To speed up, we can batch distance calculations?
    # For N=300, _calc_radius is ~1-2ms.
    # We can do ~50k iterations in 110s.
    
    while time.time() - start_time < time_limit:
        # Propose move
        # Pick a random subset of points to move
        n_move = np.random.randint(1, 6) # 1 to 5 points
        idx = np.random.choice(N, n_move, replace=False)
        
        new_points = points.copy()
        # Gaussian noise
        noise = np.random.normal(0, move_size, size=(n_move, 2))
        new_points[idx] += noise
        
        # Keep points inside circle? 
        # If they go outside, radius becomes negative or very small.
        # It's better to allow them to go outside but penalize heavily via radius calc.
        # _calc_radius handles this: 1 - |c_i| will be negative.
        
        new_radius = _calc_radius(new_points)
        
        # Metropolis criterion
        if new_radius > current_radius:
            points = new_points
            current_radius = new_radius
            if current_radius > best_radius:
                best_radius = current_radius
                best_points = points.copy()
        else:
            # Accept with probability exp((new - current) / T)
            delta = new_radius - current_radius
            if delta < 0:
                prob = np.exp(delta / T)
                if np.random.rand() < prob:
                    points = new_points
                    current_radius = new_radius
            # else: delta == 0, accept? usually yes.
            else:
                points = new_points
                current_radius = new_radius
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
        # Occasionally reheat or reset move size?
        # If stuck (no improvement in many steps), maybe increase move size?
        # For now, keep it simple.
        
    # 3. Polish with Local Optimization
    # Use the best points found by SA
    x0 = best_points.flatten()
    
    def objective(x):
        pts = x.reshape(-1, 2)
        # We want to maximize radius, so minimize -radius
        # Add a small penalty for points going far outside to help gradient?
        # The radius function is 0 if any point is outside (1-|c| < 0).
        # Actually, if |c| > 1, 1-|c| is negative.
        # The min will be negative.
        # This is a valid objective to minimize.
        r = _calc_radius(pts)
        return -r
        
    # Use L-BFGS-B
    # It might struggle with non-smooth min, but let's try.
    # We can also try a smooth surrogate.
    # Let's try L-BFGS-B first.
    
    try:
        res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-12})
        if res.success and -res.fun > best_radius:
            best_points = res.x.reshape(-1, 2)
            best_radius = -res.fun
    except Exception:
        pass
    
    # Final check and return
    return best_points

# EVOLVE-BLOCK-START
# (The code above is the replacement for the evolve block)
# EVOLVE-BLOCK-END
