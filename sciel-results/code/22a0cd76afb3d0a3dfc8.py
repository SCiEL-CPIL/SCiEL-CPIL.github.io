"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 300

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree

def _init_spiral(N):
    """
    Initialize points using an Archimedean spiral / Fibonacci lattice.
    This provides a uniform distribution in the unit disk.
    """
    # Golden angle
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    
    # Radii: Uniform in area
    # r_i = sqrt((i + 0.5) / N) * R_max
    # R_max slightly less than 1 to allow for boundary optimization
    R_max = 0.99
    indices = np.arange(N)
    radii = np.sqrt((indices + 0.5) / N) * R_max
    angles = indices * golden_angle
    
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    
    return np.column_stack((x, y))

def _score(pts):
    """
    Calculate the packing radius.
    r = min(min_pair_dist / 2, min_dist_to_boundary)
    """
    # Boundary constraint
    norms = np.linalg.norm(pts, axis=1)
    r_bound = np.min(1.0 - norms)
    
    # Pairwise constraint
    # Use KDTree for efficiency
    tree = cKDTree(pts)
    # Query nearest neighbor for each point
    dists, _ = tree.query(pts, k=2)
    # The first column is distance to self (0), second is nearest neighbor
    min_pair_dist = np.min(dists[:, 1])
    r_pair = min_pair_dist / 2.0
    
    return min(r_pair, r_bound)

def _simulate_annealing(init_pts, max_iter=100000, T_start=0.1, T_end=1e-6, cooling_rate=0.9995):
    """
    Simulated Annealing optimization.
    """
    pts = init_pts.copy()
    current_score = _score(pts)
    best_score = current_score
    best_pts = pts.copy()
    
    T = T_start
    
    for i in range(max_iter):
        # Select a random point to move
        idx = np.random.randint(N)
        
        # Determine move magnitude based on temperature
        # Scale factor 0.5 is a heuristic
        move_size = T * 0.5
        delta = np.random.normal(0, move_size, 2)
        
        # Apply move
        orig_pt = pts[idx].copy()
        pts[idx] += delta
        
        # Check if point is inside the unit circle
        norm_new = np.linalg.norm(pts[idx])
        if norm_new > 1.0:
            # Revert move
            pts[idx] = orig_pt
            T *= cooling_rate
            if T < T_end:
                break
            continue
        
        # Calculate new score
        new_score = _score(pts)
        
        # Metropolis Acceptance Criterion
        if new_score >= current_score:
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_pts = pts.copy()
        else:
            # Accept with probability exp(-delta_E / T)
            delta_e = current_score - new_score
            # Avoid overflow/underflow issues, though delta_e is small
            if T > 1e-9:
                prob = np.exp(-delta_e / T)
            else:
                prob = 0.0
            
            if np.random.random() < prob:
                current_score = new_score
            else:
                # Revert move
                pts[idx] = orig_pt
        
        # Cool down
        T *= cooling_rate
        if T < T_end:
            break
            
    return best_pts

def construct():
    """
    Main entry point.
    Runs multiple trials of Simulated Annealing starting from Spiral initialization.
    """
    best_overall_pts = None
    best_overall_score = -1.0
    
    # Run 3 trials to increase robustness
    for trial in range(3):
        # Initialize with spiral
        init_pts = _init_spiral(N)
        
        # Add small random noise to break symmetry and avoid local minima specific to the spiral
        noise_scale = 0.01
        init_pts += np.random.normal(0, noise_scale, init_pts.shape)
        
        # Ensure points are strictly inside the unit circle
        norms = np.linalg.norm(init_pts, axis=1)
        mask = norms > 0.999999
        if np.any(mask):
            init_pts[mask] = init_pts[mask] / norms[mask, np.newaxis] * 0.999999
            
        # Run Simulated Annealing
        optimized_pts = _simulate_annealing(
            init_pts, 
            max_iter=100000, 
            T_start=0.1, 
            T_end=1e-6, 
            cooling_rate=0.9995
        )
        
        score = _score(optimized_pts)
        
        if score > best_overall_score:
            best_overall_score = score
            best_overall_pts = optimized_pts.copy()
            
    # Final validity check
    if best_overall_pts is None:
        # Fallback to spiral if something went wrong
        best_overall_pts = _init_spiral(N)
        
    norms = np.linalg.norm(best_overall_pts, axis=1)
    mask = norms >= 1.0
    if np.any(mask):
        best_overall_pts[mask] = best_overall_pts[mask] / norms[mask, np.newaxis] * 0.999999
        
    return best_overall_pts
# EVOLVE-BLOCK-END
