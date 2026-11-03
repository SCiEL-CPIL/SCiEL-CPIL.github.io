import numpy as np
import time

N = 50

def _compute_radius(points):
    """
    Computes the maximum radius for the given centers.
    r = min( min_{i<j} ||c_i - c_j|| / 2, min_i (1 - ||c_i||) )
    """
    if len(points) == 0:
        return 0.0
    
    # Distance from boundary
    norms = np.linalg.norm(points, axis=1)
    r_boundary = np.min(1.0 - norms)
    
    if r_boundary < 0:
        return 0.0 # Invalid, at least one point is outside

    # Pairwise distances
    # Using broadcasting to compute all pairwise distances
    # diff shape (N, N, 2)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Exclude self-distances (diagonal)
    np.fill_diagonal(dists, np.inf)
    
    min_pair_dist = np.min(dists)
    r_pairs = min_pair_dist / 2.0
    
    return min(r_boundary, r_pairs)

def _initialize_hexagon():
    """
    Initializes points using a hexagonal lattice pattern clipped to the unit circle.
    """
    # Generate a grid of hexagonal points
    # Spacing needs to be small enough to get >= N points inside
    # Radius of circle is 1. Area ~ pi.
    # Area of hex cell ~ (sqrt(3)/2) * s^2.
    # Let's try spacing s such that we have plenty of points.
    
    # Start with a reasonable spacing
    spacing = 0.25
    
    while True:
        # Generate points in a hexagonal grid
        # y goes from -R to R
        # x offset depends on row
        k = int(2.0 / spacing) + 2
        points = []
        for j in range(-k, k + 1):
            y = j * spacing * np.sqrt(3) / 2.0
            offset = (j % 2) * spacing / 2.0
            for i in range(-k, k + 1):
                x = (i + offset) * spacing
                if x*x + y*y <= 1.0:
                    points.append([x, y])
        
        points = np.array(points)
        
        # Filter to keep only points strictly inside (with margin) to ensure initial validity
        # Actually, we can keep them if they are inside, but we need to ensure we have at least N.
        # If we have more than N, we select the N that are "best" (closest to center? or random?)
        # For hex lattice, all are equivalent locally.
        
        if len(points) >= N:
            # Select N points. 
            # To maximize initial radius, we might want to pick a subset that is well-distributed.
            # Simplest: just take the first N or random N.
            # Better: pick N points that maximize the minimum distance among themselves? 
            # That's an NP-hard problem. 
            # Heuristic: pick points closest to the center? No, that clusters them.
            # Pick points on a spiral? 
            # Let's just take a random subset of size N from the valid lattice points.
            # Or, better, sort by distance from center and take a mix?
            # Actually, the lattice itself is a good packing. 
            # If we have > N points, we can just take the first N. 
            # But to be safe, let's shuffle and take N.
            idx = np.random.choice(len(points), N, replace=False)
            return points[idx]
        
        spacing *= 0.95
        if spacing < 0.01:
            # Fallback: just place them in a grid
            x = np.linspace(-0.9, 0.9, 10)
            y = np.linspace(-0.9, 0.9, 10)
            X, Y = np.meshgrid(x, y)
            pts = np.column_stack((X.ravel(), Y.ravel()))
            mask = pts[:, 0]**2 + pts[:, 1]**2 <= 1.0
            pts = pts[mask]
            if len(pts) >= N:
                return pts[:N]
            else:
                # Should not happen
                return np.random.uniform(-0.9, 0.9, (N, 2))

def construct():
    start_time = time.time()
    time_limit = 150.0 # 150s wall clock limit, keep safety margin
    
    # 1. Initialization
    points = _initialize_hexagon()
    best_points = points.copy()
    best_radius = _compute_radius(best_points)
    
    # 2. Simulated Annealing
    # Parameters
    T_start = 0.05
    T_end = 0.0001
    alpha = 0.9999 # Cooling rate
    
    current_points = points.copy()
    current_radius = best_radius
    
    # Pre-calculate some things for speed? 
    # We will recompute radius every step.
    
    steps = 0
    while time.time() - start_time < time_limit * 0.9: # Leave 10% for polish/final checks
        # Generate a move: perturb one random point
        i = np.random.randint(N)
        # Step size decreases with temperature
        step_size = T_start * np.sqrt(2.0 / N) # Rough heuristic
        # Actually, let's scale step size with T
        step = T_start * 0.1
        delta = np.random.normal(0, step, 2)
        
        new_points = current_points.copy()
        new_points[i] += delta
        
        # Check if new point is inside unit circle
        # If not, reject or push back? 
        # Pushing back changes the distance to others. 
        # Simpler: reject if outside.
        if np.linalg.norm(new_points[i]) > 1.0:
            continue
            
        new_radius = _compute_radius(new_points)
        
        # Acceptance criterion
        if new_radius > current_radius:
            current_points = new_points
            current_radius = new_radius
            if current_radius > best_radius:
                best_radius = current_radius
                best_points = current_points.copy()
        else:
            # Metropolis criterion
            # delta_E = current_radius - new_radius (positive if worse)
            # We want to maximize radius, so "energy" is -radius.
            # delta_E = new_radius - current_radius (negative if worse)
            # Accept with prob exp(delta_E / T)
            if new_radius < current_radius:
                delta = new_radius - current_radius
                if np.random.rand() < np.exp(delta / T_start):
                    current_points = new_points
                    current_radius = new_radius
            
        T_start *= alpha
        steps += 1
        
    # 3. Polish Phase
    # Use a more precise local search or continue SA with very small steps
    # Let's do a "Nelder-Mead" style hill climbing on the best points found.
    # Or simply run SA with very low T for the remaining time.
    
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 1.0:
        # Refine best_points
        current_points = best_points.copy()
        current_radius = best_radius
        T_refine = 0.001
        
        while time.time() - start_time < time_limit * 0.98: # 2% safety margin
            i = np.random.randint(N)
            step = T_refine * 0.01
            delta = np.random.normal(0, step, 2)
            
            new_points = current_points.copy()
            new_points[i] += delta
            
            if np.linalg.norm(new_points[i]) > 1.0:
                continue
                
            new_radius = _compute_radius(new_points)
            
            if new_radius > current_radius:
                current_points = new_points
                current_radius = new_radius
                if current_radius > best_radius:
                    best_radius = current_radius
                    best_points = current_points.copy()
            else:
                # Accept worse moves with very low probability to escape tiny local minima
                delta_val = new_radius - current_radius
                if np.random.rand() < np.exp(delta_val / T_refine):
                    current_points = new_points
                    current_radius = new_radius
                    
            T_refine *= 0.999
            
    return best_points

# Run a quick self-test if executed directly (optional, but good for debugging)
if __name__ == "__main__":
    pts = construct()
    print("Radius:", _compute_radius(pts))
