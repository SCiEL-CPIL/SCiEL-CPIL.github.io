import numpy as np
import time

N = 50

def compute_radius(points):
    """
    Computes the maximum radius r such that circles of radius r centered at points
    are inside the unit circle and do not overlap.
    """
    # Distance to boundary: min(1 - |c_i|)
    dists_to_boundary = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(dists_to_boundary)
    
    if min_boundary <= 0:
        return 0.0 # Invalid, at least one circle is outside
    
    # Minimum distance between centers
    # Use scipy's cdist or manual numpy broadcasting for N=50 (fast enough)
    # N=50 -> 50x50 matrix, very fast
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Set diagonal to infinity to ignore self-distance
    np.fill_diagonal(dists, np.inf)
    
    min_dist = np.min(dists)
    
    # The radius is limited by half the minimum distance between centers
    # and the minimum distance to the boundary
    r = min(min_dist / 2.0, min_boundary)
    
    return r if r > 0 else 0.0

def initial_hexagonal():
    """
    Generates an initial set of points using a hexagonal lattice.
    """
    # Approximate spacing for 50 circles in unit circle
    # Area of unit circle = pi. Area per circle ~ pi/50.
    # Hexagonal packing density ~ 0.9069.
    # Let's just generate a dense grid and pick the best 50.
    
    # Generate points on a hex grid
    spacing = 0.2 # Start with a reasonable guess
    points = []
    
    # Range of x and y to cover unit circle
    # We need to cover radius 1.
    # Let's iterate over a square grid and convert to hex offsets
    max_k = 10
    for j in range(-max_k, max_k + 1):
        for i in range(-max_k, max_k + 1):
            x = i * spacing + (j % 2) * spacing * 0.5
            y = j * spacing * np.sqrt(3) / 2
            if x**2 + y**2 < 1.0:
                points.append([x, y])
                
    points = np.array(points)
    
    # We need exactly N points. 
    # Sort by distance from center? Or just take the first N?
    # Taking the central ones is usually a good start.
    dists = np.linalg.norm(points, axis=1)
    indices = np.argsort(dists)
    
    # We might have more than N points. Pick the N closest to center.
    if len(points) >= N:
        selected = points[indices[:N]]
    else:
        # If not enough, add random points or expand grid
        # For now, if we don't have enough, pad with random points near center
        needed = N - len(points)
        random_pts = np.random.uniform(-0.5, 0.5, size=(needed, 2))
        selected = np.vstack([points, random_pts])
        
    return selected

def anneal(initial_points, time_limit=140.0, seed=0):
    """
    Simulated Annealing to maximize the packing radius.
    """
    np.random.seed(seed)
    
    points = initial_points.copy()
    current_r = compute_radius(points)
    best_r = current_r
    best_points = points.copy()
    
    # Annealing parameters
    T = 0.01  # Initial temperature
    T_min = 1e-6
    cooling_rate = 0.999
    step_size = 0.05 # Initial step size
    
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        # Check for early termination if T is too low
        if T < T_min:
            break
            
        # Pick a random point to move
        idx = np.random.randint(N)
        
        # Generate a random move
        # Use Gaussian noise for smooth exploration
        move = np.random.normal(0, step_size, size=2)
        
        new_points = points.copy()
        new_points[idx] += move
        
        # Ensure the point doesn't fly too far away (optional, but good for stability)
        # If it goes outside, the radius becomes 0 or very small, which is heavily penalized.
        
        new_r = compute_radius(new_points)
        
        # Acceptance criterion
        if new_r > current_r:
            points = new_points
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_points = points.copy()
        else:
            # Metropolis criterion
            # delta_r = new_r - current_r (negative or zero)
            # prob = exp(delta_r / T)
            delta_r = new_r - current_r
            # If delta_r is very negative, prob is ~0
            if delta_r > -T * 10: # Optimization: skip if too unlikely
                prob = np.exp(delta_r / T)
                if np.random.random() < prob:
                    points = new_points
                    current_r = new_r
        
        # Cool down
        T *= cooling_rate
        
        # Optionally adjust step size
        # Keep step size constant or decrease slowly
        # step_size *= 0.9999
        
    return best_points, best_r

def construct():
    # Deadline guard: 150s total. Reserve 5s for overhead.
    deadline = time.time() + 145.0
    
    best_overall_r = 0.0
    best_overall_points = None
    
    # Strategy: Multiple restarts with different initializations
    # 1. Hexagonal lattice
    # 2. Ring-based initialization
    # 3. Random initialization (filtered)
    
    # Initialization 1: Hexagonal
    try:
        init_pts = initial_hexagonal()
        # Quick check
        r_check = compute_radius(init_pts)
        if r_check > 0:
            p, r = anneal(init_pts, time_limit=40.0, seed=42)
            if r > best_overall_r:
                best_overall_r = r
                best_overall_points = p
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_overall_points if best_overall_points is not None else initial_hexagonal()
        
    # Initialization 2: Rings
    # Place circles in concentric rings
    try:
        # Ring 0: 1 circle at center
        # Ring 1: 6 circles
        # Ring 2: 12 circles
        # Ring 3: 18 circles? Total 1+6+12+18 = 37. Need 13 more.
        # Let's just use a generic ring generator
        
        points = np.zeros((N, 2))
        points[0] = [0, 0]
        
        count = 1
        ring_radius = 0.15
        ring_step = 0.15
        
        while count < N:
            # Number of circles in this ring
            # Circumference = 2 * pi * r. Spacing ~ 2 * r_circle.
            # Approximate max circles = 2 * pi * r / (2 * 0.12) ~ pi * r / 0.12
            # Let's try a fixed number per ring for simplicity, or calculate
            n_in_ring = int(2 * np.pi * ring_radius / 0.25)
            if n_in_ring < 1:
                n_in_ring = 1
            
            for k in range(n_in_ring):
                if count >= N:
                    break
                angle = 2 * np.pi * k / n_in_ring
                points[count, 0] = ring_radius * np.cos(angle)
                points[count, 1] = ring_radius * np.sin(angle)
                count += 1
            
            ring_radius += ring_step
        
        # Adjust to fit exactly N if we overshot (we break in loop so it's fine)
        # If we undershoot, pad with random
        if count < N:
             # This shouldn't happen with increasing radii
             pass
             
        # Perturb slightly to break symmetry
        points += np.random.normal(0, 0.01, size=(N, 2))
        
        r_check = compute_radius(points)
        if r_check > 0:
             p, r = anneal(points, time_limit=40.0, seed=123)
             if r > best_overall_r:
                 best_overall_r = r
                 best_overall_points = p
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_overall_points if best_overall_points is not None else initial_hexagonal()

    # Initialization 3: Random
    try:
        # Generate random points, keep only valid ones?
        # Or just random points in unit circle.
        # Annealing will fix overlaps if T is high enough, but starting valid is better.
        # Let's try to generate a valid random packing using rejection sampling for the first few, then random.
        # Actually, just random points in the unit disk.
        
        pts = []
        while len(pts) < N:
            x, y = np.random.uniform(-1, 1, 2)
            if x**2 + y**2 < 1.0:
                pts.append([x, y])
                
        points = np.array(pts)
        
        # This start is likely invalid (overlaps). 
        # Annealing from an invalid state is hard if the objective is 0.
        # So we must ensure the start is valid or use a potential energy method.
        # My current anneal uses compute_radius which returns 0 for invalid.
        # If start is invalid, r=0. 
        # If we move, and it becomes valid, r>0, we accept.
        # If we move and it stays invalid, r=0. 
        # delta_r = 0 - 0 = 0. Prob = 1. We accept all moves within the "invalid" region.
        # This acts like a random walk until we hit a valid region.
        # This might be slow.
        
        # Better: Start with a slightly expanded hex lattice that is guaranteed valid?
        # Or just skip this if we are close to deadline.
        
        if time.time() < deadline - 40:
            # Try annealing from random
            p, r = anneal(points, time_limit=30.0, seed=999)
            if r > best_overall_r:
                best_overall_r = r
                best_overall_points = p
    except Exception:
        pass

    # Final Polish: Run a short, fine-tuned anneal on the best result found
    if best_overall_points is not None and time.time() < deadline - 5:
        # Use a very small step size and low temperature for polishing
        # We can reuse the anneal function with a tight time limit
        p, r = anneal(best_overall_points, time_limit=5.0, seed=0)
        if r > best_overall_r:
            best_overall_r = r
            best_overall_points = p
            
    return best_overall_points if best_overall_points is not None else initial_hexagonal()

# Keep the function name and signature as required
# The task says: "Keep the function names and signatures of the seed program."
# Seed program: def construct():
# It returns an array of shape (N, 2).

if __name__ == "__main__":
    # Simple test
    pts = construct()
    r = compute_radius(pts)
    print(f"Radius: {r}")
    print(f"Shape: {pts.shape}")
