"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def _get_radius(points):
    """
    Calculate the maximum common radius for the given points.
    Returns min(min_pair_dist/2, min_boundary_dist).
    """
    n = len(points)
    if n == 0:
        return 0.0
    
    # Pairwise distances
    # Using broadcasting for efficiency
    # diff shape (n, n, 2)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    # Set diagonal to infinity to exclude self-distance
    np.fill_diagonal(dist, np.inf)
    
    min_pair_dist = np.min(dist) / 2.0
    
    # Boundary distances
    norms = np.linalg.norm(points, axis=1)
    min_boundary_dist = np.min(1.0 - norms)
    
    return min(min_pair_dist, min_boundary_dist)

def _init_lattice():
    """
    Generate an initial configuration using a hexagonal lattice.
    """
    # Start with a reasonable spacing
    spacing = 0.25
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        # Filter points strictly inside to allow for some slack
        # We want to fit 50 points.
        mask = radius <= 1.0 - spacing / 2.0
        points = points[mask]
        
        if len(points) >= N:
            # Sort by distance to origin to keep the most central ones
            return points[np.argsort(radius[mask], kind="stable")[:N]].copy()
        spacing *= 0.95

def _init_random():
    """
    Generate a random initial configuration.
    """
    pts = []
    while len(pts) < N:
        # Rejection sampling in unit circle
        x = np.random.uniform(-1, 1)
        y = np.random.uniform(-1, 1)
        if x*x + y*y <= 1.0:
            pts.append([x, y])
    return np.array(pts)

def _sa_optimize(points, max_iters=5000, t_start=1.0, t_end=1e-5):
    """
    Simulated Annealing to maximize the minimum radius.
    """
    current_points = points.copy()
    current_r = _get_radius(current_points)
    
    best_points = current_points.copy()
    best_r = current_r
    
    # Pre-calculate cooling factor
    # t = t_start * (t_end / t_start) ** (i / max_iters)
    ratio = t_end / t_start
    
    for i in range(max_iters):
        # Calculate temperature
        t = t_start * (ratio ** (i / max_iters))
        
        # Choose a random point to move
        idx = np.random.randint(N)
        
        # Perturbation magnitude scales with temperature
        # As t decreases, moves become smaller
        step_size = 0.1 * t + 1e-4
        
        dx = np.random.normal(0, step_size)
        dy = np.random.normal(0, step_size)
        
        new_points = current_points.copy()
        new_points[idx] = [current_points[idx][0] + dx, current_points[idx][1] + dy]
        
        new_r = _get_radius(new_points)
        
        # Acceptance criterion
        if new_r > current_r:
            current_points = new_points
            current_r = new_r
            
            if current_r > best_r:
                best_r = current_r
                best_points = current_points.copy()
        else:
            # Calculate probability of accepting worse solution
            # Avoid division by zero or underflow
            if t > 1e-9:
                prob = np.exp((new_r - current_r) / t)
                if np.random.random() < prob:
                    current_points = new_points
                    current_r = new_r
            # If t is very small, prob is effectively 0, so reject
    
    return best_points, best_r

def _local_search(points, max_iters=2000):
    """
    Hill climbing local search to refine the solution.
    """
    current_points = points.copy()
    current_r = _get_radius(current_points)
    
    best_points = current_points.copy()
    best_r = current_r
    
    step_size = 0.05
    
    for i in range(max_iters):
        # Adaptive step size: decrease if no improvement
        # Simple version: constant step, but try smaller if stuck?
        # For now, just standard hill climbing with random restarts in direction
        
        improved = False
        for _ in range(10): # Try a few random moves per iteration
            idx = np.random.randint(N)
            dx = np.random.normal(0, step_size)
            dy = np.random.normal(0, step_size)
            
            new_points = current_points.copy()
            new_points[idx] = [current_points[idx][0] + dx, current_points[idx][1] + dy]
            
            new_r = _get_radius(new_points)
            
            if new_r > current_r:
                current_points = new_points
                current_r = new_r
                improved = True
                
                if current_r > best_r:
                    best_r = current_r
                    best_points = current_points.copy()
        
        if not improved:
            # Reduce step size
            step_size *= 0.8
            if step_size < 1e-5:
                break
                
    return best_points, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_points = None
    best_r = -1.0
    
    # 1. Start with lattice
    pts = _init_lattice()
    best_pts, best_r_curr = _sa_optimize(pts, max_iters=3000, t_start=1.0, t_end=1e-5)
    if best_r_curr > best_r:
        best_r = best_r_curr
        best_points = best_pts
        
    # Refine the best lattice-based solution
    if time.time() < deadline:
        refined_pts, refined_r = _local_search(best_points, max_iters=1000)
        if refined_r > best_r:
            best_r = refined_r
            best_points = refined_pts

    # 2. Try random restarts
    # Estimate how many restarts we can afford
    # Each SA run takes ~3000 * (cost of _get_radius)
    # _get_radius is ~1-2ms? 3000 * 2ms = 6s.
    # We have ~100s left. So ~15 restarts.
    
    num_restarts = 10
    for i in range(num_restarts):
        if time.time() > deadline - 10.0: # Leave 10s for final refinement
            break
        
        pts = _init_random()
        # Use slightly fewer iterations for random starts to allow more restarts
        # Or same? Random starts are harder. Let's do 2000 iters.
        best_pts, best_r_curr = _sa_optimize(pts, max_iters=2000, t_start=0.5, t_end=1e-5)
        
        if best_r_curr > best_r:
            best_r = best_r_curr
            best_points = best_pts
            
        # Refine if it's a new best
        if best_r_curr == best_r and time.time() < deadline - 5.0:
            refined_pts, refined_r = _local_search(best_points, max_iters=500)
            if refined_r > best_r:
                best_r = refined_r
                best_points = refined_pts

    # Final polish on the global best
    if best_points is not None and time.time() < deadline - 2.0:
        final_pts, final_r = _local_search(best_points, max_iters=2000)
        if final_r > best_r:
            best_r = final_r
            best_points = final_pts

    if best_points is None:
        # Fallback
        best_points = _init_lattice()
        
    return best_points

# EVOLVE-BLOCK-END
