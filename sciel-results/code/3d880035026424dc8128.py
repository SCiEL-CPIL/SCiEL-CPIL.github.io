import numpy as np

N = 200

def construct():
    # Initial configuration: Random points inside the square
    # Or a grid to start closer to a good solution
    # Let's use a 14x14 grid (196 points) + 4 random, or just 200 random points
    # A grid is a better start.
    # 10x20 grid
    m, n = 10, 20
    xs = np.linspace(-0.5, 0.5, n)
    ys = np.linspace(-0.5, 0.5, m)
    points = np.array([(x, y) for y in ys for x in xs])
    
    # Run Simulated Annealing
    points = simulated_annealing(points)
    
    # Final local refinement (optional, SA should be good)
    # points = local_search(points)
    
    return points

def calculate_score(points):
    # Calculate the radius
    # min distance between centers / 2
    # min distance to boundary
    
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    # Distance to boundary
    dx_min = points[:, 0] + 0.5
    dx_max = 0.5 - points[:, 0]
    dy_min = points[:, 1] + 0.5
    dy_max = 0.5 - points[:, 1]
    min_boundary_dist = np.min(np.stack([dx_min, dx_max, dy_min, dy_max], axis=1), axis=1)
    
    global_min_center = np.min(dist)
    global_min_boundary = np.min(min_boundary_dist)
    
    r = min(global_min_center / 2.0, global_min_boundary)
    return r

def simulated_annealing(points, iterations=5000):
    points = points.copy()
    N = len(points)
    
    # Initial temperature and step size
    T = 0.01
    step_size = 0.05
    current_score = calculate_score(points)
    best_points = points.copy()
    best_score = current_score
    
    # Cooling schedule
    cooling_rate = 0.999
    step_decay = 0.999
    
    for i in range(iterations):
        # Pick a random point to move
        idx = np.random.randint(N)
        original_point = points[idx].copy()
        
        # Move by random vector
        move = np.random.normal(0, step_size, 2)
        points[idx] += move
        
        # Clip to square (or penalize if outside)
        # For now, clip to ensure validity
        points[idx, 0] = np.clip(points[idx, 0], -0.5, 0.5)
        points[idx, 1] = np.clip(points[idx, 1], -0.5, 0.5)
        
        new_score = calculate_score(points)
        
        # Acceptance criterion
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = points.copy()
        else:
            # Accept with probability exp((new - old) / T)
            delta = new_score - current_score
            prob = np.exp(delta / T)
            if np.random.random() < prob:
                current_score = new_score
            else:
                # Reject move
                points[idx] = original_point
        
        # Cool down
        T *= cooling_rate
        step_size *= step_decay
        
    return best_points
