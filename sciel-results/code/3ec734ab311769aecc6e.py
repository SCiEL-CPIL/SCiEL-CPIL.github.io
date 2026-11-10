import numpy as np
import time
from scipy.spatial import Delaunay

N = 200

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # 1. Initialization: Hexagonal Lattice
    # Generate a hexagonal lattice of points
    # Spacing s. Radius approx s/2 for hex packing in bulk.
    # We want ~200 points in a 1x1 square.
    # Area per point in hex packing ~ 0.866 * s^2.
    # 200 * 0.866 * s^2 ~ 1 => s ~ 1/sqrt(173) ~ 0.076
    # Let's try to generate a grid and filter.
    
    def generate_hex_grid(num_points, square_size=1.0, seed=0):
        rng = np.random.RandomState(seed)
        # Estimate spacing
        s = np.sqrt(square_size / (num_points * 0.866))
        # Generate points in a bounding box slightly larger
        margin = 0.5
        x_min, x_max = -0.5 - margin, 0.5 + margin
        y_min, y_max = -0.5 - margin, 0.5 + margin
        
        # Hex lattice spacing
        dx = s
        dy = s * np.sqrt(3) / 2
        
        # Create grid
        x_coords = np.arange(x_min, x_max, dx)
        y_coords = np.arange(y_min, y_max, dy)
        
        points = []
        for i, y in enumerate(y_coords):
            offset = 0 if i % 2 == 0 else dx / 2
            for x in x_coords:
                points.append((x + offset, y))
        
        points = np.array(points)
        
        # Filter points inside the square [-0.5, 0.5]
        mask = (points[:, 0] >= -0.5) & (points[:, 0] <= 0.5) & (points[:, 1] >= -0.5) & (points[:, 1] <= 0.5)
        points = points[mask]
        
        # Shuffle and take N
        if len(points) > num_points:
            indices = rng.choice(len(points), num_points, replace=False)
            points = points[indices]
        else:
            # If not enough, add random points
            while len(points) < num_points:
                new_p = rng.uniform(-0.5, 0.5, size=(1, 2))
                points = np.vstack([points, new_p])
                
        return points

    # Try a few initial seeds to find a good local optimum
    best_points = None
    best_radius = 0.0
    best_time = 0.0
    
    # We will run the optimization for a few different initial configurations
    # or just one very good one. Given the time limit, one robust run is better.
    # Let's use a jittered square grid as a fallback if hex fails, but hex is better.
    
    # Let's use a single robust initial guess: Hexagonal with slight jitter
    initial_points = generate_hex_grid(N, seed=42)
    
    # Optimization Loop
    current_points = initial_points.copy()
    current_radius = 0.0
    stagnant_iterations = 0
    max_stagnant = 50
    step_size = 0.01
    
    # Pre-allocate for forces
    forces = np.zeros_like(current_points)
    
    # Helper to calculate radius and validity
    def calc_radius(pts):
        # Check bounds
        min_coord = np.min(pts)
        max_coord = np.max(pts)
        if min_coord < -0.5 or max_coord > 0.5:
            return 0.0 # Invalid
        
        # Wall distances
        # Distance to nearest wall for each point
        # Walls at -0.5 and 0.5
        d_walls = np.minimum(
            np.minimum(pts[:, 0] + 0.5, 0.5 - pts[:, 0]),
            np.minimum(pts[:, 1] + 0.5, 0.5 - pts[:, 1])
        )
        min_wall = np.min(d_walls)
        
        # Pairwise distances
        # Use Delaunay to find neighbors
        try:
            tri = Delaunay(pts)
            neighbors = tri.simplices
            # Get unique edges
            edges = set()
            for simplex in neighbors:
                for i in range(3):
                    j = (i + 1) % 3
                    edge = tuple(sorted([simplex[i], simplex[j]]))
                    edges.add(edge)
            
            min_dist = np.inf
            for i, j in edges:
                d = np.linalg.norm(pts[i] - pts[j])
                if d < min_dist:
                    min_dist = d
            
            min_pair = min_dist / 2.0
        except Exception:
            # Fallback to full O(N^2) if Delaunay fails (e.g., collinear points)
            dists = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
            np.fill_diagonal(dists, np.inf)
            min_pair = np.min(dists) / 2.0
        
        return min(min_wall, min_pair)

    # Initial evaluation
    current_radius = calc_radius(current_points)
    best_radius = current_radius
    best_points = current_points.copy()
    
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        
        # 1. Compute Delaunay and Neighbors
        try:
            tri = Delaunay(current_points)
            simplices = tri.simplices
        except Exception:
            # If Delaunay fails, perturb and retry
            current_points += np.random.normal(0, 0.001, size=current_points.shape)
            current_points = np.clip(current_points, -0.5, 0.5)
            continue
        
        # 2. Identify Constraints and Compute Forces
        forces = np.zeros_like(current_points)
        
        # Wall Constraints
        # d_walls[i] = min distance to wall
        d_walls = np.minimum(
            np.minimum(current_points[:, 0] + 0.5, 0.5 - current_points[:, 0]),
            np.minimum(current_points[:, 1] + 0.5, 0.5 - current_points[:, 1])
        )
        
        # Find current min radius to determine "active" constraints
        # We need min_pair to know the global min radius
        # We already have d_walls. Let's find min_pair efficiently from Delaunay edges.
        min_pair_dist = np.inf
        edge_list = []
        for simplex in simplices:
            for i in range(3):
                j = (i + 1) % 3
                edge = tuple(sorted([simplex[i], simplex[j]]))
                if edge not in edge_list:
                    edge_list.append(edge)
                    d = np.linalg.norm(current_points[edge[0]] - current_points[edge[1]])
                    if d < min_pair_dist:
                        min_pair_dist = d
        
        min_pair_radius = min_pair_dist / 2.0
        current_r = min(np.min(d_walls), min_pair_radius)
        
        # Update best
        if current_r > best_radius:
            best_radius = current_r
            best_points = current_points.copy()
            stagnant_iterations = 0
            # Increase step size slightly if improving? Or keep constant.
        else:
            stagnant_iterations += 1
        
        # If stuck, perturb
        if stagnant_iterations > max_stagnant:
            # Random kick
            current_points += np.random.normal(0, 0.005, size=current_points.shape)
            current_points = np.clip(current_points, -0.5, 0.5)
            stagnant_iterations = 0
            continue
        
        # Define Target Radius for Pushing
        # We want to increase the minimum radius.
        # Push points that are close to the current minimum.
        # Let target_r = current_r * 1.01 (try to expand by 1%)
        target_r = current_r * 1.02
        
        # Apply Wall Forces
        # If d_wall < target_r, push inwards
        wall_violation = target_r - d_walls
        active_walls = wall_violation > 0
        
        for i in range(N):
            if active_walls[i]:
                # Direction to push: towards center
                # x component
                if current_points[i, 0] < 0:
                    forces[i, 0] += 1.0
                else:
                    forces[i, 0] -= 1.0
                # y component
                if current_points[i, 1] < 0:
                    forces[i, 1] += 1.0
                else:
                    forces[i, 1] -= 1.0
        
        # Apply Pair Forces
        for i, j in edge_list:
            d = np.linalg.norm(current_points[i] - current_points[j])
            if d < 2 * target_r:
                # Push apart
                # Vector from j to i
                vec = current_points[i] - current_points[j]
                if d > 1e-9:
                    vec = vec / d
                else:
                    vec = np.array([1.0, 0.0]) # Random if coincident
                
                # Force on i is +vec, on j is -vec
                # Weight by violation magnitude?
                violation = 2 * target_r - d
                forces[i] += vec * violation
                forces[j] -= vec * violation
        
        # Normalize forces to prevent huge steps
        # We want the max displacement to be around step_size
        max_force = np.max(np.linalg.norm(forces, axis=1))
        if max_force > 1e-9:
            forces = forces / max_force * step_size
        
        # Update positions
        current_points += forces
        # Project back to square
        current_points = np.clip(current_points, -0.5, 0.5)
        
        # Optional: Small random jitter to escape symmetry
        if iteration % 100 == 0:
             current_points += np.random.normal(0, 0.0005, size=current_points.shape)
             current_points = np.clip(current_points, -0.5, 0.5)

    return best_points

