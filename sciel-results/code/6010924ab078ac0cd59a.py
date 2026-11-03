"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    """Construct 50 equal circles inside a unit circle to maximize the common radius."""
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit
    
    def hexagonal_init(n):
        """Generate initial hexagonal lattice positions."""
        points = []
        # Generate a hexagonal lattice and select the n closest to origin
        max_r = 2.0
        spacing = 0.2
        while len(points) < n + 50:
            k = int(max_r / spacing) + 2
            for j in range(-k, k + 1):
                for i in range(-k, k + 1):
                    x = (i + 0.5 * (j % 2)) * spacing
                    y = j * spacing * np.sqrt(3.0) / 2.0
                    if x*x + y*y <= max_r*max_r:
                        points.append([x, y])
            max_r *= 1.1
        
        points = np.array(points)
        # Sort by distance from origin and take the closest n
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)
        points = points[idx[:n]]
        
        # Scale to fit within unit circle with some margin
        max_dist = np.max(np.linalg.norm(points, axis=1))
        if max_dist > 0:
            scale = 0.95 / max_dist
            points *= scale
        
        return points

    def compute_radius(positions):
        """Compute the maximum possible radius for given positions."""
        # Distance from each center to boundary
        boundary_dists = 1.0 - np.linalg.norm(positions, axis=1)
        
        # Minimum pairwise distance
        min_pair_dist = np.inf
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(positions[i] - positions[j])
                if d < min_pair_dist:
                    min_pair_dist = d
        
        return min(min_pair_dist / 2.0, np.min(boundary_dists))

    def objective(x):
        """Objective function: minimize negative radius (i.e., maximize radius)."""
        positions = x.reshape(N, 2)
        # Distance from each center to boundary
        boundary_dists = 1.0 - np.linalg.norm(positions, axis=1)
        
        # Minimum pairwise distance
        min_pair_dist = np.inf
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(positions[i] - positions[j])
                if d < min_pair_dist:
                    min_pair_dist = d
        
        r = min(min_pair_dist / 2.0, np.min(boundary_dists))
        return -r

    def objective_smooth(x):
        """Smooth objective using log-sum-exp for soft-min."""
        positions = x.reshape(N, 2)
        
        # Boundary terms
        boundary_dists = 1.0 - np.linalg.norm(positions, axis=1)
        
        # Pairwise distances
        pair_dists = []
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(positions[i] - positions[j])
                pair_dists.append(d)
        pair_dists = np.array(pair_dists)
        
        # Soft-min using log-sum-exp
        beta = 50.0
        soft_min_boundary = -np.log(np.sum(np.exp(-beta * boundary_dists))) / beta
        soft_min_pair = -np.log(np.sum(np.exp(-beta * pair_dists))) / beta
        
        r = min(soft_min_pair / 2.0, soft_min_boundary)
        return -r

    # Try multiple initializations and optimization methods
    best_positions = None
    best_radius = 0.0
    
    # Initialization 1: Hexagonal lattice
    init1 = hexagonal_init(N)
    
    # Initialization 2: Random points within unit circle
    rng = np.random.default_rng(42)
    init2 = rng.random((N, 2)) * 2 - 1
    dists = np.linalg.norm(init2, axis=1)
    mask = dists > 1.0
    init2[mask] = init2[mask] / dists[mask, None] * 0.95
    
    # Initialization 3: Ring-based initialization
    init3 = np.zeros((N, 2))
    # Center
    init3[0] = [0, 0]
    # First ring: 6 points
    for i in range(1, 7):
        angle = 2 * np.pi * i / 6
        init3[i] = [0.3 * np.cos(angle), 0.3 * np.sin(angle)]
    # Second ring: 12 points
    for i in range(7, 19):
        angle = 2 * np.pi * (i - 6) / 12
        init3[i] = [0.6 * np.cos(angle), 0.6 * np.sin(angle)]
    # Third ring: 12 points
    for i in range(19, 31):
        angle = 2 * np.pi * (i - 18) / 12
        init3[i] = [0.85 * np.cos(angle), 0.85 * np.sin(angle)]
    # Fourth ring: 19 points
    for i in range(31, 50):
        angle = 2 * np.pi * (i - 30) / 19
        init3[i] = [0.95 * np.cos(angle), 0.95 * np.sin(angle)]
    
    initializations = [init1, init2, init3]
    
    for init_idx, init in enumerate(initializations):
        if time.time() > deadline - 30.0:
            break
        
        # Use L-BFGS-B for optimization
        x0 = init.flatten()
        
        # First try with standard objective
        result = minimize(objective, x0, method='L-BFGS-B', 
                         options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12})
        
        if result.success or result.fun < 0:
            positions = result.x.reshape(N, 2)
            r = -result.fun
            if r > best_radius:
                best_radius = r
                best_positions = positions.copy()
        
        # Try with smooth objective if time permits
        if time.time() > deadline - 20.0:
            continue
        
        result2 = minimize(objective_smooth, x0, method='L-BFGS-B',
                          options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12})
        
        if result2.success or result2.fun < 0:
            positions = result2.x.reshape(N, 2)
            r = -result2.fun
            if r > best_radius:
                best_radius = r
                best_positions = positions.copy()
        
        # Local refinement with tighter tolerances
        if time.time() > deadline - 10.0:
            break
        
        if best_positions is not None and init_idx == 0:
            # Use the best found so far for further refinement
            x_refined = best_positions.flatten()
            result3 = minimize(objective, x_refined, method='L-BFGS-B',
                              options={'maxiter': 10000, 'ftol': 1e-18, 'gtol': 1e-15})
            if result3.fun < 0:
                positions = result3.x.reshape(N, 2)
                r = -result3.fun
                if r > best_radius:
                    best_radius = r
                    best_positions = positions.copy()
    
    # Final check: ensure all circles are within the unit circle
    if best_positions is None:
        best_positions = init1
    
    # Verify and clip if necessary
    dists = np.linalg.norm(best_positions, axis=1)
    for i in range(N):
        if dists[i] >= 1.0:
            best_positions[i] *= 0.99 / dists[i]
    
    # Compute final radius for verification
    final_radius = compute_radius(best_positions)
    
    return best_positions
# EVOLVE-BLOCK-END
