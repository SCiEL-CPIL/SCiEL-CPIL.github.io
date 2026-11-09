import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) coordinates.

    """
    n = 16
    best_score = -np.inf
    best_points = None
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 500  # 500 seconds out of 600
    
    def objective(flat_points):
        """
        Minimize the negative of the normalized minimum distance.
        """
        points = flat_points.reshape((n, 2))
        # Calculate pairwise distances
        dists = pdist(points)
        d_max = np.max(dists)
        if d_max < 1e-10:
            return 0.0
        d_min = np.min(dists)
        # We want to maximize d_min / d_max
        # So minimize - (d_min / d_max)
        return - (d_min / d_max)

    def run_optimization(initial_points, max_iter=2000):
        """
        Run Nelder-Mead optimization from a given initial point.
        """
        x0 = initial_points.flatten()
        # Nelder-Mead is good for non-smooth functions
        result = minimize(objective, x0, method='Nelder-Mead', 
                          options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10})
        return result.x.reshape((n, 2)), result.fun

    # --- Initial Guess Generators ---

    def get_hex_cluster():
        """
        Generate points from a hexagonal lattice, taking the central 16.
        """
        # Hex lattice points
        pts = []
        # Center
        pts.append([0, 0])
        # Ring 1
        for i in range(6):
            angle = i * np.pi / 3
            pts.append([np.cos(angle), np.sin(angle)])
        # Ring 2
        # Standard hex ring 2 has 12 points.
        # We need 9 more to reach 16.
        # Let's generate all 12 and pick 9.
        # The 12 points are at distance sqrt(3) or 2?
        # In hex lattice with side 1:
        # Distances: 1, sqrt(3), 2, ...
        # Let's just generate a larger set and pick closest to center?
        # No, let's construct specific coordinates.
        
        # A better way: Generate a hex grid of radius 2 (in grid coords)
        # and convert to cartesian.
        # Grid coords (q, r)
        # x = q + r/2
        # y = r * sqrt(3)/2
        
        grid_pts = []
        for q in range(-2, 3):
            for r in range(-2, 3):
                # Hex distance
                if max(abs(q), abs(r), abs(-q-r)) <= 2:
                    x = q + r / 2.0
                    y = r * np.sqrt(3) / 2.0
                    grid_pts.append([x, y])
        
        grid_pts = np.array(grid_pts)
        # Sort by distance from origin
        dists_to_origin = np.linalg.norm(grid_pts, axis=1)
        indices = np.argsort(dists_to_origin)
        # Take the first 16
        selected = grid_pts[indices[:16]]
        return selected

    def get_square_grid():
        """
        4x4 grid.
        """
        x = np.linspace(-1, 1, 4)
        y = np.linspace(-1, 1, 4)
        X, Y = np.meshgrid(x, y)
        return np.column_stack((X.flatten(), Y.flatten()))

    def get_random():
        """
        Random points in a unit circle.
        """
        angles = np.random.uniform(0, 2 * np.pi, n)
        radii = np.sqrt(np.random.uniform(0, 1, n))
        return np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))

    def get_symmetric_ring():
        """
        1 center, 6 at r1, 9 at r2.
        """
        pts = [[0, 0]]
        # Ring 1
        for i in range(6):
            angle = i * np.pi / 3
            pts.append([0.5 * np.cos(angle), 0.5 * np.sin(angle)])
        # Ring 2 - 9 points
        # Start with 12, remove 3
        # Let's place 9 points evenly? 9 is not divisible by 6.
        # Let's just place 9 points at random angles on a circle of radius 1.5
        # Or use a subset of the 12.
        # Let's try 3 points at angles 0, 120, 240 and 6 points at 60, 180, 300...
        # Actually, let's just use random angles for the outer ring for diversity.
        angles = np.linspace(0, 2 * np.pi, 9, endpoint=False)
        for angle in angles:
            pts.append([1.5 * np.cos(angle), 1.5 * np.sin(angle)])
        return np.array(pts)

    # --- Main Loop ---
    
    # 1. Try specific structured guesses
    initial_guesses = [
        get_hex_cluster(),
        get_square_grid(),
        get_symmetric_ring()
    ]
    
    # Add some random guesses
    np.random.seed(42)
    for _ in range(20):
        initial_guesses.append(get_random())
        
    # If time permits, add more random guesses
    while time.time() < deadline - 60:
        initial_guesses.append(get_random())
        
    # Optimize each guess
    for i, guess in enumerate(initial_guesses):
        if time.time() > deadline - 10:
            break
            
        try:
            opt_points, score = run_optimization(guess)
            # score is negative of the ratio we want to maximize
            # So higher score (less negative) is better
            if score > best_score:
                best_score = score
                best_points = opt_points
        except Exception:
            pass
            
    # Final fallback
    if best_points is None:
        best_points = get_hex_cluster()
        
    return best_points
