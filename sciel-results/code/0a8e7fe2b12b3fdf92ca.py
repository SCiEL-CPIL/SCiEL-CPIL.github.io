# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses Simulated Annealing to explore the configuration space.
    The points are centered and scaled to have max distance 1 at each step to fix gauge freedom.
    """
    n = 14
    d = 3
    
    def normalize(points):
        """Center points and scale such that max pairwise distance is 1."""
        # Center
        points = points - np.mean(points, axis=0, keepdims=True)
        
        # Calculate distances
        dists = pdist(points)
        if len(dists) == 0:
            return points
        d_max = np.max(dists)
        if d_max > 1e-10:
            points = points / d_max
        return points

    def get_score(points):
        """Calculate (d_min / d_max)^2."""
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        ratio = d_min / d_max
        return ratio * ratio

    best_score = 0.0
    best_points = None
    
    # Simulated Annealing parameters
    T_start = 0.05
    T_end = 0.0001
    n_steps = 5000
    n_restarts = 20
    
    # We use a fixed seed for reproducibility of the search process itself, 
    # but we will run multiple restarts with different initial conditions.
    
    for r in range(n_restarts):
        np.random.seed(42 + r)
        
        # Initialize with random points
        points = np.random.randn(n, d)
        points = normalize(points)
        
        current_score = get_score(points)
        if current_score > best_score:
            best_score = current_score
            best_points = points.copy()
            
        T = T_start
        
        for step in range(n_steps):
            # Decay temperature
            # Linear decay or exponential? Exponential is common.
            # T = T_start * (T_end / T_start) ** (step / n_steps)
            T = T_start * (T_end / T_start) ** (step / float(n_steps))
            
            # Propose a move: perturb a random point
            idx = np.random.randint(n)
            # Perturbation size could scale with T
            step_size = 0.1 * (1.0 + T) 
            perturbation = np.random.randn(d) * step_size
            
            new_points = points.copy()
            new_points[idx] += perturbation
            new_points = normalize(new_points)
            
            new_score = get_score(new_points)
            
            # Acceptance criterion
            delta = new_score - current_score
            if delta > 0:
                # Accept
                points = new_points
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = points.copy()
            else:
                # Accept with probability exp(delta / T)
                # delta is negative
                prob = np.exp(delta / T)
                if np.random.random() < prob:
                    points = new_points
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_points = points.copy()
                        
    if best_points is None:
        # Fallback
        np.random.seed(0)
        best_points = np.random.randn(n, d)
        best_points = normalize(best_points)
        
    return best_points

# EVOLVE-BLOCK-END
