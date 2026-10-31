import numpy as np

def construct():
    N = 16
    D = 2
    
    # Initialize with a 4x4 grid
    side = 4
    x = np.arange(side)
    y = np.arange(side)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack((xx.ravel(), yy.ravel()))
    
    # Center the points to 0
    points -= points.mean(axis=0)
    
    def get_score(pts):
        # Compute all pairwise distances
        # pts shape (N, D)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # Mask diagonal
        np.fill_diagonal(dists, np.inf)
        min_d = np.min(dists)
        max_d = np.max(dists)
        if min_d == 0:
            return np.inf
        return (max_d / min_d) ** 2
        
    best_points = points.copy()
    best_score = get_score(best_points)
    current_points = points.copy()
    current_score = best_score
    
    # Simulated Annealing parameters
    T = 1.0
    T_min = 0.001
    cooling_rate = 0.99
    iterations = 20000
    
    for i in range(iterations):
        if T < T_min:
            break
            
        # Pick random point
        idx = np.random.randint(N)
        original = current_points[idx].copy()
        
        # Perturb
        noise = np.random.normal(0, 0.1, D)
        current_points[idx] += noise
        
        new_score = get_score(current_points)
        
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_points = current_points.copy()
        else:
            # Accept with probability exp(-delta / T)
            delta = new_score - current_score
            prob = np.exp(-delta / T)
            if np.random.rand() < prob:
                current_score = new_score
            else:
                # Revert
                current_points[idx] = original
        
        T *= cooling_rate

    return best_points
