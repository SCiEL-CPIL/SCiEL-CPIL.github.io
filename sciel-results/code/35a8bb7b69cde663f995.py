import numpy as np
from scipy.spatial.distance import pdist
import time

def min_max_dist_dim2_16() -> np.ndarray:
    n = 16
    
    def normalize(points):
        points = points - np.mean(points, axis=0)
        norm = np.linalg.norm(points)
        if norm > 1e-10:
            points = points / norm
        return points

    def score(points):
        p = normalize(points)
        dists = pdist(p)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-10 or d_min < 1e-10:
            return -np.inf
        return 2 * (np.log(d_min) - np.log(d_max))
        
    def run_sa(initial_points, time_limit):
        np.random.seed(12345) # Fixed seed for reproducibility within run? 
        # Actually, we want diversity. Let's use global random state.
        
        current_points = normalize(initial_points.copy())
        current_score = score(current_points)
        
        best_points = current_points.copy()
        best_score = current_score
        
        T = 0.1  # Initial temperature
        T_min = 1e-4
        cooling_rate = 0.99995
        
        start_time = time.time()
        
        while True:
            if time.time() - start_time > time_limit:
                break
                
            # Candidate
            candidate_points = current_points.copy()
            idx = np.random.randint(n)
            # Adaptive step size? Start with 0.2, decay?
            # Let's use a fixed step size for simplicity, or decay with T.
            step = 0.1 * (T / 0.1 + 0.1) # Roughly decay
            noise = np.random.randn(2) * step
            candidate_points[idx] += noise
            
            candidate_points = normalize(candidate_points)
            candidate_score = score(candidate_points)
            
            delta = candidate_score - current_score
            if delta > 0 or np.random.rand() < np.exp(delta / T):
                current_points = candidate_points
                current_score = candidate_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points.copy()
                    
            T *= cooling_rate
            if T < T_min:
                T = T_min
                
        return best_points, best_score

    # Generate diverse initial guesses
    np.random.seed(42)
    guesses = []
    
    # 1. Random
    guesses.append(np.random.randn(n, 2))
    
    # 2. 4x4 Grid
    x = np.arange(4) - 1.5
    y = np.arange(4) - 1.5
    xx, yy = np.meshgrid(x, y)
    guesses.append(np.column_stack((xx.ravel(), yy.ravel())))
    
    # 3. Hexagonal
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3)/2])
    hex_points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            p = i * v1 + j * v2
            hex_points.append(p)
    hex_points = np.array(hex_points)
    hex_points = hex_points - np.mean(hex_points, axis=0)
    dists_to_center = np.linalg.norm(hex_points, axis=1)
    idx = np.argsort(dists_to_center)[:16]
    guesses.append(hex_points[idx])
    
    # 4. Another Random
    guesses.append(np.random.randn(n, 2))
    
    # 5. Circle
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    guesses.append(np.column_stack((np.cos(angles), np.sin(angles))))
    
    best_overall_points = guesses[0]
    best_overall_score = -np.inf
    
    # Total time budget
    total_time = 580
    time_per_run = total_time / len(guesses)
    
    for g in guesses:
        pts, sc = run_sa(g, time_per_run)
        if sc > best_overall_score:
            best_overall_score = sc
            best_overall_points = pts
            
    return best_overall_points
