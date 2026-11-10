import numpy as np
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    np.random.seed(42)
    
    def generate_hex_grid(n):
        # Estimate side length s for hex packing of n circles in area 1
        # Area per hex circle ~ sqrt(3)/2 * s^2
        s = np.sqrt(2.0 / (n * np.sqrt(3)))
        
        points = []
        # Iterate over potential rows
        # y = row * s * sqrt(3) / 2
        max_rows = int(1.0 / (s * np.sqrt(3) / 2)) + 2
        for row in range(max_rows):
            y = -0.5 + row * s * np.sqrt(3) / 2
            if y > 0.5:
                continue
            offset = s / 2 if row % 2 else 0
            max_cols = int(1.0 / s) + 2
            for col in range(max_cols):
                x = -0.5 + col * s + offset
                if -0.5 <= x <= 0.5:
                    points.append([x, y])
                    
        points = np.array(points)
        if len(points) < n:
            # Pad with random
            while len(points) < n:
                points = np.vstack([points, np.random.uniform(-0.5, 0.5, 2)])
        else:
            # Randomly select n points to break symmetry
            idx = np.random.choice(len(points), n, replace=False)
            points = points[idx]
        return points

    centers = generate_hex_grid(N)
    
    def get_score(c):
        # Vectorized calculation of minimum distance
        # Pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        min_pair = np.min(dist)
        
        # Wall distances
        wall_dist = 0.5 - np.abs(c)
        min_wall = np.min(wall_dist)
        
        return min(min_pair, min_wall)

    current_score = get_score(centers)
    best_score = current_score
    best_centers = centers.copy()
    
    T = 0.01
    cooling = 0.9995
    steps = 50000
    step_size = 0.01
    
    start_time = time.time()
    max_time = 110.0
    
    for step in range(steps):
        if time.time() - start_time > max_time:
            break
            
        i = np.random.randint(N)
        
        delta = np.random.normal(0, step_size, 2)
        
        new_centers = centers.copy()
        new_centers[i] += delta
        new_centers[i] = np.clip(new_centers[i], -0.5, 0.5)
        
        new_score = get_score(new_centers)
        
        if new_score > current_score:
            centers = new_centers
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_centers = centers.copy()
        else:
            if np.random.random() < np.exp((new_score - current_score) / T):
                centers = new_centers
                current_score = new_score
        
        T *= cooling
        if T < 1e-6:
            T = 1e-6

    return best_centers
# EVOLVE-BLOCK-END
