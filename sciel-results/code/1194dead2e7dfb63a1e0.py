import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = 55.0 # Safety margin
    
    # Fast scoring function
    def score(points):
        # points: list of (x, y)
        # Convert to numpy array
        P = np.array(points, dtype=np.float64)
        n = len(P)
        
        # Generate all triples i < j < k
        # This is efficient for n=30
        idx = np.arange(n)
        i, j, k = np.meshgrid(idx, idx, idx, indexing='ij')
        mask = (i < j) & (j < k)
        ii, jj, kk = i[mask], j[mask], k[mask]
        
        # Calculate doubled areas
        # Area = 0.5 * | x_i(y_j - y_k) - y_i(x_j - x_k) + x_j(y_k - y_i) - y_j(x_k - x_i) |
        # Doubled Area = | ... |
        # Using vectorized operations on the selected points
        Pi = P[ii]
        Pj = P[jj]
        Pk = P[kk]
        
        # Cross product of (Pj - Pi) and (Pk - Pi)
        v1 = Pj - Pi
        v2 = Pk - Pi
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = np.abs(cross)
        
        # Exclude zero area
        areas = areas[areas > 0]
        
        if len(areas) == 0:
            return 0
            
        # Count multiplicities
        # Since areas are integers (doubled), we can cast to int if they are close
        # But due to float errors, maybe just use unique on rounded values?
        # Coordinates are integers, so doubled areas are integers.
        areas_int = np.rint(areas).astype(np.int64)
        
        vals, counts = np.unique(areas_int, return_counts=True)
        return np.max(counts)
        
    # Initial configurations
    # 1. The seed grid
    seed_grid = [(x, y) for x in range(6) for y in range(5)]
    # Shift to center? The problem says [-30, 30]. The seed is [0, 5]x[0, 4].
    # It's valid.
    
    # 2. Random clustered points
    def random_cluster():
        pts = []
        # Pick a random center
        cx = random.randint(-10, 10)
        cy = random.randint(-10, 10)
        # Generate 30 points in a small box around center
        for _ in range(30):
            x = cx + random.randint(-5, 5)
            y = cy + random.randint(-5, 5)
            x = max(-30, min(30, x))
            y = max(-30, min(30, y))
            p = (x, y)
            if p not in pts:
                pts.append(p)
        # If duplicates removed reduced count, fill with randoms
        while len(pts) < 30:
            x = random.randint(-30, 30)
            y = random.randint(-30, 30)
            p = (x, y)
            if p not in pts:
                pts.append(p)
        return pts

    candidates = []
    candidates.append(seed_grid)
    for _ in range(5):
        candidates.append(random_cluster())
        
    best_score = 0
    best_points = None
    
    for pts in candidates:
        s = score(pts)
        if s > best_score:
            best_score = s
            best_points = pts[:]
            
    # Local Search
    current_points = best_points[:]
    current_score = best_score
    
    T = 1.0
    T_min = 0.01
    cooling = 0.99
    
    # We will do random moves
    # To speed up, we can pre-compute some things? No, score is O(N^3) but N=30 is small.
    # Score takes ~1ms. 50s / 1ms = 50,000 iterations. This is a lot.
    
    while time.time() < deadline:
        # Pick a random point to move
        idx = random.randint(0, 29)
        old_x, old_y = current_points[idx]
        
        # Propose new position
        # Local move: small step
        # Or global move: random in box
        # Let's mix. 80% local, 20% global.
        if random.random() < 0.8:
            new_x = old_x + random.randint(-2, 2)
            new_y = old_y + random.randint(-2, 2)
        else:
            new_x = random.randint(-30, 30)
            new_y = random.randint(-30, 30)
            
        new_x = max(-30, min(30, new_x))
        new_y = max(-30, min(30, new_y))
        
        new_point = (new_x, new_y)
        
        # Check distinctness
        if new_point != (old_x, old_y):
            # Check if new_point is already in current_points (excluding idx)
            if new_point in current_points[:idx] + current_points[idx+1:]:
                continue # Collision, skip
                
        # Create new list
        new_points = current_points[:]
        new_points[idx] = new_point
        
        new_score = score(new_points)
        
        if new_score >= current_score:
            current_points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = current_points[:]
        else:
            # Accept with probability
            delta = new_score - current_score
            if random.random() < np.exp(delta / T):
                current_points = new_points
                current_score = new_score
                
        T *= cooling
        if T < T_min:
            T = T_min
            # Optional: restart with new random cluster if stuck?
            # For now, just keep going.

    return best_points
