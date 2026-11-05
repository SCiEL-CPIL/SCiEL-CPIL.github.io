import itertools
import random
import time
import math
import numpy as np

def construct():
    start_time = time.time()
    time_limit = 55.0  # Leave 5s buffer for safety
    
    # Initial configuration: 5x6 grid
    # Centered to allow movement
    # x in [-2, 2], y in [-3, 2] -> 5 * 6 = 30 points
    def initial_grid():
        pts = []
        for x in range(-2, 3):
            for y in range(-3, 3):
                pts.append((x, y))
        return pts

    current_points = initial_grid()
    best_points = current_points[:]
    
    def calculate_score(points):
        """Vectorized score calculation using NumPy"""
        pts = np.array(points, dtype=np.int32)
        n = len(pts)
        
        # Compute all pairwise differences
        # For each pair (i,j), compute vector from i to j
        # Then for each triple (i,j,k), compute determinant
        
        # More efficient: for each point i, compute determinants with all pairs (j,k) where j,k > i
        counts = {}
        p = pts
        
        for i in range(n - 2):
            # Vectors from point i to all other points
            dx = p[i+1:, 0] - p[i, 0]  # shape (n-i-1,)
            dy = p[i+1:, 1] - p[i, 1]  # shape (n-i-1,)
            
            # For each pair (j,k) where j,k > i, compute determinant
            # det = dx_j * dy_k - dy_j * dx_k
            # We can compute all pairwise determinants at once
            
            # Create outer product matrices
            # dx[:, None] * dy[None, :] gives dx_j * dy_k
            # dy[:, None] * dx[None, :] gives dy_j * dx_k
            
            det_matrix = np.abs(dx[:, None] * dy[None, :] - dy[:, None] * dx[None, :])
            
            # Only consider upper triangle (j < k in the subarray)
            rows, cols = np.triu_indices(n - i - 1, k=1)
            dets = det_matrix[rows, cols]
            
            # Filter out zero determinants
            nonzero_dets = dets[dets > 0]
            
            if len(nonzero_dets) > 0:
                # Count occurrences
                unique, counts_arr = np.unique(nonzero_dets, return_counts=True)
                for d, c in zip(unique, counts_arr):
                    d_int = int(d)
                    counts[d_int] = counts.get(d_int, 0) + int(c)
        
        if not counts:
            return 0
        return max(counts.values())

    current_score = calculate_score(current_points)
    best_score = current_score
    
    # Search parameters
    stuck_counter = 0
    max_stuck = 2000
    
    def try_polish(points, score):
        """Try all single-point moves to find local improvements"""
        improved = False
        for idx in range(30):
            x, y = points[idx]
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    new_x = x + dx
                    new_y = y + dy
                    if -30 <= new_x <= 30 and -30 <= new_y <= 30:
                        # Check distinctness
                        valid = True
                        for j in range(30):
                            if j != idx and points[j] == (new_x, new_y):
                                valid = False
                                break
                        if valid:
                            new_points = points[:]
                            new_points[idx] = (new_x, new_y)
                            new_score = calculate_score(new_points)
                            if new_score > score:
                                points = new_points
                                score = new_score
                                improved = True
        return points, score, improved
    
    while time.time() - start_time < time_limit:
        # Generate a move
        # 1. Move one point
        # 2. Swap two points
        move_type = random.random()
        
        new_points = current_points[:]
        
        if move_type < 0.7:
            # Move one point
            idx = random.randint(0, 29)
            x, y = new_points[idx]
            # Random step in [-3, 3]
            dx = random.randint(-3, 3)
            dy = random.randint(-3, 3)
            if dx == 0 and dy == 0:
                continue
            new_x = x + dx
            new_y = y + dy
            
            # Bounds check
            if -30 <= new_x <= 30 and -30 <= new_y <= 30:
                # Distinctness check
                # Check if (new_x, new_y) is already in new_points (excluding idx)
                valid = True
                for j in range(30):
                    if j != idx and new_points[j] == (new_x, new_y):
                        valid = False
                        break
                if valid:
                    new_points[idx] = (new_x, new_y)
                else:
                    continue
            else:
                continue
        else:
            # Swap two points
            i, j = random.sample(range(30), 2)
            new_points[i], new_points[j] = new_points[j], new_points[i]

        new_score = calculate_score(new_points)
        
        # Acceptance criteria
        if new_score > best_score:
            best_score = new_score
            best_points = new_points
            current_points = new_points
            current_score = new_score
            stuck_counter = 0
        elif new_score > current_score:
            # Accept improvement over current, even if not global best
            current_points = new_points
            current_score = new_score
            stuck_counter = 0
        else:
            # Accept with some probability to escape local minima?
            # Since the objective is discrete and we want MAX, 
            # accepting worse moves is risky but can help.
            # However, "max multiplicity" is a very sharp peak.
            # Let's try a simple hill climbing with random restarts if stuck.
            stuck_counter += 1
            
            if stuck_counter > max_stuck:
                # Restart from best with perturbation
                current_points = best_points[:]
                current_score = best_score
                stuck_counter = 0
                
                # Perturb best points
                for _ in range(5):
                    idx = random.randint(0, 29)
                    x, y = current_points[idx]
                    dx = random.randint(-5, 5)
                    dy = random.randint(-5, 5)
                    nx, ny = x + dx, y + dy
                    if -30 <= nx <= 30 and -30 <= ny <= 30:
                        # Check distinctness
                        valid = True
                        for j in range(30):
                            if j != idx and current_points[j] == (nx, ny):
                                valid = False
                                break
                        if valid:
                            current_points[idx] = (nx, ny)
                
                current_score = calculate_score(current_points)
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points[:]
        
        # Periodic polish pass
        if stuck_counter % 500 == 0 and stuck_counter > 0:
            polished_points, polished_score, improved = try_polish(best_points, best_score)
            if improved:
                best_score = polished_score
                best_points = polished_points
                current_points = polished_points
                current_score = polished_score
                stuck_counter = 0

    # Final polish pass
    if time.time() - start_time < time_limit - 5:
        polished_points, polished_score, _ = try_polish(best_points, best_score)
        if polished_score > best_score:
            best_score = polished_score
            best_points = polished_points

    return best_points
