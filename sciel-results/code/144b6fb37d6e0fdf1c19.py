"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    N = 30
    MIN_COORD = -30
    MAX_COORD = 30
    TIME_LIMIT = 50.0  # Leave 10s buffer for safety

    def calculate_max_multiplicity(pts):
        counts = {}
        # Pre-extract coordinates for speed
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        for i in range(N):
            x1, y1 = xs[i], ys[i]
            for j in range(i + 1, N):
                x2, y2 = xs[j], ys[j]
                for k in range(j + 1, N):
                    x3, y3 = xs[k], ys[k]
                    area = abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
                    if area > 0:
                        counts[area] = counts.get(area, 0) + 1
        if not counts:
            return 0
        return max(counts.values())

    def get_initial_grid():
        # Use the 5x6 grid centered, which is a good baseline
        # 5 rows, 6 cols
        rows, cols = 5, 6
        shift_x = -(cols - 1) // 2
        shift_y = -(rows - 1) // 2
        pts = []
        for x in range(cols):
            for y in range(rows):
                pts.append((x + shift_x, y + shift_y))
        return pts

    def random_valid_position(occupied_set):
        while True:
            x = random.randint(MIN_COORD, MAX_COORD)
            y = random.randint(MIN_COORD, MAX_COORD)
            if (x, y) not in occupied_set:
                return (x, y)

    # 1. Initialize
    initial_pts = get_initial_grid()
    # Ensure distinctness and bounds (grid guarantees this)
    
    current_pts = initial_pts[:]
    current_score = calculate_max_multiplicity(current_pts)
    
    best_pts = current_pts[:]
    best_score = current_score
    
    start_time = time.time()
    no_improve_count = 0
    max_no_improve = 100  # Restart if stuck for 100 moves
    
    while time.time() - start_time < TIME_LIMIT:
        # Pick a random point to move
        i = random.randint(0, N - 1)
        old_pos = current_pts[i]
        
        # Generate a new random position
        occupied = set(current_pts)
        occupied.remove(old_pos)
        new_pos = random_valid_position(occupied)
        
        # Create temporary points
        temp_pts = current_pts[:]
        temp_pts[i] = new_pos
        
        # Evaluate
        temp_score = calculate_max_multiplicity(temp_pts)
        
        if temp_score > current_score:
            # Accept
            current_pts = temp_pts
            current_score = temp_score
            no_improve_count = 0
            
            if current_score > best_score:
                best_score = current_score
                best_pts = current_pts[:]
        else:
            # Reject (Hill Climbing)
            no_improve_count += 1
            if no_improve_count >= max_no_improve:
                # Restart from best with a perturbation
                current_pts = best_pts[:]
                # Perturb a few points
                for _ in range(3):
                    idx = random.randint(0, N - 1)
                    occupied = set(current_pts)
                    occupied.remove(current_pts[idx])
                    current_pts[idx] = random_valid_position(occupied)
                current_score = calculate_max_multiplicity(current_pts)
                no_improve_count = 0
                # If this restart is worse, we'll just keep climbing from here
                # or if it's better, great.

    return best_pts
# EVOLVE-BLOCK-END
