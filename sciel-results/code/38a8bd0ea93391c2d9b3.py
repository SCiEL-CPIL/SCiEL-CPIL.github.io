import time
import random
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Leave 10s margin
    
    N = 30
    min_coord = -30
    max_coord = 30
    
    # Initial configuration: 6x5 grid
    # Centered roughly at 0
    # x in -5..4, y in -4..4 -> range 10x9=90 > 61x61? No, 10x9 fits in 61x61.
    # Let's use a compact grid to start.
    # 6 cols, 5 rows.
    init_x = np.arange(-5, 1) # -5, -4, -3, -2, -1, 0
    init_y = np.arange(-4, 1) # -4, -3, -2, -1, 0
    points = [(x, y) for x in init_x for y in init_y]
    # Ensure distinctness (grid is distinct)
    # Ensure in bounds (yes)
    
    # Pre-compute indices for triples involving each point?
    # Or just iterate all triples for initial eval.
    
    def evaluate(points):
        counts = defaultdict(int)
        max_count = 0
        max_area = 0
        n = len(points)
        for i in range(n):
            xi, yi = points[i]
            for j in range(i + 1, n):
                xj, yj = points[j]
                for k in range(j + 1, n):
                    xk, yk = points[k]
                    # Determinant
                    area = abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
                    if area > 0:
                        counts[area] += 1
                        if counts[area] > max_count:
                            max_count = counts[area]
                            max_area = area
        return max_count, max_area, counts

    best_points = [p[:] for p in points]
    best_score, best_area_val, best_counts = evaluate(best_points)
    
    # Local Search
    # We will use a simple hill climbing with random restarts if stuck.
    # To speed up, we only update triangles involving the moved point.
    
    # Helper to get triangles involving index i
    # Triangles are (i, j, k) with i < j < k.
    # If we move point i, it affects:
    # 1. Triangles where i is the smallest index: (i, j, k) for j>i, k>j
    # 2. Triangles where i is the middle index: (j, i, k) for j<i, k>i
    # 3. Triangles where i is the largest index: (j, k, i) for j<k<i
    
    def get_affected_triangles(i, n):
        tris = []
        for j in range(n):
            if j == i: continue
            for k in range(j+1, n):
                if k == i: continue
                # Ensure unique representation (a,b,c) with a<b<c
                a, b, c = sorted([i, j, k])
                tris.append((a, b, c))
        return tris

    # This pre-computation is O(N^3) which is 27000, fine.
    affected_tris = [get_affected_triangles(i, N) for i in range(N)]
    
    def calculate_area(p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        return abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

    current_points = [p[:] for p in best_points]
    current_counts = defaultdict(int, best_counts)
    current_score = best_score
    current_max_area = best_area_val
    
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            # Restart with a small random perturbation of best_points
            current_points = [p[:] for p in best_points]
            for _ in range(5):
                idx = random.randint(0, N-1)
                dx, dy = random.randint(-2, 2), random.randint(-2, 2)
                nx, ny = current_points[idx][0] + dx, current_points[idx][1] + dy
                if min_coord <= nx <= max_coord and min_coord <= ny <= max_coord:
                    # Check collision
                    collides = False
                    for p in current_points:
                        if p[0] == nx and p[1] == ny:
                            if p is not current_points[idx]: # simple identity check might fail if lists, use index
                                # better check:
                                pass
                    # Proper collision check
                    for k in range(N):
                        if k != idx and current_points[k] == (nx, ny):
                            collides = True
                            break
                    if not collides:
                        current_points[idx] = (nx, ny)
            current_counts, current_score, current_max_area = evaluate(current_points)
            no_improve_count = 0
            continue

        # Pick a random point to move
        i = random.randint(0, N-1)
        # Pick a random move
        # Try small moves first
        dx = random.randint(-3, 3)
        dy = random.randint(-3, 3)
        if dx == 0 and dy == 0:
            continue
            
        old_x, old_y = current_points[i]
        new_x, new_y = old_x + dx, old_y + dy
        
        if new_x < min_coord or new_x > max_coord or new_y < min_coord or new_y > max_coord:
            continue
            
        # Check collision
        collides = False
        for k in range(N):
            if k != i and current_points[k] == (new_x, new_y):
                collides = True
                break
        if collides:
            continue
            
        # Calculate new areas for affected triangles
        new_p = (new_x, new_y)
        affected = affected_tris[i]
        
        # Store old areas to subtract
        old_areas = []
        for (a, b, c) in affected:
            area = calculate_area(current_points[a], current_points[b], current_points[c])
            if area > 0:
                old_areas.append(area)
                current_counts[area] -= 1
                if current_counts[area] == 0:
                    del current_counts[area]
                
        # Update point
        current_points[i] = new_p
        
        # Add new areas
        new_areas = []
        for (a, b, c) in affected:
            area = calculate_area(current_points[a], current_points[b], current_points[c])
            if area > 0:
                new_areas.append(area)
                current_counts[area] += 1
                
        # Find new max
        new_score = max(current_counts.values()) if current_counts else 0
        new_max_area = max(current_counts.keys(), key=lambda x: current_counts[x]) if current_counts else 0
        
        if new_score > current_score:
            current_score = new_score
            current_max_area = new_max_area
            no_improve_count = 0
            if current_score > best_score:
                best_score = current_score
                best_area_val = current_max_area
                best_points = [p[:] for p in current_points]
                best_counts = dict(current_counts)
        else:
            # Revert
            current_points[i] = (old_x, old_y)
            # Revert counts
            for area in new_areas:
                current_counts[area] -= 1
                if current_counts[area] == 0:
                    del current_counts[area]
            for area in old_areas:
                current_counts[area] += 1
            no_improve_count += 1
            
            # If stuck, maybe try a larger jump or restart sooner
            if no_improve_count % 100 == 0:
                 # Random perturbation
                 idx = random.randint(0, N-1)
                 dx2, dy2 = random.randint(-5, 5), random.randint(-5, 5)
                 nx2, ny2 = current_points[idx][0] + dx2, current_points[idx][1] + dy2
                 if min_coord <= nx2 <= max_coord and min_coord <= ny2 <= max_coord:
                     collides2 = False
                     for k in range(N):
                         if k != idx and current_points[k] == (nx2, ny2):
                             collides2 = True
                             break
                     if not collides2:
                         # Just accept a random move to escape local optima, even if it decreases score?
                         # This is Simulated Annealing style. Let's just do it to mix up the search.
                         # But we need to update counts correctly.
                         # For simplicity, just re-evaluate if we do a random jump.
                         # Or just restart.
                         pass
                         
    return best_points

# EVOLVE-BLOCK-END
