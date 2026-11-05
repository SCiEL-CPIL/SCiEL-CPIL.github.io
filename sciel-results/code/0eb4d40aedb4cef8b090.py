import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    # Initial 6x5 grid
    points = [(x, y) for x in range(-2, 4) for y in range(-2, 3)]
    
    def get_areas(points):
        areas = {}
        n = len(points)
        # Use local variables for speed
        for i in range(n):
            x1, y1 = points[i]
            for j in range(i+1, n):
                x2, y2 = points[j]
                for k in range(j+1, n):
                    x3, y3 = points[k]
                    det = x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)
                    if det != 0:
                        adet = abs(det)
                        areas[adet] = areas.get(adet, 0) + 1
        return areas

    def max_multiplicity(areas):
        if not areas:
            return 0
        return max(areas.values())

    # Initial state
    current_areas = get_areas(points)
    current_best = max_multiplicity(current_areas)
    global_best_points = points[:]
    global_best_score = current_best
    
    occupied = set(points)
    
    # Search bounds for new points to keep cluster compact
    min_x, min_y = -5, -5
    max_x, max_y = 6, 5
    
    while time.time() < deadline:
        idx = random.randint(0, 29)
        old_pos = points[idx]
        
        # 80% local move, 20% jump within bounds
        if random.random() < 0.8:
            dx = random.randint(-2, 2)
            dy = random.randint(-2, 2)
            if dx == 0 and dy == 0:
                continue
            new_pos = (old_pos[0] + dx, old_pos[1] + dy)
        else:
            new_pos = (random.randint(min_x, max_x), random.randint(min_y, max_y))
            
        if not (min_x <= new_pos[0] <= max_x and min_y <= new_pos[1] <= max_y):
            continue
            
        if new_pos in occupied:
            continue
            
        # Move
        points[idx] = new_pos
        occupied.remove(old_pos)
        occupied.add(new_pos)
        
        new_areas = get_areas(points)
        new_best = max_multiplicity(new_areas)
        
        if new_best > current_best:
            current_best = new_best
            if current_best > global_best_score:
                global_best_score = current_best
                global_best_points = points[:]
        else:
            # Revert
            points[idx] = old_pos
            occupied.remove(new_pos)
            occupied.add(old_pos)
            
    return global_best_points
