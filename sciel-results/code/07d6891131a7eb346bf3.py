import random
from itertools import combinations

def construct():
    # Initial configuration: 4x4 grid
    # Shifted to center? No, just 0..3
    points = [[x, y] for x in range(4) for y in range(4)]
    
    def score(points_list):
        pts = set(tuple(p) for p in points_list)
        if len(pts) != len(points_list):
            return -1
        lines = set()
        invalid = False
        # Check all pairs to find lines
        for i in range(len(points_list)):
            for j in range(i + 1, len(points_list)):
                p1 = points_list[i]
                p2 = points_list[j]
                # Line through p1 and p2
                # Check all other points
                collinear = [p1, p2]
                for k in range(len(points_list)):
                    if k == i or k == j:
                        continue
                    p3 = points_list[k]
                    # Check if p3 is collinear with p1 and p2
                    # (x2-x1)*(y3-y1) == (y2-y1)*(x3-x1)
                    if (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p2[1] - p1[1]) * (p3[0] - p1[0]):
                        collinear.append(p3)
                
                if len(collinear) >= 4:
                    # Create a unique signature for this line
                    # Sort the points to make a hashable key
                    sig = frozenset(tuple(p) for p in collinear)
                    if len(sig) >= 5:
                        invalid = True
                    else:
                        lines.add(sig)
        
        if invalid:
            return -1
        return len(lines)

    best_score = score(points)
    best_points = [p[:] for p in points]
    
    # Local Search
    # Define a search box
    min_coord = -2
    max_coord = 6
    
    for iteration in range(5000):
        improved = False
        # Randomly select a point to move
        idx = random.randint(0, 15)
        old_point = best_points[idx]
        
        # Try to move it to a new position in the box
        # To avoid too many random restarts, we can try all nearby points or just random
        # Let's try a few random new positions
        for _ in range(10):
            new_x = random.randint(min_coord, max_coord)
            new_y = random.randint(min_coord, max_coord)
            new_point = [new_x, new_y]
            
            # Ensure distinctness
            if any(p == new_point for p in best_points if p != old_point):
                continue
            
            # Create new configuration
            candidate = best_points[:]
            candidate[idx] = new_point
            
            s = score(candidate)
            if s > best_score:
                best_score = s
                best_points = candidate
                improved = True
                break
            elif s == best_score and random.random() < 0.1:
                # Accept equal score with small probability to escape local minima
                best_points = candidate
                improved = True
                break
        
        if not improved and iteration % 100 == 0:
            # Maybe restart with a different random configuration?
            # Or just continue
            pass
            
    return [[int(x), int(y)] for x, y in best_points]
