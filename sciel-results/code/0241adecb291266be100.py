import time
import random

N = 64

def squared_dist(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx * dx + dy * dy

def construct():
    start_time = time.time()
    deadline = start_time + 270.0  # safety margin before 330s kill
    
    best_set = []
    best_size = 0
    
    # Precompute all grid points
    all_points = [(x, y) for x in range(N) for y in range(N)]
    
    def is_valid_addition(p, point_set):
        """Check if adding p to point_set creates any isosceles triangle."""
        pts = list(point_set)
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                a = pts[i]
                b = pts[j]
                pa2 = squared_dist(p, a)
                pb2 = squared_dist(p, b)
                ab2 = squared_dist(a, b)
                # Check if triangle (p, a, b) is isosceles
                if pa2 == pb2:  # apex at p
                    return False
                if pa2 == ab2:  # apex at a
                    return False
                if pb2 == ab2:  # apex at b
                    return False
        return True
    
    num_restarts = 200
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Random ordering of points
        shuffled = all_points[:]
        random.shuffle(shuffled)
        
        current_set = set()
        
        for p in shuffled:
            if time.time() > deadline:
                break
            if is_valid_addition(p, current_set):
                current_set.add(p)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = list(current_set)
    
    return best_set
