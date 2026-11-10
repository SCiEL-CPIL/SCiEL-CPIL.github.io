import random
import time

N = 64

def construct():
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin
    
    best_set = []
    best_score = 0
    
    # Pre-generate all points
    all_points = [(x, y) for x in range(N) for y in range(N)]
    
    # Pre-compute heuristic orders
    # 1. Random (handled in loop)
    # 2. Sorted by x
    order_x = sorted(all_points, key=lambda p: p[0])
    # 3. Sorted by y
    order_y = sorted(all_points, key=lambda p: p[1])
    # 4. Sorted by x+y
    order_xy = sorted(all_points, key=lambda p: p[0] + p[1])
    # 5. Sorted by distance to center (center first)
    center = (N/2, N/2)
    def dist_center(p):
        return (p[0]-center[0])**2 + (p[1]-center[1])**2
    order_center_in = sorted(all_points, key=dist_center)
    # 6. Sorted by distance to center (edge first)
    order_center_out = sorted(all_points, key=lambda p: -dist_center(p))
    
    deterministic_orders = [order_x, order_y, order_xy, order_center_in, order_center_out]
    
    attempt_count = 0
    while time.time() < deadline:
        attempt_count += 1
        
        # Try deterministic orders first
        if attempt_count <= len(deterministic_orders):
            points = deterministic_orders[attempt_count - 1]
        else:
            random.seed(attempt_count)
            points = all_points[:]
            random.shuffle(points)
        
        current_set = []
        dist_sets = []
        
        for p in points:
            px, py = p
            valid = True
            new_dists = []
            
            for i, q in enumerate(current_set):
                qx, qy = q
                d2 = (px - qx)**2 + (py - qy)**2
                
                if d2 in dist_sets[i]:
                    valid = False
                    break
                
                new_dists.append(d2)
            
            if valid:
                if len(new_dists) != len(set(new_dists)):
                    valid = False
            
            if valid:
                current_set.append(p)
                new_dist_set = set(new_dists)
                dist_sets.append(new_dist_set)
                
                for i in range(len(current_set) - 1):
                    q = current_set[i]
                    d2 = (px - q[0])**2 + (py - q[1])**2
                    dist_sets[i].add(d2)
        
        if len(current_set) > best_score:
            best_score = len(current_set)
            best_set = current_set[:]
            
    return best_set
