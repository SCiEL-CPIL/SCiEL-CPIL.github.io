import random

def construct():
    # Start with 4x4 grid
    points = [[x, y] for x in range(4) for y in range(4)]
    
    def score(points_list):
        pts = set(tuple(p) for p in points_list)
        if len(pts) != len(points_list):
            return -1
        lines = set()
        # Check all pairs to find lines
        for i in range(len(points_list)):
            for j in range(i + 1, len(points_list)):
                p1 = points_list[i]
                p2 = points_list[j]
                collinear = [p1, p2]
                for k in range(len(points_list)):
                    if k == i or k == j:
                        continue
                    p3 = points_list[k]
                    if (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p2[1] - p1[1]) * (p3[0] - p1[0]):
                        collinear.append(p3)
                
                if len(collinear) >= 4:
                    sig = frozenset(tuple(p) for p in collinear)
                    if len(sig) >= 5:
                        return -1
                    else:
                        lines.add(sig)
        return len(lines)

    best_score = score(points)
    best_points = [p[:] for p in points]
    
    min_coord = -5
    max_coord = 10
    
    best_global_score = best_score
    best_global_points = best_points[:]
    
    for restart in range(30):
        if restart == 0:
            current_points = best_global_points[:]
        else:
            # Random restart with slight perturbation of best global
            current_points = best_global_points[:]
            for _ in range(random.randint(3, 6)):
                idx = random.randint(0, 15)
                old_p = current_points[idx]
                new_x = old_p[0] + random.randint(-4, 4)
                new_y = old_p[1] + random.randint(-4, 4)
                if min_coord <= new_x <= max_coord and min_coord <= new_y <= max_coord:
                    if not any(p == [new_x, new_y] for i, p in enumerate(current_points) if i != idx):
                        current_points[idx] = [new_x, new_y]
            
        current_score = score(current_points)
        
        improved = True
        passes = 0
        while improved and passes < 30:
            improved = False
            passes += 1
            
            # 1. Try single point moves (neighborhood +-1)
            indices = list(range(16))
            random.shuffle(indices)
            for idx in indices:
                old_point = current_points[idx]
                found_move = False
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        if dx == 0 and dy == 0:
                            continue
                        new_x = old_point[0] + dx
                        new_y = old_point[1] + dy
                        if not (min_coord <= new_x <= max_coord and min_coord <= new_y <= max_coord):
                            continue
                        
                        new_point = [new_x, new_y]
                        if any(p == new_point for i, p in enumerate(current_points) if i != idx):
                            continue
                        
                        candidate = current_points[:]
                        candidate[idx] = new_point
                        s = score(candidate)
                        
                        if s > current_score:
                            current_score = s
                            current_points = candidate
                            improved = True
                            found_move = True
                            break
                        elif s == current_score and random.random() < 0.05:
                            current_points = candidate
                            improved = True
                            found_move = True
                            break
                    if found_move:
                        break
                # Note: If we moved, we don't update old_point for remaining dx/dy in this iteration?
                # The loop breaks, so it's fine.
                
            # 2. Try swaps
            if not improved:
                for i in range(16):
                    for j in range(i + 1, 16):
                        candidate = current_points[:]
                        candidate[i], candidate[j] = candidate[j], candidate[i]
                        s = score(candidate)
                        if s > current_score:
                            current_score = s
                            current_points = candidate
                            improved = True
                            break
                        elif s == current_score and random.random() < 0.05:
                            current_points = candidate
                            improved = True
                            break
                    if improved:
                        break
        
        if current_score > best_global_score:
            best_global_score = current_score
            best_global_points = current_points[:]
            
    return [[int(x), int(y)] for x, y in best_global_points]
