import time
import random

def construct():
    # Initial 5x6 grid
    pts = [[i, j] for i in range(5) for j in range(6)]
    
    def get_score(points):
        n = len(points)
        areas = {}
        for i in range(n):
            x1, y1 = points[i]
            for j in range(i+1, n):
                x2, y2 = points[j]
                dx1 = x2 - x1
                dy1 = y2 - y1
                for k in range(j+1, n):
                    x3, y3 = points[k]
                    det = abs(dx1 * (y3 - y1) - (x3 - x1) * dy1)
                    if det > 0:
                        areas[det] = areas.get(det, 0) + 1
        if not areas:
            return 0
        return max(areas.values())
    
    current_pts = pts
    current_score = get_score(current_pts)
    best_pts = current_pts[:]
    best_score = current_score
    
    start_time = time.time()
    time_limit = 55.0
    
    # Neighbors for moves
    offsets = [(1,0), (-1,0), (0,1), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1), (2,0), (-2,0), (0,2), (0,-2)]
    
    while time.time() - start_time < time_limit:
        improved = False
        # Try to find the best single-point move
        best_move_score = current_score
        best_move_idx = -1
        best_move_pt = None
        
        for idx in range(len(current_pts)):
            orig_x, orig_y = current_pts[idx]
            # Set of other points for distinctness
            others = set()
            for p in current_pts:
                if p != current_pts[idx]:
                    others.add((p[0], p[1]))
            
            for dx, dy in offsets:
                nx = orig_x + dx
                ny = orig_y + dy
                if nx < -30 or nx > 30 or ny < -30 or ny > 30:
                    continue
                if (nx, ny) in others:
                    continue
                
                new_pts = current_pts[:]
                new_pts[idx] = [nx, ny]
                
                score = get_score(new_pts)
                
                if score > best_move_score:
                    best_move_score = score
                    best_move_idx = idx
                    best_move_pt = [nx, ny]
        
        if best_move_idx != -1:
            current_pts[best_move_idx] = best_move_pt
            current_score = best_move_score
            improved = True
            if current_score > best_score:
                best_score = current_score
                best_pts = current_pts[:]
        else:
            # No single move improves. Try a random swap of two points
            # to escape local optimum
            if random.random() < 0.5:
                i, j = random.sample(range(30), 2)
                new_pts = current_pts[:]
                new_pts[i], new_pts[j] = new_pts[j], new_pts[i]
                score = get_score(new_pts)
                if score > current_score:
                    current_pts = new_pts
                    current_score = score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_pts = current_pts[:]
            
            if not improved:
                # Stuck. Break or try more aggressive random jumps?
                # For safety, break to ensure return.
                # Or try a few random restarts?
                # Given time, break is safer.
                break
                
    return [tuple(p) for p in best_pts]
