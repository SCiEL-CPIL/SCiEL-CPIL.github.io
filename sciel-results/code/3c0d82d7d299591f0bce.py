import time
import random
import math

def construct():
    start_time = time.time()
    time_limit = 110.0
    
    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            return None
        g = math.gcd(abs(dx), abs(dy))
        dx //= g
        dy //= g
        if dx < 0 or (dx == 0 and dy < 0):
            dx = -dx
            dy = -dy
        c = dy * x1 - dx * y1
        return (dx, dy, c)

    def evaluate(points):
        if len(set(tuple(p) for p in points)) != len(points):
            return -10000
            
        n = len(points)
        line_counts = {}
        
        for i in range(n):
            for j in range(i + 1, n):
                key = get_line_key(points[i], points[j])
                if key is None:
                    continue
                if key not in line_counts:
                    line_counts[key] = 2
                else:
                    line_counts[key] += 1
        
        score = 0
        invalid = False
        for count in line_counts.values():
            if count == 6:
                score += 1
            elif count >= 10:
                invalid = True
                break
            
        if invalid:
            return -1000 + score
        return score

    best_points = None
    best_score = -1
    
    # Try multiple restarts
    num_restarts = 20
    
    for r in range(num_restarts):
        if time.time() - start_time > time_limit - 5:
            break
            
        # Initial config: Grid with random jitter
        points = []
        for x in range(4):
            for y in range(4):
                # Small jitter to break symmetry
                jx = random.randint(-1, 1) if r > 0 else 0
                jy = random.randint(-1, 1) if r > 0 else 0
                points.append([x + jx, y + jy])
        
        # Ensure distinctness
        # If collision, re-roll or shift
        while len(set(tuple(p) for p in points)) < 16:
            idx = random.randint(0, 15)
            points[idx][0] += random.randint(-2, 2)
            points[idx][1] += random.randint(-2, 2)
        
        current_points = [p[:] for p in points]
        current_score = evaluate(current_points)
        
        if current_score > best_score:
            best_score = current_score
            best_points = [p[:] for p in current_points]
        
        T = 1.0
        T_min = 0.01
        cooling = 0.999
        stuck_count = 0
        
        # Local search for this restart
        while time.time() - start_time < time_limit - 2:
            idx = random.randint(0, 15)
            old_x, old_y = current_points[idx]
            
            new_x = old_x + random.randint(-10, 10)
            new_y = old_y + random.randint(-10, 10)
            
            if abs(new_x) > 1000 or abs(new_y) > 1000:
                continue
            
            new_point = [new_x, new_y]
            is_distinct = True
            for k in range(16):
                if k != idx and current_points[k] == new_point:
                    is_distinct = False
                    break
            
            if not is_distinct:
                continue

            current_points[idx] = new_point
            new_score = evaluate(current_points)
            
            if new_score > current_score:
                current_score = new_score
                stuck_count = 0
                if new_score > best_score:
                    best_score = new_score
                    best_points = [p[:] for p in current_points]
            else:
                delta = current_score - new_score
                if delta < 0:
                    accept = True
                else:
                    if delta / T > 100:
                        accept_prob = 0.0
                    else:
                        accept_prob = math.exp(-delta / T)
                    accept = random.random() < accept_prob
                
                if accept:
                    stuck_count += 1
                else:
                    current_points[idx] = [old_x, old_y]
                    stuck_count += 1
            
            T *= cooling
            if T < T_min:
                T = T_min
            
            if stuck_count > 2000:
                break

    return best_points
