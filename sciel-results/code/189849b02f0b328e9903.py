import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10-second safety margin
    
    def calc_score(points):
        n = len(points)
        lines_counted = set()
        score = 0
        penalty = 0
        
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                
                dx = x2 - x1
                dy = y2 - y1
                
                g = math.gcd(abs(dx), abs(dy))
                if g == 0:
                    continue
                dx_c = dx // g
                dy_c = dy // g
                
                if dx_c < 0 or (dx_c == 0 and dy_c < 0):
                    dx_c = -dx_c
                    dy_c = -dy_c
                
                c = dy_c * x1 - dx_c * y1
                line_key = (dx_c, dy_c, c)
                
                if line_key in lines_counted:
                    continue
                
                lines_counted.add(line_key)
                
                count = 0
                for k in range(n):
                    x3, y3 = points[k]
                    if dy_c * x3 - dx_c * y3 == c:
                        count += 1
                
                if count == 4:
                    score += 100
                elif count >= 5:
                    penalty += 1000
        
        return score - penalty

    # Initial configuration: 4x4 grid
    best_points = [[x, y] for x in range(4) for y in range(4)]
    best_score = calc_score(best_points)
    
    # SA Parameters
    T_start = 50.0
    T_end = 0.1
    cooling_rate = 0.999
    
    # We will run multiple restarts if time permits
    # But one long run is usually fine.
    
    current_points = [p[:] for p in best_points]
    current_score = best_score
    T = T_start
    
    # To avoid getting stuck in a local minimum, we can restart
    # if no improvement for a certain number of steps.
    steps_since_improvement = 0
    max_steps_without_improvement = 5000
    
    while time.time() < deadline:
        idx = random.randint(0, 15)
        old_x, old_y = current_points[idx]
        
        # Random move
        # Use larger steps to escape local minima
        step_x = random.randint(-20, 20)
        step_y = random.randint(-20, 20)
        
        new_x = old_x + step_x
        new_y = old_y + step_y
        
        if not (-1000 <= new_x <= 1000 and -1000 <= new_y <= 1000):
            continue
            
        is_distinct = True
        for k in range(16):
            if k == idx:
                continue
            if current_points[k][0] == new_x and current_points[k][1] == new_y:
                is_distinct = False
                break
                
        if not is_distinct:
            continue
            
        current_points[idx] = [new_x, new_y]
        new_score = calc_score(current_points)
        
        if new_score > current_score:
            current_score = new_score
            steps_since_improvement = 0
            if new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in current_points]
        else:
            delta = new_score - current_score
            if delta < 0:
                if random.random() < math.exp(delta / T):
                    steps_since_improvement += 1
                else:
                    current_points[idx] = [old_x, old_y]
                    steps_since_improvement += 1
            else:
                steps_since_improvement += 1
                
        T *= cooling_rate
        if T < T_end:
            T = T_end
            
        # Restart if stuck
        if steps_since_improvement > max_steps_without_improvement:
            # Reset to best found so far and restart SA
            current_points = [p[:] for p in best_points]
            current_score = best_score
            T = T_start
            steps_since_improvement = 0

    return best_points
