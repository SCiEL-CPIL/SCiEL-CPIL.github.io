import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    def grid_4x4():
        return [[x, y] for x in range(4) for y in range(4)]
    
    def random_points():
        return [[random.randint(-50, 50), random.randint(-50, 50)] for _ in range(16)]
    
    def evaluate(points):
        lines = {}
        n = len(points)
        for i in range(n):
            xi, yi = points[i]
            for j in range(i+1, n):
                xj, yj = points[j]
                dx = xj - xi
                dy = yj - yi
                if dx == 0 and dy == 0:
                    continue
                g = math.gcd(abs(dx), abs(dy))
                dx_n = dx // g
                dy_n = dy // g
                a = dy_n
                b = -dx_n
                c = a * xi + b * yi
                if a < 0 or (a == 0 and b < 0):
                    a = -a
                    b = -b
                    c = -c
                g2 = math.gcd(math.gcd(abs(a), abs(b)), abs(c))
                if g2 > 0:
                    a //= g2
                    b //= g2
                    c //= g2
                key = (a, b, c)
                if key not in lines:
                    lines[key] = set()
                lines[key].add(i)
                lines[key].add(j)
        
        score = 0
        num_4 = 0
        num_ge5 = 0
        for key, pts in lines.items():
            cnt = len(pts)
            if cnt == 4:
                num_4 += 1
                score += 1
            elif cnt >= 5:
                num_ge5 += 1
                score -= 1000
        return score, num_4, num_ge5

    def is_valid(points):
        _, _, num_ge5 = evaluate(points)
        return num_ge5 == 0

    def sa_search(initial_points, start_temp=10.0, cool=0.999):
        current_points = [p[:] for p in initial_points]
        current_score, _, _ = evaluate(current_points)
        
        best_points = [p[:] for p in current_points]
        best_score = current_score
        
        temp = start_temp
        last_check = time.time()
        
        while True:
            if time.time() > deadline:
                break
            if time.time() - last_check > 1.0:
                last_check = time.time()
                # Optional: break if no improvement for long time? 
                # For now, just continue.
                
            idx = random.randint(0, 15)
            old_x, old_y = current_points[idx]
            
            if random.random() < 0.7:
                step_x = random.randint(-2, 2)
                step_y = random.randint(-2, 2)
            else:
                step_x = random.randint(-20, 20)
                step_y = random.randint(-20, 20)
            
            new_x = old_x + step_x
            new_y = old_y + step_y
            
            if abs(new_x) > 1000 or abs(new_y) > 1000:
                continue
                
            valid_move = True
            for k in range(16):
                if k != idx and current_points[k][0] == new_x and current_points[k][1] == new_y:
                    valid_move = False
                    break
            
            if not valid_move:
                continue

            current_points[idx] = [new_x, new_y]
            new_score, _, _ = evaluate(current_points)
            
            delta = new_score - current_score
            
            if delta > 0 or random.random() < math.exp(delta / temp):
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_points = [p[:] for p in current_points]
            else:
                current_points[idx] = [old_x, old_y]
                
            temp *= cool
            if temp < 0.01:
                break
                
        return best_points, best_score

    def hill_climb_valid(points):
        current_points = [p[:] for p in points]
        current_score, _, _ = evaluate(current_points)
        if not is_valid(current_points):
             return current_points, current_score
             
        best_points = [p[:] for p in current_points]
        best_score = current_score
        
        no_improve_count = 0
        while time.time() < deadline - 5:
            improved = False
            for _ in range(1000): # Fixed number of attempts per epoch
                idx = random.randint(0, 15)
                old_x, old_y = current_points[idx]
                step_x = random.randint(-3, 3)
                step_y = random.randint(-3, 3)
                new_x = old_x + step_x
                new_y = old_y + step_y
                if abs(new_x) > 1000 or abs(new_y) > 1000:
                    continue
                valid_move = True
                for k in range(16):
                    if k != idx and current_points[k][0] == new_x and current_points[k][1] == new_y:
                        valid_move = False
                        break
                if not valid_move:
                    continue

                current_points[idx] = [new_x, new_y]
                new_score, _, num_ge5 = evaluate(current_points)
                
                if num_ge5 == 0 and new_score > current_score:
                    current_score = new_score
                    best_points = [p[:] for p in current_points]
                    best_score = new_score
                    improved = True
                    no_improve_count = 0
                else:
                    current_points[idx] = [old_x, old_y]
                    
            if not improved:
                no_improve_count += 1
                if no_improve_count > 5:
                    break
                    
        return best_points, best_score

    best_overall_points = grid_4x4()
    best_overall_score = 10
    best_overall_valid = True
    
    # Phase 1: Global Search
    # Try 4x4
    p_best, s_best = sa_search(grid_4x4(), start_temp=5.0, cool=0.999)
    if s_best > best_overall_score and is_valid(p_best):
        best_overall_score = s_best
        best_overall_points = p_best
        best_overall_valid = True
        
    # Try random restarts
    while time.time() < deadline - 20:
        p_rand = random_points()
        p_best, s_best = sa_search(p_rand, start_temp=5.0, cool=0.995)
        if s_best > best_overall_score and is_valid(p_best):
            best_overall_score = s_best
            best_overall_points = p_best
            best_overall_valid = True
            
    # Phase 2: Polish
    if best_overall_valid:
        p_polished, s_polished = hill_climb_valid(best_overall_points)
        if s_polished > best_overall_score:
            best_overall_score = s_polished
            best_overall_points = p_polished
            
    # Final Fallback
    if not is_valid(best_overall_points):
        return grid_4x4()
        
    return best_overall_points
