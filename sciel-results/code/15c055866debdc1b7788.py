import random
import time
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    def gcd3(a, b, c):
        return math.gcd(math.gcd(a, b), c)

    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        A = y1 - y2
        B = x2 - x1
        C = x1 * y2 - x2 * y1
        
        g = gcd3(A, B, C)
        if g != 0:
            A //= g
            B //= g
            C //= g
        else:
            return (0, 0, 0)
            
        if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
            A = -A
            B = -B
            C = -C
        return (A, B, C)

    def evaluate(points):
        if len(points) != 16:
            return 0, False
        
        if len(set(tuple(p) for p in points)) != 16:
            return 0, False
            
        for p in points:
            if not (-1000 <= p[0] <= 1000 and -1000 <= p[1] <= 1000):
                return 0, False
                
        lines = defaultdict(set)
        
        for i in range(16):
            for j in range(i + 1, 16):
                key = get_line_key(points[i], points[j])
                lines[key].add(i)
                lines[key].add(j)
        
        score = 0
        for key, pts_on_line in lines.items():
            count = len(pts_on_line)
            if count >= 5:
                return 0, False
            elif count == 4:
                score += 1
                
        return score, True

    # Initial configuration: 4x4 grid
    points = [[x, y] for x in range(4) for y in range(4)]
    
    current_score, current_valid = evaluate(points)
    best_points = [p[:] for p in points]
    best_score = current_score if current_valid else -1
    
    temperature = 10.0
    cooling_rate = 0.99995
    min_temperature = 0.01
    steps_since_best = 0
    max_steps_since_best = 10000
    
    while time.time() < deadline:
        if steps_since_best > max_steps_since_best:
            # Random restart
            points = [[random.randint(-100, 100), random.randint(-100, 100)] for _ in range(16)]
            attempts = 0
            while len(set(tuple(p) for p in points)) < 16 and attempts < 100:
                idx = random.randint(0, 15)
                points[idx] = [random.randint(-100, 100), random.randint(-100, 100)]
                attempts += 1
            
            current_score, current_valid = evaluate(points)
            if not current_valid:
                current_score = -1000
            else:
                if current_score > best_score:
                    best_score = current_score
                    best_points = [p[:] for p in points]
                    steps_since_best = 0
                else:
                    steps_since_best = 0
            
            temperature = 10.0
            continue

        new_points = [p[:] for p in points]
        idx = random.randint(0, 15)
        axis = random.randint(0, 1)
        
        step = random.randint(-15, 15)
        if step == 0:
            continue
            
        new_points[idx][axis] += step
        
        new_score, new_valid = evaluate(new_points)
        
        if not new_valid:
            new_score = -1000 
        else:
            if new_score > current_score:
                points = new_points
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = [p[:] for p in points]
                    steps_since_best = 0
                else:
                    steps_since_best += 1
            else:
                delta = current_score - new_score
                if delta == 0:
                    if random.random() < 0.5:
                        points = new_points
                        current_score = new_score
                        steps_since_best += 1
                else:
                    prob = math.exp(-delta / temperature)
                    if random.random() < prob:
                        points = new_points
                        current_score = new_score
                        steps_since_best += 1
                    else:
                        steps_since_best += 1
        
        temperature *= cooling_rate
        if temperature < min_temperature:
             temperature = min_temperature

    return best_points
