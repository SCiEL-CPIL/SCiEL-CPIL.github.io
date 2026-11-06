import random
import time
import math

def count_ordinary_lines(points):
    n = len(points)
    lines = {}
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
            if g > 0:
                A //= g
                B //= g
                C //= g
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
            key = (A, B, C)
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
    
    ordinary = 0
    for pts in lines.values():
        if len(pts) == 2:
            ordinary += 1
    return ordinary

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    best_score = float('inf')
    best_points = None
    
    def run_sa(start_points, max_iters=10000):
        current_points = [p[:] for p in start_points]
        current_score = count_ordinary_lines(current_points)
        temp = 10.0
        cooling = 0.99
        min_temp = 0.1
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            if temp < min_temp:
                break
                
            i = random.randint(0, 11)
            x, y = current_points[i]
            # Random move
            dx = random.randint(-10, 10)
            dy = random.randint(-10, 10)
            nx, ny = x + dx, y + dy
            if abs(nx) > 10**9 or abs(ny) > 10**9:
                continue
            is_distinct = True
            for j in range(12):
                if j != i and current_points[j] == [nx, ny]:
                    is_distinct = False
                    break
            if not is_distinct:
                continue
                
            new_points = [p[:] for p in current_points]
            new_points[i] = [nx, ny]
            new_score = count_ordinary_lines(new_points)
            
            delta = new_score - current_score
            if delta < 0:
                current_points = new_points
                current_score = new_score
            else:
                # Accept with probability exp(-delta/temp)
                prob = math.exp(-delta / temp)
                if random.random() < prob:
                    current_points = new_points
                    current_score = new_score
            
            temp *= cooling
            
        return current_points, current_score

    # Initial guesses
    guesses = []
    # 3x4 Grid
    guesses.append([[x, y] for x in range(4) for y in range(3)])
    # Random
    for _ in range(5):
        pts = []
        while len(pts) < 12:
            x = random.randint(-50, 50)
            y = random.randint(-50, 50)
            if [x, y] not in pts:
                pts.append([x, y])
        guesses.append(pts)
        
    for guess in guesses:
        if time.time() > deadline:
            break
        pts, score = run_sa(guess)
        if score < best_score:
            best_score = score
            best_points = pts

    return best_points
