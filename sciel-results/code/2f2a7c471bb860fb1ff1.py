"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def _score_points(points):
    """Return (num_exact_4_lines, has_5_collinear) for 16 points."""
    n = 16
    lines = {}
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = math.gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dx_n, dy_n = dx // g, dy // g
            if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                dx_n, dy_n = -dx_n, -dy_n
            a, b = dy_n, -dx_n
            c = a * x1 + b * y1
            g2 = math.gcd(abs(a), abs(b))
            if g2 > 1:
                a, b, c = a // g2, b // g2, c // g2
            if a < 0 or (a == 0 and b < 0):
                a, b, c = -a, -b, -c
            key = (a, b, c)
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
    
    exact_4 = 0
    has_5 = False
    for key, pts_on_line in lines.items():
        cnt = len(pts_on_line)
        if cnt == 4:
            exact_4 += 1
        elif cnt >= 5:
            has_5 = True
    return exact_4, has_5

def construct():
    deadline = time.time() + 145
    
    base = [[x, y] for x in range(4) for y in range(4)]
    best_score, _ = _score_points(base)
    best_points = [p[:] for p in base]
    
    rng = random.Random(42)
    
    for restart in range(200):
        if time.time() > deadline - 10:
            break
        
        if restart == 0:
            current = [p[:] for p in base]
        elif restart % 3 == 0:
            current = [[rng.randint(-20, 20), rng.randint(-20, 20)] for _ in range(16)]
            for k in range(16):
                for k2 in range(k+1, 16):
                    if current[k] == current[k2]:
                        current[k2][0] += 1
        else:
            current = [p[:] for p in best_points]
            for _ in range(rng.randint(1, 5)):
                idx = rng.randint(0, 15)
                current[idx] = [current[idx][0] + rng.randint(-10, 10), 
                                current[idx][1] + rng.randint(-10, 10)]
                if any(current[idx] == p for p in current[:idx] + current[idx+1:]):
                    current[idx][0] += 1
            
        s, h5 = _score_points(current)
        if h5:
            continue
        
        current_score = s
        
        T = 1.0
        no_improve = 0
        max_no_improve = 1000
        
        while time.time() < deadline - 5 and no_improve < max_no_improve:
            idx = rng.randint(0, 15)
            old = current[idx][:]
            
            step = rng.randint(1, 10)
            dir_x = rng.choice([-1, 0, 1])
            dir_y = rng.choice([-1, 0, 1])
            if dir_x == 0 and dir_y == 0:
                dir_x = 1
            
            new_x = old[0] + dir_x * step
            new_y = old[1] + dir_y * step
            
            if abs(new_x) > 1000 or abs(new_y) > 1000:
                continue
            new_pt = [new_x, new_y]
            
            if any(new_pt == p for p in current):
                continue
            
            current[idx] = new_pt
            s_new, h5_new = _score_points(current)
            
            if h5_new:
                current[idx] = old
                no_improve += 1
                continue
            
            if s_new >= current_score or (T > 0.01 and rng.random() < math.exp((s_new - current_score) / T)):
                if s_new > current_score:
                    no_improve = 0
                    if s_new > best_score:
                        best_score = s_new
                        best_points = [p[:] for p in current]
                else:
                    no_improve += 1
                current_score = s_new
            else:
                current[idx] = old
                no_improve += 1
            
            T = max(0.01, T * 0.999)
            
    return best_points
# EVOLVE-BLOCK-END
