import time
import random
import itertools
from collections import defaultdict

def _count_lines_exactly_4(points):
    """Count lines containing exactly 4 of the given points."""
    n = len(points)
    count = 0
    seen_lines = set()
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = _gcd(abs(dx), abs(dy))
            if g > 0:
                dx_n = dx // g
                dy_n = dy // g
            else:
                continue
            if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                dx_n = -dx_n
                dy_n = -dy_n
            a = -dy_n
            b = dx_n
            c = a * x1 + b * y1
            g2 = _gcd(_gcd(abs(a), abs(b)), abs(c))
            if g2 > 0:
                a //= g2
                b //= g2
                c //= g2
            if a < 0 or (a == 0 and b < 0):
                a = -a
                b = -b
                c = -c
            key = (a, b, c)
            if key in seen_lines:
                continue
            cnt = 0
            for k in range(n):
                xk, yk = points[k]
                if a * xk + b * yk == c:
                    cnt += 1
            if cnt == 4:
                count += 1
            seen_lines.add(key)
    return count

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _score(points):
    """Score = number of lines with exactly 4 points. Penalize if 5+ collinear."""
    n = len(points)
    seen_lines = set()
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = _gcd(abs(dx), abs(dy))
            if g > 0:
                dx_n = dx // g
                dy_n = dy // g
            else:
                continue
            if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                dx_n = -dx_n
                dy_n = -dy_n
            a = -dy_n
            b = dx_n
            c = a * x1 + b * y1
            g2 = _gcd(_gcd(abs(a), abs(b)), abs(c))
            if g2 > 0:
                a //= g2
                b //= g2
                c //= g2
            if a < 0 or (a == 0 and b < 0):
                a = -a
                b = -b
                c = -c
            key = (a, b, c)
            if key in seen_lines:
                continue
            cnt = 0
            for k in range(n):
                xk, yk = points[k]
                if a * xk + b * yk == c:
                    cnt += 1
            if cnt >= 5:
                return -1000
            seen_lines.add(key)
    return _count_lines_exactly_4(points)

def _local_search(points, deadline, max_move=20):
    """Local search: try moving individual points to improve score."""
    best_points = [p[:] for p in points]
    best_score = _score(best_points)
    n = 16
    
    while time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            bx, by = best_points[i]
            # Try moving point i to nearby positions with increasing radius
            for radius in [3, 8, 15, max_move]:
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = bx + dx, by + dy
                        if abs(nx) > 1000 or abs(ny) > 1000:
                            continue
                        if any(best_points[j] == [nx, ny] for j in range(n) if j != i):
                            continue
                        new_points = [p[:] for p in best_points]
                        new_points[i] = [nx, ny]
                        s = _score(new_points)
                        if s > best_score:
                            best_score = s
                            best_points = new_points
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            # Try random jumps to escape local optima
            for _ in range(200):
                if time.time() > deadline:
                    break
                i = random.randint(0, n - 1)
                bx, by = best_points[i]
                nx = bx + random.randint(-50, 50)
                ny = by + random.randint(-50, 50)
                if abs(nx) > 1000 or abs(ny) > 1000:
                    continue
                if any(best_points[j] == [nx, ny] for j in range(n) if j != i):
                    continue
                new_points = [p[:] for p in best_points]
                new_points[i] = [nx, ny]
                s = _score(new_points)
                if s > best_score:
                    best_score = s
                    best_points = new_points
                    improved = True
                    break
            if not improved:
                break
    return best_points, best_score

def construct():
    deadline = time.time() + 140  # Safety margin
    
    # Start from a 4x4 grid (known to give 8)
    base_points = [[x, y] for x in range(4) for y in range(4)]
    
    best_overall = [p[:] for p in base_points]
    best_score_overall = _score(best_overall)
    
    # Try multiple initial configurations
    initial_configs = []
    
    # 4x4 grid
    initial_configs.append([[x, y] for x in range(4) for y in range(4)])
    
    # 5x5 grid subset (pick 16 from 25) - more diverse samples
    for _ in range(30):
        grid_5x5 = [[x, y] for x in range(5) for y in range(5)]
        random.shuffle(grid_5x5)
        initial_configs.append(grid_5x5[:16])
    
    # 6x4 grid subset
    for _ in range(30):
        grid_6x4 = [[x, y] for x in range(6) for y in range(4)]
        random.shuffle(grid_6x4)
        initial_configs.append(grid_6x4[:16])
    
    # 4x6 grid subset
    for _ in range(20):
        grid_4x6 = [[x, y] for x in range(4) for y in range(6)]
        random.shuffle(grid_4x6)
        initial_configs.append(grid_4x6[:16])
    
    # 5x4 grid + some extras
    for _ in range(15):
        grid_5x4 = [[x, y] for x in range(5) for y in range(4)]
        random.shuffle(grid_5x4)
        subset = grid_5x4[:12]
        extra = []
        while len(extra) < 4:
            px = random.randint(-3, 8)
            py = random.randint(-3, 8)
            p = [px, py]
            if p not in subset and p not in extra:
                extra.append(p)
        initial_configs.append(subset + extra)
    
    # Random configurations with wider spread
    for _ in range(40):
        pts = []
        while len(pts) < 16:
            px = random.randint(-20, 20)
            py = random.randint(-20, 20)
            p = [px, py]
            if p not in pts:
                pts.append(p)
        initial_configs.append(pts)
    
    # Structured patterns: two 4x4 grids offset
    for _ in range(10):
        pts = [[x, y] for x in range(4) for y in range(4)]
        offset = random.randint(2, 5)
        pts2 = [[x + offset, y + offset] for x in range(4) for y in range(4)]
        # Merge and pick 16
        all_pts = pts + pts2
        random.shuffle(all_pts)
        initial_configs.append(all_pts[:16])
    
    # Run local search from each initial config
    for config in initial_configs:
        if time.time() > deadline:
            break
        # Ensure distinct points
        seen = set()
        unique_config = []
        for p in config:
            key = (p[0], p[1])
            if key not in seen:
                seen.add(key)
                unique_config.append(p[:])
        if len(unique_config) < 16:
            continue
        
        result, score = _local_search(unique_config, deadline)
        if score > best_score_overall:
            best_score_overall = score
            best_overall = result
    
    # Final polish: try wider moves on the best solution
    if best_score_overall > 0:
        final_result, final_score = _local_search(best_overall, deadline, max_move=50)
        if final_score > best_score_overall:
            best_score_overall = final_score
            best_overall = final_result
    
    # Ensure validity
    if _score(best_overall) < 0:
        best_overall = [[x, y] for x in range(4) for y in range(4)]
    
    return best_overall
