import random
import time

def count_ordinary_lines(points):
    n = len(points)
    if n < 3:
        return 0
    lines = {}
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            if A == 0 and B == 0:
                continue
            if A < 0 or (A == 0 and B < 0):
                A, B, C = -A, -B, -C
            key = (A, B, C)
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
    ordinary_count = 0
    for pts in lines.values():
        if len(pts) == 2:
            ordinary_count += 1
    return ordinary_count

def construct():
    start_time = time.time()
    deadline = start_time + 110
    best_score = float('inf')
    best_points = None

    grid_3x4 = [[x, y] for x in range(4) for y in range(3)]
    current_points = [list(p) for p in grid_3x4]
    current_score = count_ordinary_lines(current_points)
    best_score = current_score
    best_points = [list(p) for p in current_points]

    for i in range(50000):
        if time.time() > deadline:
            break
        
        # Random move
        if random.random() < 0.5:
             idx = random.randint(0, 11)
             old_x, old_y = current_points[idx]
             new_x = random.randint(-30, 30)
             new_y = random.randint(-30, 30)
             if (new_x, new_y) == (old_x, old_y):
                 continue
             if any(p[0] == new_x and p[1] == new_y for j, p in enumerate(current_points) if j != idx):
                 continue
             current_points[idx] = [new_x, new_y]
             new_score = count_ordinary_lines(current_points)
             if new_score < current_score:
                 current_score = new_score
                 if new_score < best_score:
                     best_score = new_score
                     best_points = [list(p) for p in current_points]
             else:
                 current_points[idx] = [old_x, old_y]
        else:
             # Smart move: try to place a point on intersection of two ordinary lines
             # Get ordinary lines
             n = len(current_points)
             lines = {}
             for a in range(n):
                 for b in range(a + 1, n):
                     x1, y1 = current_points[a]
                     x2, y2 = current_points[b]
                     A = y1 - y2
                     B = x2 - x1
                     C = x1 * y2 - x2 * y1
                     if A == 0 and B == 0: continue
                     if A < 0 or (A == 0 and B < 0):
                         A, B, C = -A, -B, -C
                     key = (A, B, C)
                     if key not in lines:
                         lines[key] = set()
                     lines[key].add(a)
                     lines[key].add(b)
             
             ordinary_lines = [key for key, pts in lines.items() if len(pts) == 2]
             
             if not ordinary_lines:
                 continue
             
             # Pick two random ordinary lines
             l1, l2 = random.sample(ordinary_lines, 2)
             A1, B1, C1 = l1
             A2, B2, C2 = l2
             
             # Find intersection
             det = A1 * B2 - A2 * B1
             if det == 0:
                 continue
             
             x_num = B1 * C2 - B2 * C1
             y_num = A2 * C1 - A1 * C2
             
             if x_num % det != 0 or y_num % det != 0:
                 continue
                 
             new_x = x_num // det
             new_y = y_num // det
             
             # Check bounds and distinctness
             if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                 continue
             
             # Find a point to move
             idx = random.randint(0, 11)
             old_x, old_y = current_points[idx]
             
             if (new_x, new_y) == (old_x, old_y):
                 continue
             if any(p[0] == new_x and p[1] == new_y for j, p in enumerate(current_points) if j != idx):
                 continue
                 
             current_points[idx] = [new_x, new_y]
             new_score = count_ordinary_lines(current_points)
             
             if new_score < current_score:
                 current_score = new_score
                 if new_score < best_score:
                     best_score = new_score
                     best_points = [list(p) for p in current_points]
             else:
                 current_points[idx] = [old_x, old_y]
                 
    return best_points
