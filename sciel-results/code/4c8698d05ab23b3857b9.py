import time
import random
from collections import Counter

def _full_score(points):
    areas = []
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i+1, n):
            x2, y2 = points[j]
            for k in range(j+1, n):
                x3, y3 = points[k]
                d = abs((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1))
                if d > 0:
                    areas.append(d)
    if not areas:
        return 0
    return max(Counter(areas).values())

def _initial_grid():
    return [(x, y) for x in range(6) for y in range(5)]

def construct():
    start_time = time.time()
    deadline = start_time + 85
    random.seed(123)
    
    points = _initial_grid()
    n = 30
    
    area_count = Counter()
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i+1, n):
            x2, y2 = points[j]
            for k in range(j+1, n):
                x3, y3 = points[k]
                d = abs((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1))
                if d > 0:
                    area_count[d] += 1
    
    best_score = max(area_count.values()) if area_count else 0
    best_points = list(points)
    current_score = best_score
    
    def get_best():
        return max(area_count.values()) if area_count else 0
    
    def remove_triples_with_idx(idx):
        pts = points
        for a in range(n):
            if a == idx:
                continue
            for b in range(a+1, n):
                if b == idx:
                    continue
                x1, y1 = pts[a]
                x2, y2 = pts[b]
                x3, y3 = pts[idx]
                d = abs((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1))
                if d > 0:
                    area_count[d] -= 1
                    if area_count[d] == 0:
                        del area_count[d]
    
    def add_triples_with_idx(idx):
        pts = points
        for a in range(n):
            if a == idx:
                continue
            for b in range(a+1, n):
                if b == idx:
                    continue
                x1, y1 = pts[a]
                x2, y2 = pts[b]
                x3, y3 = pts[idx]
                d = abs((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1))
                if d > 0:
                    area_count[d] += 1
    
    def move_point(idx, new_pos):
        old_pos = points[idx]
        if old_pos == new_pos:
            return
        remove_triples_with_idx(idx)
        points[idx] = new_pos
        add_triples_with_idx(idx)
    
    def swap_points(i1, i2):
        if i1 == i2:
            return
        p1 = points[i1]
        p2 = points[i2]
        remove_triples_with_idx(i1)
        for a in range(n):
            if a == i2:
                continue
            for b in range(a+1, n):
                if b == i2:
                    continue
                if a == i1 or b == i1:
                    continue
                x1, y1 = points[a]
                x2, y2 = points[b]
                x3, y3 = points[i2]
                d = abs((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1))
                if d > 0:
                    area_count[d] -= 1
                    if area_count[d] == 0:
                        del area_count[d]
        points[i1] = p2
        points[i2] = p1
        add_triples_with_idx(i1)
        for a in range(n):
            if a == i2:
                continue
            for b in range(a+1, n):
                if b == i2:
                    continue
                if a == i1 or b == i1:
                    continue
                x1, y1 = points[a]
                x2, y2 = points[b]
                x3, y3 = points[i2]
                d = abs((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1))
                if d > 0:
                    area_count[d] += 1
    
    while time.time() < deadline:
        for _ in range(100):
            if time.time() > deadline:
                break
            if random.random() < 0.7:
                idx = random.randint(0, n-1)
                ox, oy = points[idx]
                for _ in range(10):
                    if time.time() > deadline:
                        break
                    dx = random.randint(-4, 4)
                    dy = random.randint(-4, 4)
                    nx, ny = ox + dx, oy + dy
                    if -30 <= nx <= 30 and -30 <= ny <= 30:
                        new_pos = (nx, ny)
                        if new_pos in points and new_pos != (ox, oy):
                            continue
                        move_point(idx, new_pos)
                        new_score = get_best()
                        if new_score > current_score:
                            current_score = new_score
                            best_score = new_score
                            best_points = list(points)
                        else:
                            move_point(idx, (ox, oy))
            else:
                i1 = random.randint(0, n-1)
                i2 = random.randint(0, n-1)
                if i1 == i2:
                    continue
                swap_points(i1, i2)
                new_score = get_best()
                if new_score > current_score:
                    current_score = new_score
                    best_score = new_score
                    best_points = list(points)
                else:
                    swap_points(i1, i2)
        if random.random() < 0.05:
            points = list(best_points)
            for _ in range(2):
                idx = random.randint(0, n-1)
                ox, oy = points[idx]
                dx = random.randint(-3, 3)
                dy = random.randint(-3, 3)
                nx, ny = ox + dx, oy + dy
                if -30 <= nx <= 30 and -30 <= ny <= 30:
                    if (nx, ny) not in points:
                        points[idx] = (nx, ny)
            area_count = Counter()
            for i in range(n):
                x1, y1 = points[i]
                for j in range(i+1, n):
                    x2, y2 = points[j]
                    for k in range(j+1, n):
                        x3, y3 = points[k]
                        d = abs((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1))
                        if d > 0:
                            area_count[d] += 1
            current_score = max(area_count.values()) if area_count else 0
    
    return best_points
