import time
import random
import collections

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    N = 30
    COORD_MIN = -30
    COORD_MAX = 30
    
    def get_initial_grid():
        pts = []
        for x in range(6):
            for y in range(5):
                pts.append((x, y))
        return pts

    def get_initial_random():
        pts = set()
        while len(pts) < N:
            x = random.randint(COORD_MIN, COORD_MAX)
            y = random.randint(COORD_MIN, COORD_MAX)
            pts.add((x, y))
        return list(pts)

    def calculate_areas(points):
        counts = collections.Counter()
        n = len(points)
        for i in range(n):
            x1, y1 = points[i]
            for j in range(i+1, n):
                x2, y2 = points[j]
                for k in range(j+1, n):
                    x3, y3 = points[k]
                    area = abs(x1*(y2 - y3) - y1*(x2 - x3) + x2*y3 - x3*y2)
                    if area > 0:
                        counts[area] += 1
        return counts

    def get_score_from_counts(counts):
        if not counts:
            return 0
        return max(counts.values())

    best_score = 0
    best_points = None
    
    def local_search(points):
        nonlocal best_score, best_points
        
        current_points = [p for p in points]
        current_counts = calculate_areas(current_points)
        current_score = get_score_from_counts(current_counts)
        
        if current_score > best_score:
            best_score = current_score
            best_points = [p for p in current_points]

        triples_with = [[] for _ in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                for k in range(j+1, N):
                    triples_with[i].append((j, k))
                    triples_with[j].append((i, k))
                    triples_with[k].append((i, j))
                    
        T = 1.0
        cooling = 0.9995
        min_T = 0.01
        
        while time.time() < deadline - 2.0:
            idx = random.randint(0, N-1)
            old_x, old_y = current_points[idx]
            
            if random.random() < 0.8:
                dx = random.randint(-2, 2)
                dy = random.randint(-2, 2)
                new_x = old_x + dx
                new_y = old_y + dy
            else:
                new_x = random.randint(COORD_MIN, COORD_MAX)
                new_y = random.randint(COORD_MIN, COORD_MAX)
                
            if new_x < COORD_MIN or new_x > COORD_MAX or new_y < COORD_MIN or new_y > COORD_MAX:
                continue
                
            is_valid = True
            for i, p in enumerate(current_points):
                if i != idx and p == (new_x, new_y):
                    is_valid = False
                    break
            
            if not is_valid:
                continue
                
            # Update counts: Remove old
            for j, k in triples_with[idx]:
                x2, y2 = current_points[j]
                x3, y3 = current_points[k]
                old_area = abs(old_x*(y2 - y3) - old_y*(x2 - x3) + x2*y3 - x3*old_y)
                if old_area > 0:
                    current_counts[old_area] -= 1
                    if current_counts[old_area] == 0:
                        del current_counts[old_area]
                        
            # Apply move
            current_points[idx] = (new_x, new_y)
            
            # Update counts: Add new
            for j, k in triples_with[idx]:
                x2, y2 = current_points[j]
                x3, y3 = current_points[k]
                new_area = abs(new_x*(y2 - y3) - new_y*(x2 - x3) + x2*y3 - x3*new_y)
                if new_area > 0:
                    current_counts[new_area] += 1
                    
            new_score = get_score_from_counts(current_counts)
            
            accepted = False
            if new_score > current_score:
                accepted = True
            elif new_score == current_score:
                if random.random() < 0.1:
                    accepted = True
            else:
                delta = new_score - current_score
                if random.random() < pow(2.71828, delta / T):
                    accepted = True

            if accepted:
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = [p for p in current_points]
            else:
                # Revert
                current_points[idx] = (old_x, old_y)
                # Remove NEW
                for j, k in triples_with[idx]:
                    x2, y2 = current_points[j]
                    x3, y3 = current_points[k]
                    new_area = abs(new_x*(y2 - y3) - new_y*(x2 - x3) + x2*y3 - x3*new_y)
                    if new_area > 0:
                        current_counts[new_area] -= 1
                        if current_counts[new_area] == 0:
                            del current_counts[new_area]
                # Add OLD
                for j, k in triples_with[idx]:
                    x2, y2 = current_points[j]
                    x3, y3 = current_points[k]
                    old_area = abs(old_x*(y2 - y3) - old_y*(x2 - x3) + x2*y3 - x3*old_y)
                    if old_area > 0:
                        current_counts[old_area] += 1

            T *= cooling
            if T < min_T:
                T = min_T

    # Main loop
    inits = [get_initial_grid, get_initial_random]
    
    while time.time() < deadline - 5.0:
        init_fn = random.choice(inits)
        pts = init_fn()
        local_search(pts)

    if best_points is None:
        return [(x, y) for x in range(6) for y in range(5)]
        
    return best_points
