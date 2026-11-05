import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    def has_four_collinear(points):
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    p1, p2, p3 = points[i], points[j], points[k]
                    # Check if p1, p2, p3 are collinear
                    v1 = (p2[0]-p1[0], p2[1]-p1[1])
                    v2 = (p3[0]-p1[0], p3[1]-p1[1])
                    if v1[0]*v2[1] - v1[1]*v2[0] != 0:
                        continue
                    # They are collinear. Check for a 4th point
                    A = p2[1] - p1[1]
                    B = p1[0] - p2[0]
                    C = p2[0]*p1[1] - p1[0]*p2[1]
                    for l in range(n):
                        if l in (i, j, k):
                            continue
                        if A*points[l][0] + B*points[l][1] + C == 0:
                            return True
        return False

    def count_triples(points):
        n = len(points)
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    p1, p2, p3 = points[i], points[j], points[k]
                    v1 = (p2[0]-p1[0], p2[1]-p1[1])
                    v2 = (p3[0]-p1[0], p3[1]-p1[1])
                    if v1[0]*v2[1] - v1[1]*v2[0] == 0:
                        count += 1
        return count

    def random_points():
        pts = []
        used = set()
        while len(pts) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            if (x, y) not in used:
                used.add((x, y))
                pts.append([x, y])
        return pts

    best_p = None
    best_count = -1
    
    # Try a few random restarts
    for _ in range(5):
        if time.time() > deadline:
            break
        p = random_points()
        if has_four_collinear(p):
            continue
            
        current_count = count_triples(p)
        if current_count > best_count:
            best_count = current_count
            best_p = p[:]
            
        # Hill Climb
        while time.time() < deadline:
            improved = False
            for idx in range(20):
                if time.time() > deadline:
                    break
                old_pt = p[idx]
                # Try several random moves
                for _ in range(20):
                    if time.time() > deadline:
                        break
                    new_x = random.randint(-100, 100)
                    new_y = random.randint(-100, 100)
                    new_pt = [new_x, new_y]
                    if any(pt == new_pt for i, pt in enumerate(p) if i != idx):
                        continue
                    p[idx] = new_pt
                    if not has_four_collinear(p):
                        new_count = count_triples(p)
                        if new_count > current_count:
                            current_count = new_count
                            improved = True
                            break
                        else:
                            p[idx] = old_pt
                    else:
                        p[idx] = old_pt
                if improved:
                    break
            if not improved:
                break
                
        if current_count > best_count:
            best_count = current_count
            best_p = p[:]

    if best_p is None:
        # Fallback
        return [[x, x * x] for x in range(-10, 10)]
        
    return best_p
