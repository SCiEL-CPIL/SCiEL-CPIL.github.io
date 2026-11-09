import random
import time

def construct():
    deadline = time.time() + 130
    
    def count_edges(pts):
        n = len(pts)
        edges = 0
        for i in range(n):
            ai, bi = pts[i]
            for j in range(i + 1, n):
                aj, bj = pts[j]
                da = ai - aj
                db = bi - bj
                if da * da + da * db + db * db == 1:
                    edges += 1
        return edges
    
    def is_valid(pts):
        n = len(pts)
        for i in range(n):
            ai, bi = pts[i]
            if abs(ai) > 64 or abs(bi) > 64:
                return False
            for j in range(i + 1, n):
                aj, bj = pts[j]
                da = ai - aj
                db = bi - bj
                if da * da + da * db + db * db == 0:
                    return False
        return True
    
    def make_spiral(n):
        pts = [(0, 0)]
        if n <= 1:
            return pts
        ring = 1
        while len(pts) < n:
            pos = [ring, 0]
            pts.append(tuple(pos))
            if len(pts) >= n:
                break
            dirs = [(0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1), (1, 0)]
            for d in dirs:
                for _ in range(ring):
                    pos[0] += d[0]
                    pos[1] += d[1]
                    pts.append(tuple(pos))
                    if len(pts) >= n:
                        break
                if len(pts) >= n:
                    break
            ring += 1
        return pts[:n]
    
    points = [list(p) for p in make_spiral(32)]
    best_points = [list(p) for p in points]
    best_score = count_edges(best_points)
    
    random.seed(98765)
    
    moves = []
    for dx in range(-3, 4):
        for dy in range(-3, 4):
            d2 = dx * dx + dx * dy + dy * dy
            if 1 <= d2 <= 9:
                moves.append((dx, dy))
    
    no_improve = 0
    while time.time() < deadline:
        improved = False
        
        for i in range(32):
            if time.time() > deadline:
                break
            orig = points[i]
            for dx, dy in moves:
                np_ = [orig[0] + dx, orig[1] + dy]
                if abs(np_[0]) > 64 or abs(np_[1]) > 64:
                    continue
                points[i] = np_
                if not is_valid(points):
                    points[i] = orig
                    continue
                s = count_edges(points)
                if s > best_score:
                    best_score = s
                    best_points = [list(p) for p in points]
                    improved = True
                else:
                    points[i] = orig
            points[i] = orig
        
        for i in range(32):
            if time.time() > deadline:
                break
            for j in range(i + 1, 32):
                if time.time() > deadline:
                    break
                oi = list(points[i])
                oj = list(points[j])
                points[i] = oj
                points[j] = oi
                if not is_valid(points):
                    points[i] = oi
                    points[j] = oj
                    continue
                s = count_edges(points)
                if s > best_score:
                    best_score = s
                    best_points = [list(p) for p in points]
                    improved = True
                else:
                    points[i] = oi
                    points[j] = oj
        
        if improved:
            no_improve = 0
        else:
            no_improve += 1
        
        if no_improve > 30:
            if random.random() < 0.5:
                points = [list(p) for p in best_points]
                for _ in range(3):
                    idx = random.randint(0, 31)
                    dx, dy = random.choice(moves)
                    np_ = [points[idx][0] + dx, points[idx][1] + dy]
                    if abs(np_[0]) <= 64 and abs(np_[1]) <= 64:
                        points[idx] = np_
                if is_valid(points):
                    s = count_edges(points)
                    if s >= best_score - 5:
                        no_improve = 0
                    else:
                        points = [list(p) for p in best_points]
                else:
                    points = [list(p) for p in best_points]
            else:
                break
    
    return [list(p) for p in best_points]
