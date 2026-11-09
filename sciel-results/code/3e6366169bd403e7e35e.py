import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    triples = list(combinations(range(30), 3))
    i_idx = np.array([t[0] for t in triples], dtype=np.int64)
    j_idx = np.array([t[1] for t in triples], dtype=np.int64)
    k_idx = np.array([t[2] for t in triples], dtype=np.int64)
    
    def fast_score(pts):
        xs = np.array([p[0] for p in pts], dtype=np.int64)
        ys = np.array([p[1] for p in pts], dtype=np.int64)
        xi = xs[i_idx]; yi = ys[i_idx]
        xj = xs[j_idx]; yj = ys[j_idx]
        xk = xs[k_idx]; yk = ys[k_idx]
        area2 = np.abs((xj - xi) * (yk - yi) - (yj - yi) * (xk - xi))
        mask = area2 > 0
        if not mask.any():
            return 0, 0
        unique, counts = np.unique(area2[mask], return_counts=True)
        max_idx = np.argmax(counts)
        return int(counts[max_idx]), int(unique[max_idx])
    
    def count_area(pts, target):
        xs = np.array([p[0] for p in pts], dtype=np.int64)
        ys = np.array([p[1] for p in pts], dtype=np.int64)
        xi = xs[i_idx]; yi = ys[i_idx]
        xj = xs[j_idx]; yj = ys[j_idx]
        xk = xs[k_idx]; yk = ys[k_idx]
        area2 = np.abs((xj - xi) * (yk - yi) - (yj - yi) * (xk - xi))
        return int(np.sum(area2 == target))
    
    def valid(pts):
        if len(pts) != 30 or len(set(pts)) != 30:
            return False
        return all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in pts)
    
    # Generate 2-line candidates
    candidates = []
    for a in range(10, 21):
        b = 30 - a
        for d in [1, 2, 3]:
            for h in [1, 2, 3, 4, 5]:
                max_x = max(a, b) * d
                if max_x > 60:
                    continue
                for x0 in range(-30, 31 - max_x):
                    for y0 in range(-30, 31 - h):
                        pts = [(x0 + i*d, y0) for i in range(a)] + [(x0 + i*d, y0+h) for i in range(b)]
                        if valid(pts):
                            candidates.append(pts)
    
    # Also try 2-line with different x-offsets for the two lines
    for a in range(10, 21):
        b = 30 - a
        for d in [1, 2, 3]:
            for h in [1, 2, 3, 4, 5]:
                for x0 in range(-30, 10):
                    for x1 in range(-30, 10):
                        for y0 in range(-30, 10):
                            pts = [(x0 + i*d, y0) for i in range(a)] + [(x1 + i*d, y0+h) for i in range(b)]
                            if valid(pts):
                                candidates.append(pts)
    
    random.shuffle(candidates)
    candidates = candidates[:300]
    
    best_points = None
    best_score = 0
    best_target = 0
    
    for pts in candidates:
        if time.time() > deadline - 30:
            break
        s, ta = fast_score(pts)
        if s > best_score:
            best_score = s
            best_points = pts[:]
            best_target = ta
    
    if best_points is None:
        best_points = [(i, 0) for i in range(15)] + [(i, 1) for i in range(15)]
        best_score, best_target = fast_score(best_points)
    
    def local_search(points, max_iters, time_limit):
        current = points[:]
        current_score, current_target = fast_score(current)
        
        for it in range(max_iters):
            if time.time() > time_limit:
                break
            
            idx = random.randint(0, 29)
            improved = False
            
            for _ in range(25):
                ox = random.randint(-6, 6)
                oy = random.randint(-6, 6)
                nx = current[idx][0] + ox
                ny = current[idx][1] + oy
                if not (-30 <= nx <= 30 and -30 <= ny <= 30):
                    continue
                np_ = (nx, ny)
                others = current[:idx] + current[idx+1:]
                if np_ in others:
                    continue
                cand = current[:]
                cand[idx] = np_
                ns, nt = fast_score(cand)
                if ns > current_score:
                    current = cand
                    current_score = ns
                    current_target = nt
                    improved = True
                    break
                elif ns == current_score and random.random() < 0.08:
                    current = cand
                    current_target = nt
                    improved = True
                    break
            
            if not improved:
                nx = random.randint(-30, 30)
                ny = random.randint(-30, 30)
                np_ = (nx, ny)
                others = current[:idx] + current[idx+1:]
                if np_ not in others:
                    cand = current[:]
                    cand[idx] = np_
                    ns, nt = fast_score(cand)
                    if ns > current_score:
                        current = cand
                        current_score = ns
                        current_target = nt
        
        return current, current_score
    
    if time.time() < deadline - 5:
        best_points, best_score = local_search(best_points, 8000, deadline - 3)
    
    for r in range(15):
        if time.time() > deadline - 5:
            break
        if r < len(candidates):
            sp = candidates[r]
        else:
            sp = []
            while len(sp) < 30:
                x = random.randint(-30, 30)
                y = random.randint(-30, 30)
                if (x, y) not in sp:
                    sp.append((x, y))
        if len(sp) != 30:
            continue
        rp, rs = local_search(sp, 4000, deadline - 3)
        if rs > best_score:
            best_score = rs
            best_points = rp
    
    assert len(best_points) == 30
    assert len(set(best_points)) == 30
    for x, y in best_points:
        assert -30 <= x <= 30 and -30 <= y <= 30
    
    return best_points
