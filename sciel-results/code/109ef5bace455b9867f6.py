import time
import random
from itertools import combinations
from math import gcd

def count_ordinary_lines(points):
    n = len(points)
    if n < 3:
        return 0
    lines = {}
    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        A = y1 - y2
        B = x2 - x1
        C = x1 * y2 - x2 * y1
        g = gcd(gcd(abs(A), abs(B)), abs(C))
        if g != 0:
            A //= g
            B //= g
            C //= g
        if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
            A = -A
            B = -B
            C = -C
        return (A, B, C)
    for i in range(n):
        for j in range(i + 1, n):
            key = get_line_key(points[i], points[j])
            lines[key] = lines.get(key, 0) + 1
    return sum(1 for v in lines.values() if v == 1)

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def construct():
    start_time = time.time()
    timeout = 110

    best_points = None
    best_score = float('inf')

    def try_config(config):
        nonlocal best_points, best_score
        if len(config) != 12 or len(set(map(tuple, config))) != 12:
            return
        if any(abs(p[0]) > 10**9 or abs(p[1]) > 10**9 for p in config):
            return
        s = count_ordinary_lines(config)
        if s < best_score:
            best_score = s
            best_points = [list(p) for p in config]

    # Try several structured configurations
    # 3x3 grid + 3 points
    base = [[x, y] for x in range(3) for y in range(3)]
    # Extend ordinary lines of 3x3 grid
    def get_ordinary_line_pairs(pts):
        n = len(pts)
        lines = {}
        for i in range(n):
            for j in range(i+1, n):
                key = get_line_key(pts[i], pts[j])
                if key not in lines:
                    lines[key] = []
                lines[key].append((i, j))
        return [(pts[i], pts[j]) for pairs in lines.values() if len(pairs) == 1 for i, j in pairs]

    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        A = y1 - y2
        B = x2 - x1
        C = x1 * y2 - x2 * y1
        g = gcd(gcd(abs(A), abs(B)), abs(C))
        if g != 0:
            A //= g; B //= g; C //= g
        if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
            A = -A; B = -B; C = -C
        return (A, B, C)

    ord_pairs = get_ordinary_line_pairs(base)
    candidates = set()
    for p1, p2 in ord_pairs:
        x1, y1 = p1; x2, y2 = p2
        dx, dy = x2-x1, y2-y1
        for k in [-3,-2,-1,1,2,3,4,5]:
            px, py = x1+k*dx, y1+k*dy
            if abs(px)<=10**9 and abs(py)<=10**9 and [px,py] not in base:
                candidates.add((px,py))
    cand_list = list(candidates)
    if len(cand_list) > 60:
        random.shuffle(cand_list)
        cand_list = cand_list[:60]

    # Exhaustive search over triples of candidates
    for i in range(len(cand_list)):
        if time.time() - start_time > timeout: break
        for j in range(i+1, len(cand_list)):
            if time.time() - start_time > timeout: break
            for k in range(j+1, len(cand_list)):
                cfg = base + [list(cand_list[i]), list(cand_list[j]), list(cand_list[k])]
                try_config(cfg)
                if best_score <= 6:
                    return best_points

    # Local search from best found
    if best_points is not None and time.time() - start_time < timeout - 5:
        improved = True
        while improved and time.time() - start_time < timeout - 5:
            improved = False
            cur = best_score
            swap_cands = set()
            for pt in best_points:
                x,y = pt
                for dx in range(-10,11):
                    for dy in range(-10,11):
                        nx,ny = x+dx, y+dy
                        if abs(nx)<=10**9 and abs(ny)<=10**9:
                            swap_cands.add((nx,ny))
            for pt in best_points:
                swap_cands.discard(tuple(pt))
            swap_cands = list(swap_cands)
            for idx in range(12):
                if time.time() - start_time > timeout: break
                for cand in swap_cands:
                    new_pts = best_points.copy()
                    new_pts[idx] = list(cand)
                    if len(set(map(tuple, new_pts))) != 12: continue
                    s = count_ordinary_lines(new_pts)
                    if s < cur:
                        cur = s; best_points = new_pts; best_score = s; improved = True
                        break
                if improved: break
            if best_score <= 6:
                break

    if best_points is None:
        best_points = base + [[3,0],[0,3],[3,3]]
    return best_points

# EVOLVE-BLOCK-END
