import time
from itertools import combinations
from collections import Counter

def _doubled_area(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))

def _score(points):
    areas = Counter()
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            for k in range(j+1, len(points)):
                d = _doubled_area(points[i], points[j], points[k])
                if d > 0:
                    areas[d] += 1
    if not areas:
        return 0
    return max(areas.values())

def _make_grid(m, n, x0=0, y0=0):
    """m columns, n rows, starting at (x0,y0)"""
    return [(x0 + i, y0 + j) for i in range(m) for j in range(n)]

def construct():
    start = time.time()
    deadline = start + 80  # safety margin before 90s
    
    best_score = 0
    best_points = None
    
    # Try all rectangular grid factorizations of 30
    factors = [(1,30),(2,15),(3,10),(5,6),(6,5),(10,3),(15,2),(30,1)]
    
    for (m, n) in factors:
        # Try different offsets to center in [-30,30]
        for x0 in range(-30, 31 - m + 1):
            for y0 in range(-30, 31 - n + 1):
                if time.time() > deadline:
                    break
                pts = _make_grid(m, n, x0, y0)
                # Quick check: all in bounds
                if all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in pts):
                    s = _score(pts)
                    if s > best_score:
                        best_score = s
                        best_points = pts
        
        if time.time() > deadline:
            break
    
    if best_points is None:
        # Fallback
        best_points = [(x, y) for x in range(6) for y in range(5)]
        best_score = _score(best_points)
    
    return best_points
