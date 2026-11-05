"""Search-based program for maximizing 4-point lines among 16 integer points."""
import time
import random

_START_TIME = time.time()
_DEADLINE = 150.0
_SAFETY = 5.0
_DEADLINE_EFF = _START_TIME + _DEADLINE - _SAFETY


def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def _line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    A = y1 - y2
    B = x2 - x1
    C = x1 * y2 - x2 * y1
    g = _gcd(_gcd(abs(A), abs(B)), abs(C)) if (A != 0 or B != 0) else 0
    if g > 0:
        A //= g
        B //= g
        C //= g
    if A < 0 or (A == 0 and B < 0):
        A = -A
        B = -B
        C = -C
    return (A, B, C)


def _count_4lines(points):
    n = 16
    line_points = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key not in line_points:
                line_points[key] = set()
            line_points[key].add(i)
            line_points[key].add(j)
    return sum(1 for pts in line_points.values() if len(pts) == 4)


def _has_5collinear(points):
    n = 16
    line_points = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key not in line_points:
                line_points[key] = set()
            line_points[key].add(i)
            line_points[key].add(j)
    return any(len(pts) >= 5 for pts in line_points.values())


def _is_valid(points):
    if len(points) != 16:
        return False
    if len(set(tuple(p) for p in points)) != 16:
        return False
    for x, y in points:
        if abs(x) > 1000 or abs(y) > 1000:
            return False
    if _has_5collinear(points):
        return False
    return True


def _grid4x4():
    return [[x, y] for x in range(4) for y in range(4)]


def construct():
    best_points = _grid4x4()
    best_score = _count_4lines(best_points)
    
    current = [p[:] for p in best_points]
    current_score = best_score
    
    T = 2.0
    T_min = 0.01
    cooling = 0.9995
    
    while time.time() < _DEADLINE_EFF:
        idx = random.randint(0, 15)
        ox, oy = current[idx]
        step = random.randint(1, 5)
        nx = ox + random.randint(-step, step)
        ny = oy + random.randint(-step, step)
        
        if abs(nx) > 1000 or abs(ny) > 1000:
            continue
        if any(current[k][0] == nx and current[k][1] == ny for k in range(16) if k != idx):
            continue
        
        current[idx] = [nx, ny]
        if _has_5collinear(current):
            current[idx] = [ox, oy]
            continue
        
        new_score = _count_4lines(current)
        
        if new_score >= current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in current]
        else:
            delta = new_score - current_score
            if random.random() < (2.718281828 ** (delta / T)):
                current_score = new_score
            else:
                current[idx] = [ox, oy]
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best_points
