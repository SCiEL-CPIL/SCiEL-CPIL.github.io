"""Rewritten program: search for 30 points maximizing multiplicity of a doubled area."""
import time
import itertools
import random
from collections import Counter

def _doubled_area(p1, p2, p3):
    """Return absolute doubled area (determinant) of triangle p1,p2,p3."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))

def _score(points):
    """Compute the largest multiplicity of a positive doubled area among all triples."""
    n = len(points)
    if n < 3:
        return 0
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = _doubled_area(points[i], points[j], points[k])
                if a > 0:
                    areas.append(a)
    if not areas:
        return 0
    c = Counter(areas)
    return max(c.values())

def _all_triples_areas(points):
    """Return list of all positive doubled areas for all triples."""
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = _doubled_area(points[i], points[j], points[k])
                if a > 0:
                    areas.append(a)
    return areas

def _local_search(points, max_iters=500, deadline=None):
    """Improve a set of points via local search: move one point to a better position."""
    points = list(points)
    n = len(points)
    best_score = _score(points)
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        improved = False
        # Try moving each point to a nearby random position
        for idx in range(n):
            if deadline and time.time() > deadline:
                break
            orig = points[idx]
            # Try several candidate positions
            for _ in range(8):
                nx = orig[0] + random.randint(-3, 3)
                ny = orig[1] + random.randint(-3, 3)
                if -30 <= nx <= 30 and -30 <= ny <= 30:
                    new_pt = (nx, ny)
                    if new_pt not in points:
                        points[idx] = new_pt
                        s = _score(points)
                        if s > best_score:
                            best_score = s
                            improved = True
                            break
                        else:
                            points[idx] = orig
            if improved:
                break
        if not improved:
            # Try larger jumps
            for _ in range(3):
                if deadline and time.time() > deadline:
                    break
                idx = random.randint(0, n-1)
                orig = points[idx]
                for _ in range(12):
                    nx = random.randint(-30, 30)
                    ny = random.randint(-30, 30)
                    new_pt = (nx, ny)
                    if new_pt not in points:
                        points[idx] = new_pt
                        s = _score(points)
                        if s > best_score:
                            best_score = s
                            improved = True
                            break
                        else:
                            points[idx] = orig
                if improved:
                    break
            if not improved:
                break
    return points, best_score

def construct():
    deadline = time.time() + 80  # 80s safety margin within 90s limit
    
    # Start with a grid-like configuration
    best_points = [(x, y) for x in range(6) for y in range(5)]
    best_score = _score(best_points)
    
    # Try multiple random restarts with local search
    random.seed(42)
    
    for restart in range(200):
        if time.time() > deadline:
            break
        
        # Generate a random starting configuration
        if restart == 0:
            pts = best_points[:]
        else:
            # Mix of structured and random starts
            if restart % 3 == 0:
                # Grid with some perturbation
                pts = []
                for x in range(6):
                    for y in range(5):
                        px = x + random.randint(-1, 1)
                        py = y + random.randint(-1, 1)
                        px = max(-30, min(30, px))
                        py = max(-30, min(30, py))
                        pts.append((px, py))
                # Ensure distinctness
                seen = set()
                clean = []
                for p in pts:
                    if p not in seen:
                        seen.add(p)
                        clean.append(p)
                while len(clean) < 30:
                    p = (random.randint(-30, 30), random.randint(-30, 30))
                    if p not in seen:
                        seen.add(p)
                        clean.append(p)
                pts = clean
            elif restart % 3 == 1:
                # Random points
                pts = []
                seen = set()
                while len(pts) < 30:
                    p = (random.randint(-30, 30), random.randint(-30, 30))
                    if p not in seen:
                        seen.add(p)
                        pts.append(p)
            else:
                # Structured: points on a few lines or lattice
                pts = []
                seen = set()
                # Pick a small set of x-coordinates and y-coordinates
                xs = random.sample(range(-30, 31), 6)
                ys = random.sample(range(-30, 31), 5)
                for x in xs:
                    for y in ys:
                        pts.append((x, y))
                        seen.add((x, y))
            
            # Local search
            pts, s = _local_search(pts, max_iters=200, deadline=deadline)
            if s > best_score:
                best_score = s
                best_points = pts
        
        # Periodically do a deeper local search on the best
        if restart % 10 == 0 and best_points:
            pts, s = _local_search(best_points[:], max_iters=300, deadline=deadline)
            if s > best_score:
                best_score = s
                best_points = pts
    
    # Final polish
    if time.time() < deadline - 5:
        pts, s = _local_search(best_points[:], max_iters=500, deadline=deadline)
        if s > best_score:
            best_score = s
            best_points = pts
    
    return best_points
