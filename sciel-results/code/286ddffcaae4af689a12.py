"""Rewritten program: maximize largest multiplicity of a positive doubled area among 30 points."""
import time
import itertools
from collections import Counter

DEADLINE = 85.0  # safety margin before 90s kill

def _doubled_areas(points):
    """Compute all positive doubled areas for triples of points."""
    n = len(points)
    areas = []
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i+1, n):
            x2, y2 = points[j]
            for k in range(j+1, n):
                x3, y3 = points[k]
                det = abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
                if det > 0:
                    areas.append(det)
    return areas

def _max_multiplicity(points):
    """Return the maximum multiplicity of any positive doubled area."""
    areas = _doubled_areas(points)
    if not areas:
        return 0
    return Counter(areas).most_common(1)[0][1]

def _valid(points):
    """Check that points are 30 distinct integer pairs in [-30,30]^2."""
    if len(points) != 30:
        return False
    seen = set()
    for p in points:
        if not (isinstance(p, tuple) and len(p) == 2):
            return False
        x, y = p
        if not (isinstance(x, int) and isinstance(y, int)):
            return False
        if x < -30 or x > 30 or y < -30 or y > 30:
            return False
        if p in seen:
            return False
        seen.add(p)
    return True

def _grid_family(cx, cy, w, h, step=1):
    """Generate a grid of points centered at (cx,cy) with width w, height h."""
    pts = []
    for i in range(w):
        for j in range(h):
            x = cx - w//2 + i*step
            y = cy - h//2 + j*step
            pts.append((x, y))
    return pts

def _try_local_search(points, time_limit, rng):
    """Local search: perturb random points and keep improvements."""
    best = list(points)
    best_score = _max_multiplicity(best)
    start = time.time()
    while time.time() - start < time_limit:
        # Pick a random point to move
        idx = rng.randint(0, 29)
        old_x, old_y = best[idx]
        # Try small perturbation
        dx = rng.randint(-3, 3)
        dy = rng.randint(-3, 3)
        new_x = old_x + dx
        new_y = old_y + dy
        if new_x < -30 or new_x > 30 or new_y < -30 or new_y > 30:
            continue
        new_pt = (new_x, new_y)
        # Check distinctness
        if any(new_pt == best[i] for i in range(30) if i != idx):
            continue
        best[idx] = new_pt
        score = _max_multiplicity(best)
        if score > best_score:
            best_score = score
        else:
            best[idx] = (old_x, old_y)
    return best, best_score

def construct():
    start_time = time.time()
    import random
    rng = random.Random(42)
    
    best_points = [(x, y) for x in range(6) for y in range(5)]
    best_score = _max_multiplicity(best_points)
    
    # Try various structured families
    families = []
    
    # 6x5 grid (baseline)
    families.append([(x, y) for x in range(6) for y in range(5)])
    
    # Various grid sizes with different spacings
    for w, h in [(5, 6), (6, 5), (10, 3), (15, 2), (30, 1), (1, 30), (3, 10), (2, 15)]:
        for step in [1, 2]:
            for cx in range(0, 31, 5):
                for cy in range(0, 31, 5):
                    pts = []
                    valid = True
                    for i in range(w):
                        for j in range(h):
                            x = cx - (w-1)*step//2 + i*step
                            y = cy - (h-1)*step//2 + j*step
                            if x < -30 or x > 30 or y < -30 or y > 30:
                                valid = False
                                break
                            pts.append((x, y))
                        if not valid:
                            break
                    if valid and len(pts) == 30 and len(set(pts)) == 30:
                        families.append(pts)
    
    # Diagonal lines
    for offset in range(-20, 21, 2):
        pts = []
        for i in range(30):
            x = i
            y = offset + i
            if -30 <= x <= 30 and -30 <= y <= 30:
                pts.append((x, y))
        if len(pts) == 30 and len(set(pts)) == 30:
            families.append(pts)
    
    # Parabolic
    for a in [1, 2, 3, -1, -2, -3]:
        for c in range(-10, 11, 2):
            pts = []
            for i in range(30):
                x = i - 15
                y = a * x * x + c
                if -30 <= x <= 30 and -30 <= y <= 30:
                    pts.append((x, y))
            if len(pts) == 30 and len(set(pts)) == 30:
                families.append(pts)
    
    # Hyperbolic-like
    for k in [5, 10, 15, 20, -5, -10, -15, -20]:
        pts = []
        for i in range(30):
            x = i - 15
            if x != 0:
                y = k // x
            else:
                y = 0
            if -30 <= x <= 30 and -30 <= y <= 30:
                pts.append((x, y))
        if len(pts) == 30 and len(set(pts)) == 30:
            families.append(pts)
    
    # Random structured: pick from small lattice
    for trial in range(50):
        pts = set()
        while len(pts) < 30:
            x = rng.randint(-15, 15)
            y = rng.randint(-15, 15)
            pts.add((x, y))
        if len(pts) == 30:
            families.append(list(pts))
    
    # Evaluate all families
    for pts in families:
        if time.time() - start_time > DEADLINE - 5:
            break
        if not _valid(pts):
            continue
        score = _max_multiplicity(pts)
        if score > best_score:
            best_score = score
            best_points = list(pts)
    
    # Local search from best
    if time.time() - start_time < DEADLINE - 3:
        remaining = DEADLINE - 3 - (time.time() - start_time)
        best_points, best_score = _try_local_search(best_points, remaining, rng)
    
    # Final local search with fresh random restarts if time allows
    if time.time() - start_time < DEADLINE - 2:
        remaining = DEADLINE - 2 - (time.time() - start_time)
        # Try a few random restarts
        for _ in range(3):
            if time.time() - start_time > DEADLINE - 1:
                break
            rand_pts = []
            while len(rand_pts) < 30:
                x = rng.randint(-20, 20)
                y = rng.randint(-20, 20)
                if (x, y) not in rand_pts:
                    rand_pts.append((x, y))
            if _valid(rand_pts):
                rp, rs = _try_local_search(rand_pts, min(remaining/3, 5.0), rng)
                if rs > best_score:
                    best_score = rs
                    best_points = rp
    
    return best_points

# EVOLVE-BLOCK-START
# (kept for compatibility)
# EVOLVE-BLOCK-END
