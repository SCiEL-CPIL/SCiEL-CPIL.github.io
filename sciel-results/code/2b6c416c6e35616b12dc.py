"""Optimized search for 30 points maximizing max multiplicity of a doubled area."""
import numpy as np
import itertools
import time
from collections import Counter

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
from collections import Counter

def _max_multiplicity(points):
    """Compute the maximum multiplicity of any positive doubled area."""
    n = len(points)
    pts = np.array(points, dtype=np.int64)
    best = 0
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                # doubled area = |det|
                area = abs(pts[i,0]*(pts[j,1]-pts[k,1]) + pts[j,0]*(pts[k,1]-pts[i,1]) + pts[k,0]*(pts[i,1]-pts[j,1]))
                if area > 0:
                    if area > best:
                        best = area
    return best

def _max_multiplicity_count(points):
    """Compute the maximum count of any single positive doubled area value."""
    n = len(points)
    pts = np.array(points, dtype=np.int64)
    areas = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                area = abs(pts[i,0]*(pts[j,1]-pts[k,1]) + pts[j,0]*(pts[k,1]-pts[i,1]) + pts[k,0]*(pts[i,1]-pts[j,1]))
                if area > 0:
                    areas.append(area)
    if not areas:
        return 0
    return max(Counter(areas).values())

def _eval(points):
    """Score: max multiplicity of a single positive doubled area."""
    n = len(points)
    if n != 30:
        return 0
    pts = np.array(points, dtype=np.int64)
    areas = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                area = abs(pts[i,0]*(pts[j,1]-pts[k,1]) + pts[j,0]*(pts[k,1]-pts[i,1]) + pts[k,0]*(pts[i,1]-pts[j,1]))
                if area > 0:
                    areas.append(area)
    if not areas:
        return 0
    return max(Counter(areas).values())

def _random_points():
    """Generate 30 distinct random points in [-30,30]^2."""
    pts = set()
    while len(pts) < 30:
        x = np.random.randint(-30, 31)
        y = np.random.randint(-30, 31)
        pts.add((x, y))
    return list(pts)

def _hill_climb(points, time_limit=50.0):
    """Local search: try swapping one point with a random point."""
    start = time.time()
    current = list(points)
    current_score = _eval(current)
    best = current
    best_score = current_score
    n = len(current)
    pts_arr = np.array(current, dtype=np.int64)
    
    while time.time() - start < time_limit:
        # Pick a random index to replace
        idx = np.random.randint(n)
        old_x, old_y = current[idx]
        new_x = np.random.randint(-30, 31)
        new_y = np.random.randint(-30, 31)
        if (new_x, new_y) == (old_x, old_y):
            continue
        # Check distinctness
        if any(current[i] == (new_x, new_y) for i in range(n) if i != idx):
            continue
        # Evaluate
        trial = list(current)
        trial[idx] = (new_x, new_y)
        s = _eval(trial)
        if s > current_score:
            current = trial
            current_score = s
            if s > best_score:
                best = list(current)
                best_score = s
    return best, best_score

def construct():
    deadline = time.time() + 55.0  # 55s safety margin
    
    # Start with a structured initial guess: grid pattern
    # Try a few structured starts
    best_points = None
    best_score = 0
    
    # Initial structured candidate: 6x5 grid
    grid = [(x, y) for x in range(6) for y in range(5)]
    s = _eval(grid)
    if s > best_score:
        best_score = s
        best_points = grid
    
    # Random restarts with hill climbing
    np.random.seed(42)
    n_restarts = 0
    while time.time() < deadline:
        # Random start
        pts = _random_points()
        s = _eval(pts)
        if s > best_score:
            best_score = s
            best_points = list(pts)
        
        # Hill climb from this start
        remaining = deadline - time.time()
        if remaining > 5.0:
            hc_pts, hc_score = _hill_climb(pts, time_limit=min(remaining - 3.0, 20.0))
            if hc_score > best_score:
                best_score = hc_score
                best_points = hc_pts
        
        n_restarts += 1
        # If we're doing well, do more focused search
        if best_score >= 100 and time.time() < deadline - 10.0:
            # Try perturbing the best solution
            trial = list(best_points)
            idx = np.random.randint(30)
            old_x, old_y = trial[idx]
            new_x = old_x + np.random.randint(-3, 4)
            new_y = old_y + np.random.randint(-3, 4)
            new_x = max(-30, min(30, new_x))
            new_y = max(-30, min(30, new_y))
            if (new_x, new_y) != (old_x, old_y):
                if not any(trial[i] == (new_x, new_y) for i in range(30) if i != idx):
                    trial[idx] = (new_x, new_y)
                    s = _eval(trial)
                    if s > best_score:
                        best_score = s
                        best_points = trial
    
    # Final check
    if best_points is None:
        return [(x, y) for x in range(6) for y in range(5)]
    return best_points
# EVOLVE-BLOCK-END
