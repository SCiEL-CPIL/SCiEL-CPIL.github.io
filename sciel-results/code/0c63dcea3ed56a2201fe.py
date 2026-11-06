import time
import itertools
import random
import numpy as np
from collections import Counter

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 75  # safety margin before 90s kill
    
    def score_points(points):
        """Compute the largest multiplicity of a single positive doubled area."""
        n = len(points)
        areas = []
        for i in range(n):
            xi, yi = points[i]
            for j in range(i+1, n):
                xj, yj = points[j]
                dx1 = xj - xi
                dy1 = yj - yi
                for k in range(j+1, n):
                    xk, yk = points[k]
                    dx2 = xk - xi
                    dy2 = yk - yi
                    det = abs(dx1 * dy2 - dy1 * dx2)
                    if det > 0:
                        areas.append(det)
        if not areas:
            return 0
        c = Counter(areas)
        return max(c.values())
    
    def score_points_fast(points):
        """Faster scoring using numpy vectorization where possible."""
        n = len(points)
        pts = np.array(points, dtype=np.int64)
        areas = []
        for i in range(n):
            pi = pts[i]
            for j in range(i+1, n):
                pj = pts[j]
                v1 = pj - pi
                for k in range(j+1, n):
                    pk = pts[k]
                    v2 = pk - pi
                    det = abs(v1[0]*v2[1] - v1[1]*v2[0])
                    if det > 0:
                        areas.append(det)
        if not areas:
            return 0
        c = Counter(areas)
        return max(c.values())
    
    # Start with a good initial configuration
    # Use a grid-like pattern that tends to produce many equal areas
    initial = [(x, y) for x in range(6) for y in range(5)]
    best_score = score_points(initial)
    best_points = list(initial)
    
    # Try several different initial configurations
    random.seed(42)
    
    # Configuration 2: scaled grid
    for scale in [1, 2, 3]:
        config = [(x*scale, y*scale) for x in range(6) for y in range(5)]
        # Check bounds
        if all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in config):
            s = score_points(config)
            if s > best_score:
                best_score = s
                best_points = config
    
    # Configuration 3: centered grid
    config = [(x-2, y-2) for x in range(6) for y in range(5)]
    if all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in config):
        s = score_points(config)
        if s > best_score:
            best_score = s
            best_points = config
    
    # Local search: swap points to improve score
    def local_search(points, deadline):
        points = list(points)
        current_score = score_points(points)
        n = len(points)
        
        while time.time() < deadline:
            improved = False
            # Try swapping two points
            for i in range(n):
                if time.time() >= deadline:
                    break
                for j in range(i+1, n):
                    # Try swapping points i and j
                    new_points = points.copy()
                    new_points[i], new_points[j] = points[j], points[i]
                    s = score_points(new_points)
                    if s > current_score:
                        current_score = s
                        points = new_points
                        improved = True
                        break
                if time.time() >= deadline:
                    break
                if improved:
                    break
            if not improved:
                break
        return points, current_score
    
    # Do a few local search iterations
    for attempt in range(3):
        if time.time() >= deadline:
            break
        # Start from best or a perturbed version
        if attempt == 0:
            start = best_points
        else:
            start = list(best_points)
            # Randomly perturb a few points
            for _ in range(3):
                idx = random.randint(0, len(start)-1)
                new_pt = (random.randint(-30, 30), random.randint(-30, 30))
                # Ensure distinctness
                if new_pt not in start or start[idx] == new_pt:
                    start[idx] = new_pt
        
        pts, sc = local_search(start, deadline)
        if sc > best_score:
            best_score = sc
            best_points = pts
    
    # Try some structured configurations
    # Parallelogram-based: points on a lattice
    for base_x in range(-10, 11, 5):
        for base_y in range(-10, 11, 5):
            if time.time() >= deadline:
                break
            config = []
            for i in range(6):
                for j in range(5):
                    x = base_x + i * 3
                    y = base_y + j * 3
                    if -30 <= x <= 30 and -30 <= y <= 30:
                        config.append((x, y))
            if len(config) == 30:
                s = score_points(config)
                if s > best_score:
                    best_score = s
                    best_points = config
    
    # Try hexagonal-like lattice
    for offset in [0, 1, 2]:
        if time.time() >= deadline:
            break
        config = []
        for row in range(5):
            for col in range(6):
                x = col * 4 + (row % 2) * 2 + offset
                y = row * 4
                if -30 <= x <= 30 and -30 <= y <= 30:
                    config.append((x, y))
        if len(config) == 30:
            s = score_points(config)
            if s > best_score:
                best_score = s
                best_points = config
    
    # Final local search from best
    if time.time() < deadline - 5:
        pts, sc = local_search(best_points, deadline)
        if sc > best_score:
            best_score = sc
            best_points = pts
    
    return best_points
# EVOLVE-BLOCK-END
