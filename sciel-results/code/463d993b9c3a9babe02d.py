"""Improved program: local search to maximize max multiplicity of doubled triangle area."""
import random
import time
import numpy as np
from collections import Counter

def _score(points):
    """Compute the largest multiplicity of a positive doubled area among all triples."""
    n = len(points)
    areas = []
    for i in range(n):
        xi, yi = points[i]
        for j in range(i+1, n):
            xj, yj = points[j]
            dx_ij = xj - xi
            dy_ij = yj - yi
            for k in range(j+1, n):
                xk, yk = points[k]
                det = dx_ij * (yk - yi) - dy_ij * (xk - xi)
                if det != 0:
                    areas.append(abs(det))
    if not areas:
        return 0
    c = Counter(areas)
    return max(c.values())

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    
    # Start with a reasonable initial configuration
    # Use a grid-like structure that might produce repeated areas
    def _initial_points():
        # 6x5 grid
        pts = [(x, y) for x in range(6) for y in range(5)]
        return pts
    
    points = _initial_points()
    best_score = _score(points)
    best_points = points[:]
    
    # Local search: randomly perturb one point, keep if improves
    rng = random.Random(42)
    
    def _random_point():
        return (rng.randint(-30, 30), rng.randint(-30, 30))
    
    def _valid(points):
        """Check all points are distinct and in range."""
        seen = set()
        for p in points:
            if p in seen:
                return False
            seen.add(p)
            if not (-30 <= p[0] <= 30 and -30 <= p[1] <= 30):
                return False
        return True
    
    # Simulated annealing style local search
    current_score = best_score
    current_points = best_points[:]
    temp = 100.0
    cooling = 0.999
    
    while time.time() < deadline:
        # Pick a random point to move
        idx = rng.randint(0, 29)
        old_pt = current_points[idx]
        new_pt = _random_point()
        
        # Try the move
        current_points[idx] = new_pt
        
        if not _valid(current_points):
            current_points[idx] = old_pt
            continue
        
        new_score = _score(current_points)
        
        if new_score >= current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = current_points[:]
        else:
            # Accept with some probability (annealing)
            if rng.random() < temp:
                # Keep the move
                pass
            else:
                current_points[idx] = old_pt
        
        temp *= cooling
        if temp < 1e-4:
            # Reset temperature occasionally
            temp = 50.0
            current_points = best_points[:]
            current_score = best_score
    
    return best_points
