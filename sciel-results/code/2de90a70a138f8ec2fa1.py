"""Program to maximize the largest multiplicity of a single positive doubled area among 30 points."""
# EVOLVE-BLOCK-START
import time
import itertools
from collections import Counter
import numpy as np

def _doubled_areas(points):
    """Return list of absolute doubled areas for all triples (excluding zero)."""
    n = len(points)
    areas = []
    for i in range(n):
        xi, yi = points[i]
        for j in range(i+1, n):
            xj, yj = points[j]
            for k in range(j+1, n):
                # 2 * area = |cross product| = |xj-xi)(yk-yi) - (yj-yi)(xk-xi)|
                det = abs((xj - xi) * (yk - yi) - (yj - yi) * (xk - xi))
                if det != 0:
                    areas.append(det)
    return areas

def _score(points):
    """Largest multiplicity of a single positive doubled area."""
    areas = _doubled_areas(points)
    if not areas:
        return 0
    counts = Counter(areas)
    return max(counts.values())

def construct():
    """
    Use a structured approach: start with a grid-like configuration, then use 
    local search to maximize the largest multiplicity of a single doubled area.
    """
    deadline = time.time() + 85  # Leave 5s margin
    
    # Initial configuration: 6x5 grid
    points = [(x, y) for x in range(6) for y in range(5)]
    best_score = _score(points)
    best_points = points[:]
    
    # Local search: try to move points to improve the score
    # We'll do a simple hill climbing with random restarts
    
    import random
    
    def _neighborhood(points):
        """Generate neighboring configurations by moving one point by a small amount."""
        neighbors = []
        n = len(points)
        for i in range(n):
            x, y = points[i]
            for dx in [-2, -1, 1, 2]:
                for dy in [-2, -1, 1, 2]:
                    nx, ny = x + dx, y + dy
                    if -30 <= nx <= 30 and -30 <= ny <= 30:
                        # Check if point already exists
                        if (nx, ny) not in points:
                            new_points = points[:]
                            new_points[i] = (nx, ny)
                            neighbors.append(new_points)
        return neighbors
    
    current_points = best_points[:]
    current_score = best_score
    
    while time.time() < deadline:
        # Generate neighbors
        neighbors = _neighborhood(current_points)
        if not neighbors:
            break
        
        best_neighbor = None
        best_neighbor_score = current_score
        
        for nb in neighbors:
            if time.time() > deadline:
                break
            s = _score(nb)
            if s > best_neighbor_score:
                best_neighbor_score = s
                best_neighbor = nb
        
        if best_neighbor is None or best_neighbor_score <= current_score:
            # Random restart
            if time.time() > deadline - 10:
                break
            # Create a new random configuration
            new_points = []
            used = set()
            for _ in range(30):
                x = random.randint(-30, 30)
                y = random.randint(-30, 30)
                if (x, y) not in used:
                    new_points.append((x, y))
                    used.add((x, y))
            if len(new_points) == 30:
                current_points = new_points
                current_score = _score(current_points)
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points[:]
        else:
            current_points = best_neighbor
            current_score = best_neighbor_score
            if current_score > best_score:
                best_score = current_score
                best_points = current_points[:]
    
    return best_points
# EVOLVE-BLOCK-END
