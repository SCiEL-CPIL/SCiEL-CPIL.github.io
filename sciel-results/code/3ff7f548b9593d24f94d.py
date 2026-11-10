"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
from math import gcd

def _line_key(p1, p2):
    """Return a canonical key for the line through two integer points."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    g = gcd(abs(dx), abs(dy))
    if g > 0:
        dx //= g
        dy //= g
    # Canonical direction: make first non-zero component positive
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    # Line is determined by direction and a point on it; use the point with smallest x (then y) that is on the line
    # Actually, use: line = (dx, dy, c) where c = x1*dy - y1*dx (normalized)
    # But we need a canonical representation. Use the direction (dx,dy) and the intercept.
    # For a line with direction (dx,dy), the normal form: dy*x - dx*y = c
    c = dy * x1 - dx * y1
    # Normalize c with respect to direction: since direction is primitive, c is already in simplest form
    return (dx, dy, c)

def _count_lines(points):
    """Count lines with exactly 4 points, and check for any line with 5+ points."""
    from collections import defaultdict
    line_map = defaultdict(set)
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            key = _line_key(points[i], points[j])
            if key is not None:
                line_map[key].add(i)
    
    four_lines = 0
    five_plus = 0
    for key, idxs in line_map.items():
        cnt = len(idxs)
        if cnt == 4:
            four_lines += 1
        elif cnt >= 5:
            five_plus += 1
    return four_lines, five_plus

def _score(points):
    """Score: number of 4-point lines minus a large penalty for 5+ collinear."""
    four, five = _count_lines(points)
    if five > 0:
        return four - 1000 * five
    return four

def _generate_candidate(lattice_size, num_points=16):
    """Generate a random subset of num_points from a lattice_size x lattice_size grid."""
    candidates = [(x, y) for x in range(lattice_size) for y in range(lattice_size)]
    chosen = random.sample(candidates, num_points)
    return chosen

def _neighbor(points, lattice_size):
    """Generate a neighbor by moving one point to a random empty lattice position."""
    n = len(points)
    occupied = set(points)
    # Pick a random point to move
    idx = random.randint(0, n-1)
    old_pt = points[idx]
    # Pick a random new position
    while True:
        new_pt = (random.randint(0, lattice_size-1), random.randint(0, lattice_size-1))
        if new_pt not in occupied:
            break
    new_points = list(points)
    new_points[idx] = new_pt
    return new_points, idx, old_pt, new_pt

def _anneal(points, lattice_size, max_iters=50000, initial_temp=10.0, final_temp=0.1):
    """Simulated annealing to maximize score."""
    current = list(points)
    current_score = _score(current)
    best = list(current)
    best_score = current_score
    
    n_iters = max_iters
    for i in range(n_iters):
        temp = initial_temp * (final_temp / initial_temp) ** (i / n_iters)
        neighbor_pts, idx, old_pt, new_pt = _neighbor(current, lattice_size)
        neighbor_score = _score(neighbor_pts)
        delta = neighbor_score - current_score
        if delta >= 0 or random.random() < (2.718281828 ** (delta / max(temp, 1e-10))):
            current = neighbor_pts
            current_score = neighbor_score
            if current_score > best_score:
                best = list(current)
                best_score = current_score
    return best, best_score

def construct():
    deadline = time.time() + 140.0  # safety margin
    best_points = None
    best_score = -1
    
    # Try different lattice sizes
    for lattice_size in [5, 6, 7]:
        if time.time() > deadline - 30:
            break
        
        # Multiple restarts
        num_restarts = 5
        for restart in range(num_restarts):
            if time.time() > deadline - 15:
                break
            
            # Generate initial candidate
            points = _generate_candidate(lattice_size, 16)
            init_score = _score(points)
            
            # Simulated annealing
            max_iters = 30000
            result, score = _anneal(points, lattice_size, max_iters=max_iters)
            
            if score > best_score:
                best_score = score
                best_points = result
    
    if best_points is not None:
        return [[x, y] for x, y in best_points]
    
    # Fallback: 4x4 grid
    return [[x, y] for x in range(4) for y in range(4)]
# EVOLVE-BLOCK-END
