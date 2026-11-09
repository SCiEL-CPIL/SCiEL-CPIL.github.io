import numpy as np
import time
import random
from math import gcd
from itertools import combinations

def normalize_line(A, B, C):
    """Normalize line coefficients to ensure uniqueness and integer validity."""
    if A == 0 and B == 0:
        return None
    
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g == 0:
        g = 1
    A //= g
    B //= g
    C //= g
    
    # Make first non-zero entry positive
    if A < 0 or (A == 0 and B < 0):
        A = -A
        B = -B
        C = -C
        
    return (A, B, C)

def count_incidences(points, lines):
    """
    points: (n, 2) array
    lines: (m, 3) array (A, B, C)
    Returns number of (p, l) such that A*x + B*y + C == 0
    """
    if len(points) == 0 or len(lines) == 0:
        return 0
    P = points.astype(np.int64)
    L = lines.astype(np.int64)
    
    px = P[:, 0].reshape(-1, 1)
    py = P[:, 1].reshape(-1, 1)
    
    A = L[:, 0].reshape(1, -1)
    B = L[:, 1].reshape(1, -1)
    C = L[:, 2].reshape(1, -1)
    
    vals = A * px + B * py + C
    count = np.sum(vals == 0)
    return int(count)

def get_best_points_for_lines(lines, box_range=20, num_points=54):
    """
    Given a set of lines, find the integer points within [-box_range, box_range]^2
    that have the highest number of line incidences.
    Returns top 'num_points' distinct points.
    """
    xs = np.arange(-box_range, box_range + 1)
    ys = np.arange(-box_range, box_range + 1)
    px, py = np.meshgrid(xs, ys, indexing='ij')
    candidate_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    L = lines.astype(np.int64)
    px_c = candidate_points[:, 0].reshape(-1, 1)
    py_c = candidate_points[:, 1].reshape(-1, 1)
    
    A = L[:, 0].reshape(1, -1)
    B = L[:, 1].reshape(1, -1)
    C = L[:, 2].reshape(1, -1)
    
    vals = A * px_c + B * py_c + C
    counts = np.sum(vals == 0, axis=1)
    
    top_indices = np.argsort(counts)[::-1][:num_points]
    best_points = candidate_points[top_indices]
    
    return best_points

def generate_line_from_points(p1, p2):
    """Generate normalized line coefficients from two points."""
    x1, y1 = p1
    x2, y2 = p2
    
    # Line through (x1,y1) and (x2,y2): (y2-y1)x - (x2-x1)y + (x2-x1)y1 - (y2-y1)x2 = 0
    A = y2 - y1
    B = -(x2 - x1)
    C = (x2 - x1) * y1 - (y2 - y1) * x2
    
    return normalize_line(A, B, C)

def generate_candidate_lines(pool_size=2000, box_range=15):
    """
    Generate a large pool of candidate lines by taking lines through pairs of random integer points.
    """
    lines = set()
    xs = np.arange(-box_range, box_range + 1)
    ys = np.arange(-box_range, box_range + 1)
    
    # Sample pairs of points
    for _ in range(pool_size):
        p1 = (random.choice(xs), random.choice(ys))
        p2 = (random.choice(xs), random.choice(ys))
        if p1 == p2:
            continue
        line = generate_line_from_points(p1, p2)
        if line:
            lines.add(line)
            
    return list(lines)

def evaluate_solution(points, lines):
    """
    Evaluate a solution (points, lines).
    Returns the incidence count I.
    """
    return count_incidences(points, lines)

def simulated_annealing(lines_pool, time_limit, initial_points=None, initial_lines=None):
    """
    Simulated annealing to jointly optimize points and lines.
    """
    start_time = time.time()
    deadline = start_time + time_limit
    
    # Initialize
    if initial_points is None:
        # Start with a random set of points
        box_range = 15
        xs = np.arange(-box_range, box_range + 1)
        ys = np.arange(-box_range, box_range + 1)
        initial_points = np.array([
            (random.choice(xs), random.choice(ys)) for _ in range(54)
        ], dtype=np.int64)
        # Ensure uniqueness
        unique_pts, indices = np.unique(initial_points, axis=0, return_index=True)
        if len(unique_pts) < 54:
            # Fill gaps
            while len(unique_pts) < 54:
                p = (random.choice(xs), random.choice(ys))
                if p not in unique_pts:
                    unique_pts = np.vstack([unique_pts, p])
        initial_points = unique_pts[:54]
        
    if initial_lines is None:
        # Start with random lines from pool
        initial_lines = np.array([lines_pool[i] for i in random.sample(range(len(lines_pool)), 27)], dtype=np.int64)
        
    current_points = initial_points.copy()
    current_lines = initial_lines.copy()
    current_I = count_incidences(current_points, current_lines)
    
    best_points = current_points.copy()
    best_lines = current_lines.copy()
    best_I = current_I
    
    T = 50.0
    T_min = 0.1
    alpha = 0.995
    
    while time.time() < deadline:
        # Generate a neighbor
        move_type = random.random()
        
        if move_type < 0.5:
            # Move a point
            idx = random.randint(0, 53)
            old_point = current_points[idx].copy()
            # New random point
            box_range = 20
            new_point = np.array([random.randint(-box_range, box_range), random.randint(-box_range, box_range)], dtype=np.int64)
            
            # Check uniqueness
            if np.any(np.all(current_points == new_point, axis=1)):
                continue
                
            current_points[idx] = new_point
            new_I = count_incidences(current_points, current_lines)
            
            delta = new_I - current_I
            if delta > 0 or random.random() < np.exp(delta / T):
                current_I = new_I
                if current_I > best_I:
                    best_I = current_I
                    best_points = current_points.copy()
                    best_lines = current_lines.copy()
            else:
                current_points[idx] = old_point
        else:
            # Move a line
            idx = random.randint(0, 26)
            old_line = current_lines[idx].copy()
            # New random line from pool
            new_line = np.array(lines_pool[random.randint(0, len(lines_pool) - 1)], dtype=np.int64)
            
            # Check uniqueness
            if np.any(np.all(current_lines == new_line, axis=1)):
                continue
                
            current_lines[idx] = new_line
            new_I = count_incidences(current_points, current_lines)
            
            delta = new_I - current_I
            if delta > 0 or random.random() < np.exp(delta / T):
                current_I = new_I
                if current_I > best_I:
                    best_I = current_I
                    best_points = current_points.copy()
                    best_lines = current_lines.copy()
            else:
                current_lines[idx] = old_line
                
        T *= alpha
        if T < T_min:
            T = T_min
            # Reheat occasionally? Or just stop? Let's keep going with T_min.
            
    return best_points, best_lines, best_I

def szemeredi_trotter_incidences():
    deadline = time.time() + 625 # 5s safety margin
    
    best_points_global = None
    best_lines_global = None
    best_I_global = -1
    
    # Strategy 1: Standard Grid (Quick Baseline)
    k = 3
    xs = np.arange(1, k + 1)
    ys = np.arange(1, 2 * k * k + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points_grid = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    lines_grid = []
    for a in range(1, k + 1):
        for b in range(1, k * k + 1):
            lines_grid.append((a, -1, b))
    lines_grid = np.array(lines_grid, dtype=np.int64)
    
    I_grid = count_incidences(points_grid, lines_grid)
    best_I_global = max(best_I_global, I_grid)
    best_points_global = points_grid
    best_lines_global = lines_grid
    
    # Strategy 2: Generate Candidate Pool
    if time.time() < deadline - 50:
        lines_pool = generate_candidate_lines(pool_size=3000, box_range=15)
        
        # Run Simulated Annealing
        if time.time() < deadline - 10:
            points_sa, lines_sa, I_sa = simulated_annealing(
                lines_pool, 
                time_limit=min(300, deadline - time.time() - 10),
                initial_points=points_grid,
                initial_lines=lines_grid
            )
            
            if I_sa > best_I_global:
                best_I_global = I_sa
                best_points_global = points_sa
                best_lines_global = lines_sa
                
    # Final Safety Check: Ensure validity
    # Points must be unique
    if best_points_global is not None:
        unique_points, indices = np.unique(best_points_global, axis=0, return_index=True)
        if len(unique_points) != 54:
            # Fallback to grid
            best_points_global = points_grid
            best_lines_global = lines_grid
        else:
            best_points_global = unique_points[indices]
            
    # Lines must be unique and valid
    if best_lines_global is not None:
        unique_lines = []
        for line in best_lines_global:
            n_line = normalize_line(line[0], line[1], line[2])
            if n_line and n_line not in unique_lines:
                unique_lines.append(n_line)
                
        if len(unique_lines) < 27:
            best_lines_global = lines_grid
            best_points_global = points_grid
        else:
            best_lines_global = np.array(unique_lines[:27], dtype=np.int64)
            
    return best_points_global, best_lines_global

# EVOLVE-BLOCK-END
