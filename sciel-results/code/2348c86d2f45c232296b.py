"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import numpy as np
from itertools import combinations
from collections import Counter
import time

def _compute_best_area_multiplicity(points):
    """Compute the largest multiplicity of a single positive doubled area."""
    n = len(points)
    pts = np.array(points, dtype=np.int64)
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                det = abs(int(pts[i][0])*(pts[j][1]-pts[k][1]) + 
                         int(pts[j][0])*(pts[k][1]-pts[i][1]) + 
                         int(pts[k][0])*(pts[i][1]-pts[j][1]))
                if det > 0:
                    areas.append(det)
    if not areas:
        return 0
    counter = Counter(areas)
    return counter.most_common(1)[0][1]

def _make_grid(w, h, offset_x=0, offset_y=0):
    """Create a w x h grid of points."""
    points = []
    for x in range(w):
        for y in range(h):
            points.append((x + offset_x, y + offset_y))
    return points

def _make_hexagonal(rows, cols):
    """Create a hexagonal-like grid."""
    points = []
    for r in range(rows):
        for c in range(cols):
            x = c + (r % 2) * 0  # simple grid for now
            y = r
            points.append((x, y))
    return points

def _get_areas_involving_index(points, idx):
    """
    Efficiently compute the list of doubled areas for all triangles involving points[idx].
    Returns a list of areas.
    """
    n = len(points)
    pts = np.array(points, dtype=np.int64)
    p_i = pts[idx]
    
    # Get all other points
    mask = np.ones(n, dtype=bool)
    mask[idx] = False
    other_pts = pts[mask]
    
    # Vectors from p_i to other points
    # Shape: (n-1, 2)
    vecs = other_pts - p_i
    
    # Calculate cross product (determinant) for all pairs of vectors
    # We need pairs (j, k) where j < k in the original indices, but here we just need all pairs of distinct other points.
    # The triangle is (i, j, k). The area is 0.5 * |det(vec_j, vec_k)|.
    # We can compute all pairwise cross products.
    
    # To avoid double counting and self-pairs, we can use a loop or vectorized tricks.
    # Given n=30, n-1=29. Pairs = 29*28/2 = 406. 
    # Vectorizing 406 determinants is fast.
    
    areas = []
    # Extract x and y components
    vx = vecs[:, 0]
    vy = vecs[:, 1]
    
    # Compute determinant for all pairs (a, b) with a < b
    # We can use broadcasting to create matrices of size (29, 29)
    # det(a, b) = vx[a]*vy[b] - vx[b]*vy[a]
    vx_mat = vx[:, np.newaxis]
    vy_mat = vy[:, np.newaxis]
    
    # D[a, b] = vx[a]*vy[b] - vx[b]*vy[a]
    # This is antisymmetric. We only need upper triangle.
    D = vx_mat * vy_mat.T - vx_mat.T * vy_mat
    
    # Extract upper triangle indices
    rows, cols = np.triu_indices(29, k=1)
    dets = D[rows, cols]
    
    # Absolute values
    abs_dets = np.abs(dets)
    
    # Filter zeros
    non_zero = abs_dets[abs_dets > 0]
    
    return non_zero.tolist()

def _simulated_annealing(initial_points, steps=20000, start_temp=10.0, end_temp=0.01):
    """
    Simulated Annealing to maximize the largest multiplicity of a single positive doubled area.
    """
    current_points = [p[:] for p in initial_points]
    n = len(current_points)
    
    # Initialize area counts
    area_counts = Counter()
    # Add all areas
    for i in range(n):
        areas = _get_areas_involving_index(current_points, i)
        # Note: This counts each triangle 3 times (once for each vertex).
        # We need to divide counts by 3 at the end, or just track relative counts.
        # Since we are comparing multiplicities, scaling by a constant factor (1/3) doesn't change the argmax.
        # However, the "multiplicity" in the task is the count of triangles with that area.
        # If we sum counts from each vertex, a triangle with area A contributes 1 to area_counts[A] for each of its 3 vertices.
        # So the final count for area A will be 3 * (actual number of triangles with area A).
        # The ratio of multiplicities remains the same. The absolute value is 3x.
        # The task asks for "largest multiplicity". If the evaluator expects the raw count of triangles, 
        # we should divide by 3. Let's check the seed.
        # Seed: "Compute the absolute determinant for every triple... maximize the largest multiplicity".
        # Seed implementation: loops over triples (i,j,k), adds det to list. Counter(list).
        # So seed counts each triangle exactly once.
        # My method counts each triangle 3 times.
        # So I must divide the final score by 3, or keep track of unique triangles.
        # Easiest: keep the counter as is, but when reporting/comparing score, use count // 3?
        # No, counts might not be divisible by 3 if there's a bug, but theoretically they are.
        # Let's just use the raw counts for comparison during SA (since 3*max is max of 3*counts),
        # and return the correct value at the end.
        area_counts.update(areas)
        
    def get_score(counts):
        if not counts:
            return 0
        # Max count in counter. Since each triangle is counted 3 times, 
        # the actual multiplicity is count // 3.
        # However, for SA comparison, maximizing count is equivalent to maximizing count//3.
        return counts.most_common(1)[0][1]
    
    current_score = get_score(area_counts)
    best_score = current_score
    best_points = [p[:] for p in current_points]
    
    # Temperature schedule
    temp = start_temp
    cooling_rate = (end_temp / start_temp) ** (1.0 / steps)
    
    for step in range(steps):
        # Pick a random point to move
        i = random.randint(0, n - 1)
        old_x, old_y = current_points[i]
        
        # Generate new position
        # Random walk with step size 1 or 2
        dx = random.randint(-2, 2)
        dy = random.randint(-2, 2)
        if dx == 0 and dy == 0:
            continue
            
        new_x, new_y = old_x + dx, old_y + dy
        
        if not (-30 <= new_x <= 30 and -30 <= new_y <= 30):
            continue
            
        new_point = (new_x, new_y)
        
        # Check distinctness
        is_distinct = True
        for j, p in enumerate(current_points):
            if j != i and p == new_point:
                is_distinct = False
                break
                
        if not is_distinct:
            continue
            
        # Calculate delta
        # 1. Remove old areas involving i
        old_areas = _get_areas_involving_index(current_points, i)
        for a in old_areas:
            area_counts[a] -= 1
            if area_counts[a] == 0:
                del area_counts[a]
                
        # 2. Move point
        current_points[i] = new_point
        
        # 3. Add new areas involving i
        new_areas = _get_areas_involving_index(current_points, i)
        for a in new_areas:
            area_counts[a] += 1
            
        new_score = get_score(area_counts)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta > 0 or random.random() < np.exp(delta / temp):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in current_points]
        else:
            # Revert
            current_points[i] = (old_x, old_y)
            # Revert counts
            for a in new_areas:
                area_counts[a] -= 1
                if area_counts[a] == 0:
                    del area_counts[a]
            for a in old_areas:
                area_counts[a] += 1
                
        temp *= cooling_rate
        
    # The best_score is 3x the actual multiplicity.
    # We need to return the points, and the construct function will compute the score?
    # No, construct returns points. The evaluator computes the score.
    # But _try_configurations compares scores.
    # We should return the points that yielded best_score.
    # The actual multiplicity is best_score // 3.
    
    return best_points, best_score // 3

def _try_configurations():
    """Try various configurations and return the best one."""
    initial_configs = []
    
    # 6x5 Grid centered
    grid_6_5 = _make_grid(6, 5, -2, -2)
    initial_configs.append(grid_6_5)
    
    # 5x6 Grid centered
    grid_5_6 = _make_grid(5, 6, -2, -2)
    initial_configs.append(grid_5_6)
    
    # Random initial config
    random.seed(42)
    random_points = set()
    while len(random_points) < 30:
        x = random.randint(-10, 10)
        y = random.randint(-10, 10)
        random_points.add((x, y))
    initial_configs.append(list(random_points))
    
    best_pts = None
    best_mult = -1
    
    # Use Simulated Annealing with more steps
    for config in initial_configs:
        pts, score = _simulated_annealing(config, steps=10000)
        if score > best_mult:
            best_mult = score
            best_pts = pts
            
    if best_pts is None:
        best_pts = _make_grid(6, 5)
        
    return best_pts

def construct():
    return _try_configurations()
