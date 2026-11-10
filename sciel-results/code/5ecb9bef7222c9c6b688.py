"""Combined approach: Structured initialization + Local Search/Hill Climbing."""

import time
import random
import math
from collections import Counter
import numpy as np

def _doubled_area_det(p1, p2, p3):
    return abs(
        (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p1[1] - p3[1]) * (p2[0] - p3[0])
    )

def _max_multiplicity(points):
    """Compute the maximum multiplicity of any positive doubled area among all triples."""
    if len(points) < 3:
        return 0
    pts = np.array(points, dtype=np.int64)
    n = len(pts)
    
    # Vectorized computation of all triangle areas
    # We need areas for triples (i, j, k) with i < j < k
    # Area = 0.5 * | (x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i) |
    # Doubled Area = | (x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i) |
    
    # To avoid huge memory usage, we can process in blocks or use a more efficient loop structure
    # For N=30, total triples is ~4060. This is small.
    # We can create arrays of all pairs and compute cross products with the third point.
    
    # A simpler vectorized way for small N:
    # Generate indices
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    # Mask for i < j < k
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    
    # Extract coordinates
    xi = pts[i_idx[mask], 0]
    yi = pts[i_idx[mask], 1]
    xj = pts[j_idx[mask], 0]
    yj = pts[j_idx[mask], 1]
    xk = pts[k_idx[mask], 0]
    yk = pts[k_idx[mask], 1]
    
    # Compute doubled areas
    areas = np.abs((xj - xi) * (yk - yi) - (yj - yi) * (xk - xi))
    
    # Filter zero areas
    areas = areas[areas > 0]
    
    if len(areas) == 0:
        return 0
        
    # Count multiplicities
    vals, counts = np.unique(areas, return_counts=True)
    return int(counts.max())

def _generate_initial_candidates():
    """Generate diverse initial point sets."""
    candidates = []
    
    # 1. Best Rectangular Grids
    grid_dims = [(5, 6), (6, 5), (4, 8), (8, 4), (3, 10), (10, 3), (2, 15), (15, 2)]
    for w, h in grid_dims:
        pts = [(i, j) for i in range(w) for j in range(h)]
        if len(pts) == 30:
            cx = (w - 1) // 2
            cy = (h - 1) // 2
            centered = [(x - cx, y - cy) for x, y in pts]
            candidates.append(centered)

    # 2. Diamond Shape (Manhattan distance)
    diamond_pts = []
    for x in range(-10, 11):
        for y in range(-10, 11):
            diamond_pts.append((x, y, abs(x) + abs(y)))
    diamond_pts.sort(key=lambda p: p[2])
    seen = set()
    cand_diamond = []
    for x, y, d in diamond_pts:
        if (x, y) not in seen:
            seen.add((x, y))
            cand_diamond.append((x, y))
            if len(cand_diamond) == 30:
                break
    candidates.append(cand_diamond)

    # 3. Circle Shape (Euclidean distance)
    circle_pts = []
    for x in range(-10, 11):
        for y in range(-10, 11):
            circle_pts.append((x, y, x*x + y*y))
    circle_pts.sort(key=lambda p: p[2])
    seen = set()
    cand_circle = []
    for x, y, d in circle_pts:
        if (x, y) not in seen:
            seen.add((x, y))
            cand_circle.append((x, y))
            if len(cand_circle) == 30:
                break
    candidates.append(cand_circle)
    
    # 4. Lattice patterns with different bases
    # Hexagonal-like lattice
    hex_pts = []
    for row in range(6):
        for col in range(5):
            x = col * 2 + (row % 2)
            y = row
            hex_pts.append((x, y))
    if len(hex_pts) == 30:
        cx = 4
        cy = 2
        candidates.append([(x - cx, y - cy) for x, y in hex_pts])
    
    # Triangular lattice
    tri_pts = []
    for row in range(6):
        for col in range(5):
            x = col
            y = row
            tri_pts.append((x, y))
    if len(tri_pts) == 30:
        cx = 2
        cy = 2
        candidates.append([(x - cx, y - cy) for x, y in tri_pts])
    
    # 5. Diagonal line patterns
    for offset in range(-5, 6, 2):
        diag_pts = []
        for i in range(30):
            x = i - 15
            y = i + offset
            if -30 <= x <= 30 and -30 <= y <= 30:
                diag_pts.append((x, y))
        if len(diag_pts) == 30:
            candidates.append(diag_pts)
    
    # 6. Parabolic patterns
    for a in [1, 2, -1, -2]:
        parab_pts = []
        for i in range(-15, 15):
            x = i
            y = a * i * i // 10  # Scale to stay in bounds
            if -30 <= x <= 30 and -30 <= y <= 30:
                parab_pts.append((x, y))
        if len(parab_pts) == 30:
            candidates.append(parab_pts)
    
    # 7. Random Clusters
    for _ in range(10):
        pts = set()
        while len(pts) < 30:
            x = random.randint(-5, 5)
            y = random.randint(-5, 5)
            pts.add((x, y))
        candidates.append(list(pts))
    
    # 8. More structured random: pick from small grid with some randomness
    for _ in range(10):
        pts = set()
        while len(pts) < 30:
            x = random.randint(-4, 4)
            y = random.randint(-4, 4)
            pts.add((x, y))
        candidates.append(list(pts))
    
    # 9. Random from larger grid
    for _ in range(10):
        pts = set()
        while len(pts) < 30:
            x = random.randint(-8, 8)
            y = random.randint(-8, 8)
            pts.add((x, y))
        candidates.append(list(pts))

    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # 1. Initialize with diverse candidates
    candidates = _generate_initial_candidates()
    
    best_score = -1
    best_points = None
    
    # Evaluate initial candidates
    for pts in candidates:
        if time.time() > deadline:
            break
        score = _max_multiplicity(pts)
        if score > best_score:
            best_score = score
            best_points = pts[:]
            
    if best_points is None:
        # Fallback
        best_points = [(x, y) for x in range(6) for y in range(5)]
        best_score = _max_multiplicity(best_points)
        
    # 2. Local Search / Hill Climbing with occasional restarts
    current_points = best_points[:]
    current_score = best_score
    occupied = set(current_points)
    
    # Parameters
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.99
    
    while time.time() < deadline:
        # Pick a random point to move
        idx = random.randint(0, 29)
        old_p = current_points[idx]
        
        # Generate a single random neighbor
        # Bias towards small moves
        if random.random() < 0.7:
            radius = 1
        else:
            radius = random.choice([2, 3, 5])
        
        dx = random.randint(-radius, radius)
        dy = random.randint(-radius, radius)
        
        if dx == 0 and dy == 0:
            continue
            
        new_p = (old_p[0] + dx, old_p[1] + dy)
        
        if -30 <= new_p[0] <= 30 and -30 <= new_p[1] <= 30 and new_p not in occupied:
            # Perform move
            occupied.remove(old_p)
            occupied.add(new_p)
            current_points[idx] = new_p
            
            new_score = _max_multiplicity(current_points)
            
            delta = new_score - current_score
            
            if delta >= 0:
                # Accept
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_points = current_points[:]
            else:
                # Reject with probability exp(-delta/T)
                prob = math.exp(delta / T)
                if random.random() < prob:
                    current_score = new_score
                else:
                    # Revert
                    occupied.remove(new_p)
                    occupied.add(old_p)
                    current_points[idx] = old_p
            
            # Cool down
            T = max(T_min, T * cooling_rate)
            
            # Restart if stuck or cooled down too much
            if T <= T_min * 1.1 or (current_score < best_score and random.random() < 0.01):
                # Restart from best
                current_points = best_points[:]
                current_score = best_score
                occupied = set(current_points)
                T = 1.0
    
    return best_points
