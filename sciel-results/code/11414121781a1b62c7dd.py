# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random
from itertools import combinations

def _normalize_line(A, B, C):
    A, B, C = int(A), int(B), int(C)
    g = math.gcd(abs(A), abs(B))
    g = math.gcd(g, abs(C))
    if g == 0:
        return (0, 0, 0)
    A, B, C = A // g, B // g, C // g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _count_incidence(points, lines):
    """Count incidences between points (N,2) and lines (M,3)."""
    if len(points) == 0 or len(lines) == 0:
        return 0
    px = points[:, 0]
    py = points[:, 1]
    A = lines[:, 0]
    B = lines[:, 1]
    C = lines[:, 2]
    # (M, N) matrix
    vals = A[:, None] * px[None, :] + B[:, None] * py[None, :] + C[:, None]
    return int(np.sum(vals == 0))

def _find_best_lines_for_points(points, k=27, deadline=None):
    """Find the k lines with maximum total incidence for the given points."""
    n = len(points)
    points_int = points.astype(np.int64)
    
    # Collect all distinct lines from pairs of points
    seen_lines = set()
    candidate_lines = []
    
    for i in range(n):
        x1, y1 = int(points_int[i, 0]), int(points_int[i, 1])
        for j in range(i + 1, n):
            x2, y2 = int(points_int[j, 0]), int(points_int[j, 1])
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            if A == 0 and B == 0:
                continue
            key = _normalize_line(A, B, C)
            if key not in seen_lines:
                seen_lines.add(key)
                candidate_lines.append(key)
    
    # Add small-coefficient lines that might not be determined by point pairs
    # These can capture lines that pass through many points but aren't the
    # "primary" line for any pair
    for A in range(-4, 5):
        for B in range(-4, 5):
            if A == 0 and B == 0:
                continue
            for C in range(-50, 51):
                key = _normalize_line(A, B, C)
                if key not in seen_lines:
                    seen_lines.add(key)
                    candidate_lines.append(key)
    
    arr = np.array(candidate_lines, dtype=np.int64)
    if arr.size == 0:
        # Fallback: 27 horizontal lines
        return np.array([[_normalize_line(0, 1, -c)] for c in range(-27, 0)], dtype=np.int64)
    
    px = points_int[:, 0:1].T  # (1, N)
    py = points_int[:, 1:2].T  # (1, N)
    A = arr[:, 0:1]
    B = arr[:, 1:2]
    C = arr[:, 2:3]
    
    # Compute incidence counts for all candidates
    # (M, N) comparison
    vals = A * px + B * py + C
    counts = np.sum(vals == 0, axis=1)
    
    # Sort by count descending
    if len(counts) > k:
        idx = np.argsort(-counts)
        top_idx = idx[:k]
    else:
        top_idx = np.arange(len(counts))
    
    return arr[top_idx].astype(np.int64)

def _generate_point_sets(deadline):
    """Generate a variety of point configurations."""
    sets = []
    
    # 1. Rectangular Grid 6x9
    # This is a strong baseline.
    for ox in range(0, 3):
        for oy in range(0, 3):
            pts = np.array([[ox + i, oy + j] for i in range(6) for j in range(9)], dtype=np.int64)
            sets.append(pts)
            
    # 2. Triangular Lattice (staggered rows)
    for stagger_type in ['even', 'odd', 'both']:
        pts = []
        for j in range(9):
            for i in range(6):
                if stagger_type == 'even' and j % 2 == 1:
                    x = i + 1
                elif stagger_type == 'odd' and j % 2 == 0:
                    x = i + 1
                else:
                    x = i
                pts.append([x, j])
        pts = np.array(pts, dtype=np.int64)
        # Check uniqueness
        if len(set(map(tuple, pts))) == 54:
            sets.append(pts)

    # 3. Clique-star hybrid: 27 points on a vertical line + 27 on a 3x9 grid
    # This exploits the high local density of the grid
    # Vertical line at x=0, y=0..26
    # Grid at x=2..4, y=0..8 (3 columns, 9 rows)
    vpts = [[0, y] for y in range(27)]
    gpts = [[x, y] for x in range(2, 5) for y in range(9)]
    pts = np.array(vpts + gpts, dtype=np.int64)
    if len(set(map(tuple, pts))) == 54:
        sets.append(pts)
    
    # 4. Another clique-star variant: vertical line + 6x4.5 grid (not integer, so use 3x9)
    # Try different grid shapes
    for gw, gh in [(3, 9), (9, 3), (2, 13.5)]:  # 2x13.5 not integer
        if gw * gh != 27:
            continue
        if not isinstance(gh, int) or gh < 1:
            continue
        vpts = [[0, y] for y in range(27)]
        gpts = [[x + 2, y] for x in range(gw) for y in range(gh)]
        pts = np.array(vpts + gpts, dtype=np.int64)
        if len(set(map(tuple, pts))) == 54:
            sets.append(pts)
    
    # 5. Sheared grid: x = i + j//3, y = j
    for shear in [0, 1, 2]:
        pts = []
        for j in range(9):
            for i in range(6):
                x = i + shear * (j // 3)
                pts.append([x, j])
        pts = np.array(pts, dtype=np.int64)
        if len(set(map(tuple, pts))) == 54:
            sets.append(pts)
    
    # 6. Scaled grid: 6x9 with spacing
    for sx, sy in [(1, 1), (2, 1), (1, 2), (2, 2), (3, 1), (1, 3)]:
        pts = np.array([[i * sx, j * sy] for i in range(6) for j in range(9)], dtype=np.int64)
        if len(set(map(tuple, pts))) == 54:
            sets.append(pts)
    
    # 7. Random perturbations of the 6x9 grid
    for _ in range(5):
        base = np.array([[i, j] for i in range(6) for j in range(9)], dtype=np.int64)
        pts = base.copy()
        # Move a few points randomly
        for _ in range(3):
            idx = random.randint(0, 53)
            pts[idx, 0] += random.randint(-2, 2)
            pts[idx, 1] += random.randint(-2, 2)
        # Check uniqueness
        if len(set(map(tuple, pts))) == 54:
            sets.append(pts)
    
    return sets

def _evaluate_set(points, deadline):
    """Evaluate a specific point set by finding best lines."""
    lines = _find_best_lines_for_points(points, 27, deadline)
    score = _count_incidence(points, lines)
    return points, lines, score

def _refine_set(points, lines, deadline, max_steps=2000):
    """
    Refine the configuration by moving points or changing lines.
    Strategy:
    1. Try moving one point to a nearby integer location.
    2. Recompute best lines for the new points.
    3. Accept if score improves.
    """
    current_score = _count_incidence(points, lines)
    best_points = points.copy()
    best_lines = lines.copy()
    best_score = current_score
    
    no_improve_count = 0
    
    for _ in range(max_steps):
        if time.time() > deadline:
            break
        
        # Pick a random point to move
        i = random.randint(0, 53)
        ox, oy = int(best_points[i, 0]), int(best_points[i, 1])
        
        # Try moving to a nearby location
        moved = False
        for dx in [-1, 0, 1, -2, 2]:
            for dy in [-1, 0, 1, -2, 2]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = ox + dx, oy + dy
                
                # Check if this position is already taken by another point
                conflict = False
                for j in range(54):
                    if j != i and int(best_points[j, 0]) == nx and int(best_points[j, 1]) == ny:
                        conflict = True
                        break
                if conflict:
                    continue
                
                # Create new points
                new_points = best_points.copy()
                new_points[i, 0] = nx
                new_points[i, 1] = ny
                
                # Find best lines for new points
                new_lines = _find_best_lines_for_points(new_points, 27)
                new_score = _count_incidence(new_points, new_lines)
                
                if new_score > current_score:
                    best_points = new_points
                    best_lines = new_lines
                    current_score = new_score
                    best_score = current_score
                    no_improve_count = 0
                    moved = True
                    break
            
            if moved:
                break
        
        if not moved:
            no_improve_count += 1
            # If we stagnate for a while, try a bigger jump
            if no_improve_count > 50:
                i = random.randint(0, 53)
                ox, oy = int(best_points[i, 0]), int(best_points[i, 1])
                nx = ox + random.randint(-5, 5)
                ny = oy + random.randint(-5, 5)
                
                conflict = False
                for j in range(54):
                    if j != i and int(best_points[j, 0]) == nx and int(best_points[j, 1]) == ny:
                        conflict = True
                        break
                if not conflict:
                    new_points = best_points.copy()
                    new_points[i, 0] = nx
                    new_points[i, 1] = ny
                    new_lines = _find_best_lines_for_points(new_points, 27)
                    new_score = _count_incidence(new_points, new_lines)
                    
                    if new_score > current_score:
                        best_points = new_points
                        best_lines = new_lines
                        current_score = new_score
                        best_score = current_score
                        no_improve_count = 0
                    elif new_score == current_score:
                        # Accept equal score moves to escape local optima
                        if random.random() < 0.1:
                            best_points = new_points
                            best_lines = new_lines
                            no_improve_count = 0
    
    return best_points, best_lines, best_score

def _polish_set(points, lines, deadline, max_steps=500):
    """
    Final high-precision polish pass with tighter tolerance and more iterations.
    Uses smaller step sizes and accepts only strict improvements.
    """
    current_score = _count_incidence(points, lines)
    best_points = points.copy()
    best_lines = lines.copy()
    best_score = current_score
    
    for _ in range(max_steps):
        if time.time() > deadline:
            break
        
        # Pick a random point to move
        i = random.randint(0, 53)
        ox, oy = int(best_points[i, 0]), int(best_points[i, 1])
        
        # Try moving to a nearby location with smaller steps
        moved = False
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = ox + dx, oy + dy
                
                # Check if this position is already taken by another point
                conflict = False
                for j in range(54):
                    if j != i and int(best_points[j, 0]) == nx and int(best_points[j, 1]) == ny:
                        conflict = True
                        break
                if conflict:
                    continue
                
                # Create new points
                new_points = best_points.copy()
                new_points[i, 0] = nx
                new_points[i, 1] = ny
                
                # Find best lines for new points
                new_lines = _find_best_lines_for_points(new_points, 27)
                new_score = _count_incidence(new_points, new_lines)
                
                # Only accept strict improvements
                if new_score > current_score:
                    best_points = new_points
                    best_lines = new_lines
                    current_score = new_score
                    best_score = current_score
                    moved = True
                    break
            
            if moved:
                break
    
    return best_points, best_lines, best_score

def szemeredi_trotter_incidences():
    start_time = time.time()
    # Safety margin: return before 620s
    deadline = start_time + 600
    
    best_points = None
    best_lines = None
    best_score = -1
    
    # Phase 1: Evaluate initial candidate sets
    candidate_sets = _generate_point_sets(deadline)
    
    for pts in candidate_sets:
        if time.time() > deadline:
            break
        # Ensure points are unique and integer
        pts, lines, score = _evaluate_set(pts, deadline)
        if score > best_score:
            best_score = score
            best_points = pts.copy()
            best_lines = lines.copy()
    
    # Phase 2: Refinement
    if best_points is not None:
        # Use remaining time for refinement
        elapsed = time.time() - start_time
        remaining = deadline - start_time - elapsed
        
        # Set a budget for refinement steps
        max_steps = int(remaining * 80)  # conservative estimate
        if max_steps < 100:
            max_steps = 100
        if max_steps > 4000:
            max_steps = 4000
            
        best_points, best_lines, best_score = _refine_set(
            best_points, best_lines, deadline, max_steps
        )
        
        # Phase 3: Polish pass
        if time.time() < deadline - 30:
            best_points, best_lines, best_score = _polish_set(
                best_points, best_lines, deadline, max_steps=300
            )
    
    # Final Validation and Cleanup
    if best_points is None or len(best_points) != 54:
        # Emergency fallback: 6x9 grid
        best_points = np.array([[i, j] for i in range(6) for j in range(9)], dtype=np.int64)
        best_lines = _find_best_lines_for_points(best_points, 27)
    
    # Ensure shapes and types
    points = np.array(best_points, dtype=np.int64)
    lines = np.array(best_lines, dtype=np.int64)
    
    # Deduplicate points just in case
    if len(points) > 54:
        points = points[:54]
    elif len(points) < 54:
        while len(points) < 54:
            points = np.vstack([points, [len(points) * 100, len(points) * 200]])
    
    # Deduplicate lines
    valid_lines = []
    seen = set()
    for l in lines:
        key = _normalize_line(int(l[0]), int(l[1]), int(l[2]))
        if key not in seen and (key[0] != 0 or key[1] != 0):
            seen.add(key)
            valid_lines.append(key)
    
    lines = np.array(valid_lines, dtype=np.int64)
    
    # Ensure exactly 27 lines
    if len(lines) > 27:
        lines = lines[:27]
    elif len(lines) < 27:
        c = 1000
        while len(lines) < 27:
            key = _normalize_line(0, 1, -c)
            if key not in seen:
                seen.add(key)
                lines = np.vstack([lines, np.array(key, dtype=np.int64)])
            c += 1
    
    return points, lines

# EVOLVE-BLOCK-END
