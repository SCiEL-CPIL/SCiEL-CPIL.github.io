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
    if len(points) == 0 or len(lines) == 0:
        return 0
    px = points[:, 0].astype(np.int64)
    py = points[:, 1].astype(np.int64)
    A = lines[:, 0].astype(np.int64)
    B = lines[:, 1].astype(np.int64)
    C = lines[:, 2].astype(np.int64)
    # (M, 1) * (1, N) + (M, 1) * (1, N) + (M, 1) -> (M, N)
    vals = A[:, None] * px[None, :] + B[:, None] * py[None, :] + C[:, None]
    return int(np.sum(vals == 0))

def _build_candidate_lines(points, max_coeff=10, max_C_range=100):
    """Build a rich set of candidate lines."""
    n = len(points)
    points_int = points.astype(np.int64)
    seen_lines = set()
    candidate_lines = []
    
    # 1. Lines from point pairs (captures structure of the specific point set)
    # To save time, we might subsample pairs if n is large, but 54 is small (54*53/2 = 1431 pairs)
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
    
    # 2. Systematic small-coefficient lines (captures general geometric structures)
    # We want primitive (A, B) pairs
    px = points_int[:, 0]
    py = points_int[:, 1]
    
    for A in range(-max_coeff, max_coeff + 1):
        for B in range(-max_coeff, max_coeff + 1):
            if A == 0 and B == 0:
                continue
            g = math.gcd(abs(A), abs(B))
            if g > 1:
                continue  # Only primitive (A,B) pairs to avoid duplicates via normalization anyway, but this reduces iteration
            
            # Determine C range based on point spread
            c_vals = A * px + B * py
            if len(c_vals) > 0:
                c_min = int(c_vals.min()) - 20
                c_max = int(c_vals.max()) + 20
            else:
                c_min = -max_C_range
                c_max = max_C_range
            
            # Cap the range to avoid explosion
            c_min = max(c_min, -max_C_range)
            c_max = min(c_max, max_C_range)
            
            for C in range(c_min, c_max + 1):
                key = _normalize_line(A, B, C)
                if key not in seen_lines:
                    seen_lines.add(key)
                    candidate_lines.append(key)
    
    return np.array(candidate_lines, dtype=np.int64), seen_lines

def _find_best_lines_for_points(points, k=27, candidate_arr=None):
    """Find the k lines with maximum total incidence for the given points."""
    if candidate_arr is None or len(candidate_arr) == 0:
        candidate_arr, _ = _build_candidate_lines(points)
    
    if len(candidate_arr) == 0:
        # Fallback: vertical lines
        return np.array([[1, 0, -i] for i in range(1, k+1)], dtype=np.int64)
    
    px = points[:, 0].astype(np.int64)
    py = points[:, 1].astype(np.int64)
    A = candidate_arr[:, 0:1].astype(np.int64)
    B = candidate_arr[:, 1:2].astype(np.int64)
    C = candidate_arr[:, 2:3].astype(np.int64)
    
    # Compute incidence counts vectorized
    # Shape: (M, 1) * (1, N) ... -> (M, N)
    vals = A * px[None, :] + B * py[None, :] + C
    counts = np.sum(vals == 0, axis=1)
    
    if len(counts) > k:
        # Use argpartition for speed, then sort the top k
        idx = np.argpartition(-counts, k)[:k]
        idx = idx[np.argsort(-counts[idx])]
    else:
        idx = np.argsort(-counts)
    
    return candidate_arr[idx].astype(np.int64)

def _generate_point_sets():
    """Generate diverse initial point configurations."""
    sets = []
    
    # Standard grids
    for w, h in [(6, 9), (9, 6), (3, 18), (18, 3), (2, 27), (27, 2)]:
        pts = np.array([[i, j] for i in range(w) for j in range(h)], dtype=np.int64)
        sets.append(pts)
        
    # Scaled grids
    for scale in [2, 3]:
        pts = np.array([[i*scale, j*scale] for i in range(6) for j in range(9)], dtype=np.int64)
        sets.append(pts)
        
    # Sheared grids
    for shear in [1, 2, -1, 3]:
        pts = np.array([[i + shear*j, j] for i in range(6) for j in range(9)], dtype=np.int64)
        if len(set(map(tuple, pts.tolist()))) == 54:
            sets.append(pts)
            
    # Staggered rows
    for stagger in [1, 2]:
        pts = []
        for j in range(9):
            offset = (j % 2) * stagger
            for i in range(6):
                pts.append([i + offset, j])
        pts = np.array(pts, dtype=np.int64)
        if len(set(map(tuple, pts.tolist()))) == 54:
            sets.append(pts)
            
    # Checkerboard pattern on a larger grid
    pts = []
    for i in range(8):
        for j in range(8):
            if (i + j) % 2 == 0:
                pts.append([i, j])
    # We have 32 points, need 54. Let's just use a dense 7x8 grid minus 2?
    # Better: 6x9 grid with some random shifts
    
    # Random perturbations of 6x9 grid
    for _ in range(15):
        base = np.array([[i, j] for i in range(6) for j in range(9)], dtype=np.int64)
        pts = base.copy()
        # Move a few points randomly
        for _ in range(random.randint(5, 15)):
            idx = random.randint(0, 53)
            pts[idx, 0] += random.randint(-4, 4)
            pts[idx, 1] += random.randint(-4, 4)
        if len(set(map(tuple, pts.tolist()))) == 54:
            sets.append(pts)
            
    return sets

def _simulated_annealing(points, candidate_lines, deadline, T_start=100.0, T_end=1.0, cooling_rate=0.9995):
    """
    Optimizes points using simulated annealing.
    candidate_lines is a fixed set of lines to check against.
    """
    n_points = len(points)
    n_lines = len(candidate_lines)
    
    # Pre-compute line coefficients as int64 arrays for speed
    A = candidate_lines[:, 0].astype(np.int64)
    B = candidate_lines[:, 1].astype(np.int64)
    C = candidate_lines[:, 2].astype(np.int64)
    
    current_points = points.copy()
    current_score = _count_incidence(current_points, candidate_lines)
    best_points = current_points.copy()
    best_score = current_score
    
    point_set = set(map(tuple, current_points.tolist()))
    
    T = T_start
    steps = 0
    max_steps = 100000 # Safety limit, usually deadline controls
    
    while time.time() < deadline and steps < max_steps:
        if T < T_end:
            break
            
        # Pick a random point
        i = random.randint(0, n_points - 1)
        ox, oy = int(current_points[i, 0]), int(current_points[i, 1])
        
        # Generate a neighbor
        # Bias towards small moves, but allow larger ones occasionally
        move_size = random.choice([1, 1, 1, 1, 2, 2, 3, 4, 5])
        dx = random.randint(-move_size, move_size)
        dy = random.randint(-move_size, move_size)
        
        if dx == 0 and dy == 0:
            continue
            
        nx, ny = ox + dx, oy + dy
        
        # Check collision
        if (nx, ny) in point_set:
            continue
            
        # Apply move
        current_points[i, 0] = nx
        current_points[i, 1] = ny
        point_set.discard((ox, oy))
        point_set.add((nx, ny))
        
        # Evaluate new score
        # We only need the top 27 lines' sum.
        # Calculating all counts is O(N*M). N=54, M~2000-5000.
        # 54 * 5000 = 270,000 ops. This is fast in NumPy.
        
        px = current_points[:, 0].astype(np.int64)
        py = current_points[:, 1].astype(np.int64)
        
        vals = A[:, None] * px[None, :] + B[:, None] * py[None, :] + C[:, None]
        counts = np.sum(vals == 0, axis=1)
        
        # Sum of top 27
        if n_lines > 27:
            # argpartition is faster than full sort
            top_idx = np.argpartition(-counts, 27)[:27]
            new_score = int(np.sum(counts[top_idx]))
        else:
            new_score = int(np.sum(counts))
            
        delta = new_score - current_score
        
        # Acceptance criterion
        if delta > 0 or random.random() < math.exp(delta / T):
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
        else:
            # Revert
            current_points[i, 0] = ox
            current_points[i, 1] = oy
            point_set.discard((nx, ny))
            point_set.add((ox, oy))
            
        T *= cooling_rate
        steps += 1
        
    return best_points, best_score

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s limit
    
    # Phase 1: Generate initial candidates and find the best starting point
    initial_sets = _generate_point_sets()
    
    best_points = None
    best_score = -1
    best_candidate_lines = None
    
    # We need a common candidate line set for SA, but it depends on points.
    # Strategy:
    # 1. For each initial set, build candidate lines.
    # 2. Find best 27 lines for that set.
    # 3. Keep the best configuration.
    # 4. For SA, we can use a large candidate set built from the *current* best points,
    #    or a universal set. A universal set is faster to prepare but might miss specific structures.
    #    Let's build a universal set from a standard grid and use it for all SA runs.
    
    # Build a robust universal candidate line set
    base_grid = np.array([[i, j] for i in range(6) for j in range(9)], dtype=np.int64)
    universal_lines, _ = _build_candidate_lines(base_grid, max_coeff=15, max_C_range=150)
    
    # Evaluate initial sets against universal lines
    for pts in initial_sets:
        if time.time() > deadline - 120:
            break
        # Just in case, ensure points are valid
        if len(set(map(tuple, pts.tolist()))) != 54:
            continue
            
        # Find best 27 lines from universal set
        best_lines_subset = _find_best_lines_for_points(pts, 27, universal_lines)
        score = _count_incidence(pts, best_lines_subset)
        
        if score > best_score:
            best_score = score
            best_points = pts.copy()
            best_candidate_lines = universal_lines # Keep reference
            
    if best_points is None:
        best_points = base_grid
        best_score = _count_incidence(best_points, _find_best_lines_for_points(best_points, 27, universal_lines))
        
    # Phase 2: Simulated Annealing
    # We use the universal lines for SA. This might be suboptimal if the best solution requires lines not in the universal set,
    # but the universal set is large (thousands of lines).
    # To be safe, we can periodically rebuild the candidate set or use a very large universal set.
    
    # Let's run SA for the remaining time
    if time.time() < deadline - 20:
        best_points, best_score = _simulated_annealing(
            best_points, 
            universal_lines, 
            deadline, 
            T_start=50.0, 
            T_end=0.1, 
            cooling_rate=0.999
        )
        
    # Phase 3: Final Polish
    # Rebuild candidate lines specific to the found points to ensure we capture the exact best 27 lines
    if time.time() < deadline - 5:
        specific_lines, _ = _build_candidate_lines(best_points, max_coeff=20, max_C_range=200)
        # Merge with universal lines to be safe
        all_lines = np.vstack([universal_lines, specific_lines])
        # Remove duplicates
        unique_lines_list = []
        seen = set()
        for l in all_lines:
            key = _normalize_line(int(l[0]), int(l[1]), int(l[2]))
            if key not in seen and (key[0] != 0 or key[1] != 0):
                seen.add(key)
                unique_lines_list.append(key)
        
        if unique_lines_list:
            final_candidate_lines = np.array(unique_lines_list, dtype=np.int64)
            final_best_lines = _find_best_lines_for_points(best_points, 27, final_candidate_lines)
            final_score = _count_incidence(best_points, final_best_lines)
            
            if final_score >= best_score:
                best_lines = final_best_lines
            else:
                # This shouldn't happen, but fallback to previous
                best_lines = _find_best_lines_for_points(best_points, 27, universal_lines)
        else:
            best_lines = _find_best_lines_for_points(best_points, 27, universal_lines)
    else:
        best_lines = _find_best_lines_for_points(best_points, 27, universal_lines)
        
    # Final Validation and Cleanup
    points = np.array(best_points, dtype=np.int64)
    lines = np.array(best_lines, dtype=np.int64)
    
    # Ensure exactly 54 unique points
    unique_pts = []
    seen = set()
    for p in points:
        key = (int(p[0]), int(p[1]))
        if key not in seen:
            seen.add(key)
            unique_pts.append(key)
    
    # If duplicates existed, we might have fewer than 54. Fill with far-away points.
    # However, SA maintains uniqueness, so this is a safety net.
    c = 1000
    while len(unique_pts) < 54:
        unique_pts.append((c, 2000 + c))
        c += 1
    unique_pts = unique_pts[:54]
    points = np.array(unique_pts, dtype=np.int64)
    
    # Ensure exactly 27 unique valid lines
    valid_lines = []
    seen_lines = set()
    for l in lines:
        key = _normalize_line(int(l[0]), int(l[1]), int(l[2]))
        if key not in seen_lines and (key[0] != 0 or key[1] != 0):
            seen_lines.add(key)
            valid_lines.append(key)
            
    c = 1000
    while len(valid_lines) < 27:
        key = _normalize_line(0, 1, -c)
        if key not in seen_lines:
            seen_lines.add(key)
            valid_lines.append(key)
        c += 1
        
    valid_lines = valid_lines[:27]
    lines = np.array(valid_lines, dtype=np.int64)
    
    return points, lines

if __name__ == "__main__":
    p, l = szemeredi_trotter_incidences()
    print(f"Score: {_count_incidence(p, l)}")
