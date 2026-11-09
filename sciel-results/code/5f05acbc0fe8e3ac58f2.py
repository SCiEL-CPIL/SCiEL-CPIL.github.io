# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from itertools import combinations


def _canonical_line(a, b, c):
    """Normalize line (A,B,C) so first nonzero entry is positive and gcd divides all."""
    if a == 0 and b == 0:
        return (0, 0, 0)
    g = math.gcd(abs(a), math.gcd(abs(b), abs(c)))
    if g > 0:
        a, b, c = a // g, b // g, c // g
    if a < 0:
        a, b, c = -a, -b, -c
    elif a == 0 and b < 0:
        a, b, c = -a, -b, -c
    return (a, b, c)


def _line_from_points(p1, p2):
    """Given two points (x1,y1) and (x2,y2), return canonical line (A,B,C)."""
    x1, y1 = p1
    x2, y2 = p2
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2
    return _canonical_line(a, b, c)


def _count_incidences(points, lines):
    """Count incidences: number of (point, line) pairs where A*x+B*y+C==0."""
    total = 0
    for i in range(lines.shape[0]):
        A, B, C = lines[i]
        vals = A * points[:, 0] + B * points[:, 1] + C
        total += np.sum(vals == 0)
    return total


def _get_top_k_lines(points, k=27, deadline=None):
    """Find the top k lines by incidence count for the given points.
    Returns exactly k lines. If fewer unique lines exist, pads with arbitrary unique lines."""
    n = len(points)
    line_counts = {}
    
    # Generate all lines from pairs of points
    for i in range(n):
        if deadline is not None and time.time() > deadline:
            break
        for j in range(i + 1, n):
            line = _line_from_points(points[i], points[j])
            if line not in line_counts:
                line_counts[line] = 0
            line_counts[line] += 1
            
    # Count actual points on each line
    unique_lines = list(line_counts.keys())
    scored_lines = []
    for line in unique_lines:
        A, B, C = line
        count = 0
        for p in points:
            if A * p[0] + B * p[1] + C == 0:
                count += 1
        scored_lines.append((count, line))
        
    # Sort by count descending
    scored_lines.sort(key=lambda x: -x[0])
    
    # Pick top k
    top_lines = [l[1] for l in scored_lines[:k]]
    
    # If we don't have enough unique lines, we need to pad.
    # This can happen if points are highly collinear (e.g., 1x54 grid has only 1 unique line).
    # We must return exactly k lines. We can reuse lines? No, "two rows describing the same line" is invalid.
    # So we need *distinct* lines.
    # If the set of all lines defined by pairs is smaller than k, we are in trouble for validity.
    # However, for 1x54, there is only 1 line. We cannot form 27 distinct lines.
    # But the task requires returning 27 lines. If it's impossible to have 27 distinct lines, 
    # the configuration is invalid. 
    # Wait, the error was "cannot reshape array of size 3". This implies _get_top_k_lines returned 1 line.
    # If we return 1 line, reshape fails.
    # To make the program *run* and return *something* (even if score is 0 or low), we should avoid crashing.
    # But the contract says "Invalid: ... two rows describing the same line".
    # If we pad with duplicates, it's invalid.
    # If we can't find 27 distinct lines, we should probably skip this grid configuration or return a valid fallback.
    # However, the main goal is to fix the crash. 
    # Let's check if we can generate more lines. 
    # Actually, if points are all on one line, we can't make 27 distinct lines from pairs.
    # But we can define lines that don't pass through any pair? No, lines are defined by A,B,C.
    # Any line is valid. We just need 27 distinct lines.
    # We can generate arbitrary lines that don't intersect the points?
    # Or just lines that are distinct.
    # Let's try to generate arbitrary distinct lines if we don't have enough from pairs.
    
    if len(top_lines) < k:
        existing = set(top_lines)
        # Generate arbitrary lines
        # Line: x = c, y = c, x+y=c, etc.
        # We just need to ensure they are distinct and not in existing.
        # Also they should be "valid" (A,B,C not all 0).
        # Let's generate lines x = i for i in range...
        # And y = i
        # And x+y=i
        # We need to make sure they are distinct from each other and existing.
        
        # Simple strategy: iterate through small integers for A,B,C
        # and check if canonical form is in existing.
        # But we also want them to be "bad" (0 incidences) so they don't interfere, 
        # or just distinct.
        # Since we are padding, we likely want 0 incidence lines to not change the score of the good ones?
        # No, the score is sum of incidences. Adding 0-incidence lines is fine.
        
        counter = 0
        while len(top_lines) < k and counter < 1000:
            # Try lines x = counter
            l = _canonical_line(1, 0, -counter)
            if l not in existing:
                top_lines.append(l)
                existing.add(l)
                continue
            # Try lines y = counter
            l = _canonical_line(0, 1, -counter)
            if l not in existing:
                top_lines.append(l)
                existing.add(l)
                continue
            # Try lines x + y = counter
            l = _canonical_line(1, 1, -counter)
            if l not in existing:
                top_lines.append(l)
                existing.add(l)
                continue
            # Try lines x - y = counter
            l = _canonical_line(1, -1, -counter)
            if l not in existing:
                top_lines.append(l)
                existing.add(l)
                continue
            counter += 1
        
        # If we still don't have enough, something is very wrong, but we truncate/pad to k to avoid crash
        # If we still have less than k, we repeat the last one? No, invalid.
        # If we truly cannot find k distinct lines, we should probably return None or a valid fallback.
        # But for now, let's assume we can find enough arbitrary lines.
        
        if len(top_lines) < k:
            # Fallback: just return what we have? No, shape mismatch.
            # Let's force fill with distinct lines using a larger range
            for a in range(-10, 11):
                for b in range(-10, 11):
                    if a == 0 and b == 0: continue
                    for c in range(-50, 51):
                        l = _canonical_line(a, b, c)
                        if l not in existing:
                            top_lines.append(l)
                            existing.add(l)
                            if len(top_lines) >= k:
                                break
                    if len(top_lines) >= k:
                        break
                if len(top_lines) >= k:
                    break

    return top_lines[:k]


def _local_search_lines(points, initial_lines, k=27, deadline=None):
    """Local search to improve line selection by swapping lines."""
    n_points = len(points)
    n_lines = k
    
    # Precompute all unique lines and their point counts
    all_lines = {}
    for i in range(n_points):
        if deadline is not None and time.time() > deadline:
            break
        for j in range(i + 1, n_points):
            line = _line_from_points(points[i], j if isinstance(j, tuple) else points[j])
            if line not in all_lines:
                # Count points on this line
                A, B, C = line
                count = 0
                for p in points:
                    if A * p[0] + B * p[1] + C == 0:
                        count += 1
                all_lines[line] = count
    
    # Current set of lines
    current_lines = list(initial_lines)
    current_total = 0
    for line in current_lines:
        current_total += all_lines.get(line, 0)
    
    # Local search: try swapping each line with a better one
    improved = True
    while improved and (deadline is None or time.time() < deadline):
        improved = False
        for i in range(n_lines):
            if deadline is not None and time.time() > deadline:
                break
            current_line = current_lines[i]
            current_count = all_lines.get(current_line, 0)
            
            # Find the best replacement line not already in the set
            best_line = None
            best_count = current_count
            for line, count in all_lines.items():
                if line in current_lines:
                    continue
                if count > best_count:
                    best_count = count
                    best_line = line
            
            if best_line is not None and best_count > current_count:
                current_lines[i] = best_line
                current_total += best_count - current_count
                improved = True
    
    return current_lines


def _try_grid(rows, cols, deadline=None):
    """Try a grid configuration and return (points, lines, score)."""
    if rows * cols != 54:
        return None, None, -1
    
    xs = np.arange(rows)
    ys = np.arange(cols)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    # Find top 27 lines
    lines_list = _get_top_k_lines(points, 27, deadline=deadline)
    
    # Ensure we have exactly 27 lines
    if len(lines_list) != 27:
        # If _get_top_k_lines failed to provide enough, we can't proceed with this grid
        # unless we pad, but _get_top_k_lines now handles padding.
        # If it still fails (len < 27), return None
        if len(lines_list) < 27:
             return None, None, -1
        else:
             lines_list = lines_list[:27]

    lines_arr = np.array(lines_list, dtype=np.int64).reshape(27, 3)
    
    # Local search to improve
    lines_list = _local_search_lines(points, lines_list, 27, deadline=deadline)
    
    # Ensure we have exactly 27 lines after local search too (though it shouldn't change count)
    if len(lines_list) != 27:
        if len(lines_list) < 27:
             return None, None, -1
        else:
             lines_list = lines_list[:27]

    lines_arr = np.array(lines_list, dtype=np.int64).reshape(27, 3)
    
    # Check for duplicate lines
    canonical_lines = [_canonical_line(*l) for l in lines_arr]
    if len(set(canonical_lines)) != 27:
        unique = []
        seen = set()
        for cl in canonical_lines:
            if cl not in seen:
                seen.add(cl)
                unique.append(cl)
        if len(unique) < 27:
            return None, None, -1
        lines_arr = np.array(unique[:27], dtype=np.int64).reshape(27, 3)
    
    # Check for duplicate points
    point_tuples = [tuple(p) for p in points]
    if len(set(point_tuples)) != 54:
        return None, None, -1
    
    # Count incidences
    score = _count_incidences(points, lines_arr)
    
    return points, lines_arr, score


def szemeredi_trotter_incidences():
    """
    Maximizes point-line incidences with 54 points and 27 lines.
    Uses grid configurations with local search refinement.
    """
    start_time = time.time()
    deadline = start_time + 590  # Safety margin before 600s limit
    
    # Try multiple grid configurations
    best_points = None
    best_lines = None
    best_score = -1
    
    # 6x9 grid
    pts, lns, sc = _try_grid(6, 9, deadline=deadline)
    if pts is not None and sc > best_score:
        best_points, best_lines, best_score = pts, lns, sc
    
    # 9x6 grid
    if time.time() < deadline:
        pts, lns, sc = _try_grid(9, 6, deadline=deadline)
        if pts is not None and sc > best_score:
            best_points, best_lines, best_score = pts, lns, sc
    
    # 2x27 grid
    if time.time() < deadline:
        pts, lns, sc = _try_grid(2, 27, deadline=deadline)
        if pts is not None and sc > best_score:
            best_points, best_lines, best_score = pts, lns, sc
    
    # 27x2 grid
    if time.time() < deadline:
        pts, lns, sc = _try_grid(27, 2, deadline=deadline)
        if pts is not None and sc > best_score:
            best_points, best_lines, best_score = pts, lns, sc
    
    # 3x18 grid
    if time.time() < deadline:
        pts, lns, sc = _try_grid(3, 18, deadline=deadline)
        if pts is not None and sc > best_score:
            best_points, best_lines, best_score = pts, lns, sc
    
    # 18x3 grid
    if time.time() < deadline:
        pts, lns, sc = _try_grid(18, 3, deadline=deadline)
        if pts is not None and sc > best_score:
            best_points, best_lines, best_score = pts, lns, sc
    
    # 1x54 grid
    if time.time() < deadline:
        pts, lns, sc = _try_grid(1, 54, deadline=deadline)
        if pts is not None and sc > best_score:
            best_points, best_lines, best_score = pts, lns, sc
    
    # 54x1 grid
    if time.time() < deadline:
        pts, lns, sc = _try_grid(54, 1, deadline=deadline)
        if pts is not None and sc > best_score:
            best_points, best_lines, best_score = pts, lns, sc
    
    # If no valid result found, use 6x9 as fallback
    if best_points is None:
        xs = np.arange(6)
        ys = np.arange(9)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        best_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        lines_list = _get_top_k_lines(best_points, 27)
        
        if len(lines_list) < 27:
            # Extremely unlikely for 6x9, but handle just in case
            # Pad with arbitrary lines
            existing = set(lines_list)
            counter = 0
            while len(lines_list) < 27 and counter < 100:
                l = _canonical_line(1, 0, -counter)
                if l not in existing:
                    lines_list.append(l)
                    existing.add(l)
                counter += 1
            lines_list = lines_list[:27]
            
        best_lines = np.array(lines_list, dtype=np.int64).reshape(27, 3)
    
    return best_points, best_lines

# EVOLVE-BLOCK-END
