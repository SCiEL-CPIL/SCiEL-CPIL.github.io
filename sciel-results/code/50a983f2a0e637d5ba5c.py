# EVOLVE-BLOCK-START
import time
import random
import itertools
from math import gcd
import numpy as np


def _normalize_line(A, B, C):
    """Normalize (A,B,C) to canonical form: gcd(|A|,|B|,|C|) divided out,
    first nonzero entry positive."""
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g == 0:
        return None
    A, B, C = A // g, B // g, C // g
    # Make first nonzero entry positive
    if A < 0 or (A == 0 and (B < 0 or (B == 0 and C < 0))):
        A, B, C = -A, -B, -C
    return (A, B, C)


def _is_valid(points, lines):
    """Check validity of (points, lines)."""
    if points.shape != (54, 2) or lines.shape != (27, 3):
        return False
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(lines)):
        return False
    if np.any(np.abs(points - np.round(points)) > 1e-6):
        return False
    if np.any(np.abs(lines - np.round(lines)) > 1e-6):
        return False
    
    pts = points.astype(np.int64)
    if len(set(map(tuple, pts.tolist()))) != 54:
        return False
    
    lns = lines.astype(np.int64)
    for row in lns:
        if row[0] == 0 and row[1] == 0:
            return False
            
    canon = set()
    for row in lns:
        c = _normalize_line(int(row[0]), int(row[1]), int(row[2]))
        if c is None:
            return False
        if c in canon:
            return False
        canon.add(c)
    return True


def _count_incidences(points, lines):
    """Count point-line incidences exactly in int64."""
    pts = points.astype(np.int64)
    lns = lines.astype(np.int64)
    count = 0
    for A, B, C in lns:
        if A == 0 and B == 0:
            continue
        val = A * pts[:, 0] + B * pts[:, 1] + C
        count += int(np.sum(val == 0))
    return count


def _generate_candidate_point_sets():
    """
    Generate a diverse set of candidate point configurations.
    We focus on structures that are likely to support high incidence:
    1. Rectangular grids (k x m)
    2. Punctured grids (remove some points to allow denser line packing? No, keep full grids mostly)
    3. Shifted grids
    4. Random integer points within a bounding box
    """
    candidates = []
    
    # 1. Standard Grids
    # n=54. Factors: 1, 2, 3, 6, 9, 18, 27, 54
    dims = [(1, 54), (2, 27), (3, 18), (6, 9), (9, 6), (18, 3), (27, 2), (54, 1)]
    for w, h in dims:
        xs = np.arange(w)
        ys = np.arange(h)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        pts = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        candidates.append(pts)
        
        # Shifted grids (shift by 1)
        pts_shifted = pts + 1
        candidates.append(pts_shifted)
        
    # 2. Near-Grid structures: Take a 6x9 grid and randomly perturb a few points
    # This helps break symmetry if the optimal solution is slightly off-grid
    base_6x9 = candidates[3] # 6x9 is at index 3? Let's just generate one
    xs = np.arange(6)
    ys = np.arange(9)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    base_pts = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    for _ in range(20):
        perturbed = base_pts.copy()
        # Perturb 5 random points by -1, 0, 1
        indices = random.sample(range(54), 5)
        for idx in indices:
            perturbed[idx, 0] += random.randint(-1, 1)
            perturbed[idx, 1] += random.randint(-1, 1)
        # Ensure uniqueness
        if len(set(map(tuple, perturbed.tolist()))) == 54:
            candidates.append(perturbed)

    # 3. Random integer points in a box
    # Box size should be small enough that lines have small coefficients
    for box_size in [10, 15, 20]:
        for _ in range(10):
            pts = np.random.randint(0, box_size, size=(54, 2)).astype(np.int64)
            # Ensure uniqueness
            unique_pts = []
            seen = set()
            for p in pts:
                t = tuple(p)
                if t not in seen:
                    seen.add(t)
                    unique_pts.append(t)
            if len(unique_pts) == 54:
                candidates.append(np.array(unique_pts, dtype=np.int64))
            else:
                # Fill in missing random points
                while len(unique_pts) < 54:
                    p = np.random.randint(0, box_size, size=2)
                    t = tuple(p)
                    if t not in seen:
                        seen.add(t)
                        unique_pts.append(t)
                candidates.append(np.array(unique_pts, dtype=np.int64))

    return candidates


def _find_best_lines_for_points(points, n_lines=27):
    """
    Given a fixed set of 54 points, find the 27 lines that maximize incidences.
    Strategy:
    1. Enumerate all lines passing through at least 2 points.
    2. Sort by number of points on the line (descending).
    3. Greedy selection: Pick the line with the most *new* points not yet covered by selected lines?
       Actually, we want to maximize TOTAL incidences.
       Let S be the set of selected lines.
       Incidence = sum_{l in S} |P(l) ∩ Points|.
       This is not just about unique points. A line with 5 points is better than a line with 3,
       even if they overlap.
       
       However, if we have 27 lines, and we pick the top 27 by absolute count, we might pick
       27 parallel lines that are all distinct but cover the same points.
       
       Let's use a greedy approach that maximizes marginal gain in total incidence count.
       Current Incidence I.
       For each candidate line l not in S:
           Gain = |P(l) ∩ Points|
           Wait, the total incidence is just the sum of counts for each line.
           So the "best" set of 27 lines is simply the 27 lines with the highest individual point counts!
           
           Is that true?
           I = sum_{i=1}^{27} c_i, where c_i is the number of points on line i.
           Yes! The objective function is linear in the line selection if the points are fixed.
           There is no penalty for overlapping lines in the incidence count definition 
           (I is the number of (point, line) pairs).
           
           So, for a FIXED set of points, the optimal set of 27 lines is simply the 27 lines
           that pass through the most points.
    """
    pts = points
    n = len(pts)
    
    # Map point index to coordinates
    # We need to find all lines through >= 2 points.
    # To avoid O(N^3) or high constant factor, we use the property that 
    # lines through 2 points are unique.
    
    line_counts = {} # normalized line -> count of points
    
    # Optimization: Only consider lines with small coefficients?
    # No, we must consider all lines through pairs of points in the set.
    # N=54, pairs = 54*53/2 = 1431. This is very small.
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                continue
            A = dy
            B = -dx
            C = x1 * y2 - x2 * y1
            c = _normalize_line(A, B, C)
            if c is None:
                continue
            
            if c not in line_counts:
                line_counts[c] = 0
            # We just need to know if this line is "new" for this pair?
            # No, we need the total count of points on this line.
            # Storing a set is safer.
            if c not in line_counts:
                line_counts[c] = set()
            line_counts[c].add(i)
            line_counts[c].add(j)
            
    # Convert sets to counts
    line_list = [(c, len(s)) for c, s in line_counts.items()]
    
    # Sort by count descending
    line_list.sort(key=lambda x: -x[1])
    
    # Select top 27
    selected = line_list[:n_lines]
    
    if len(selected) < n_lines:
        # If we don't have enough lines (shouldn't happen with 54 points), 
        # add arbitrary valid lines
        # Generate random lines
        while len(selected) < n_lines:
            A = random.randint(-10, 10)
            B = random.randint(-10, 10)
            if A == 0 and B == 0:
                continue
            C = random.randint(-50, 50)
            c = _normalize_line(A, B, C)
            if c and c not in [s[0] for s in selected]:
                selected.append((c, 0))
                
    lines = np.array([s[0] for s in selected], dtype=np.int64)
    return lines


def _local_search(points, lines, time_limit):
    """
    Improve (points, lines) via local search.
    Moves:
    1. Move a point to a nearby integer location (if it doesn't collide).
    2. Change a line to a nearby integer line (if it doesn't duplicate).
    """
    best_count = _count_incidences(points, lines)
    current_points = points.copy()
    current_lines = lines.copy()
    
    start_time = time.time()
    
    # Precompute incidence matrix? 
    # (54, 27) boolean matrix. 
    # Moving a point changes 27 entries. 
    # Changing a line changes 54 entries.
    
    while time.time() < time_limit:
        improved = False
        # Try a few random moves
        for _ in range(10):
            move_type = random.choice([0, 1])
            
            if move_type == 0:
                # Move a point
                idx = random.randint(0, 53)
                old_p = current_points[idx].copy()
                # Try small shifts
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        new_p = old_p + np.array([dx, dy])
                        # Check collision
                        # Simple check: is new_p in current_points?
                        # O(N) check is fine for 54 points.
                        if np.any(np.all(current_points == new_p, axis=1)):
                            continue
                        
                        # Apply
                        current_points[idx] = new_p
                        
                        # Check validity of points (integers, finite)
                        # They are integers by construction.
                        
                        # Recalculate count
                        # We can update incrementally, but _count_incidences is fast enough (54*27 ops)
                        new_count = _count_incidences(current_points, current_lines)
                        
                        if new_count > best_count:
                            best_count = new_count
                            improved = True
                            # Keep the move
                            break
                        else:
                            # Revert
                            current_points[idx] = old_p
                    if improved:
                        break
            else:
                # Change a line
                idx = random.randint(0, 26)
                old_l = current_lines[idx].copy()
                
                # Generate a candidate line
                # Strategy: Find a line that passes through 2 existing points
                # This guarantees high incidence.
                p1_idx = random.randint(0, 53)
                p2_idx = random.randint(0, 53)
                if p1_idx == p2_idx:
                    continue
                p1 = current_points[p1_idx]
                p2 = current_points[p2_idx]
                
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                if dx == 0 and dy == 0:
                    continue
                
                A = dy
                B = -dx
                C = p1[0] * p2[1] - p2[0] * p1[1]
                new_l_canon = _normalize_line(A, B, C)
                
                if new_l_canon is None:
                    continue
                
                new_l = np.array(new_l_canon, dtype=np.int64)
                
                # Check duplicate line
                # Check if new_l_canon is already in current_lines (normalized)
                dup = False
                for l in current_lines:
                    c = _normalize_line(int(l[0]), int(l[1]), int(l[2]))
                    if c == new_l_canon:
                        if idx == np.where(np.all(current_lines == l, axis=1))[0][0]:
                            # It's the same line, so no change
                            dup = True
                            break
                        else:
                            dup = True
                            break
                if dup:
                    continue
                
                # Apply
                current_lines[idx] = new_l
                
                new_count = _count_incidences(current_points, current_lines)
                
                if new_count > best_count:
                    best_count = new_count
                    improved = True
                    break
                else:
                    current_lines[idx] = old_l
        
        if not improved:
            # Try a slightly larger move or break if stuck for too long?
            # For now, just continue to try random moves.
            pass
            
    return current_points, current_lines, best_count


def szemeredi_trotter_incidences():
    """
    Szemeredi-Trotter incidence maximization.
    """
    start_time = time.time()
    deadline = start_time + 600
    safety_margin = 30
    
    best_points = None
    best_lines = None
    best_count = -1
    
    # 1. Generate candidates
    candidates = _generate_candidate_point_sets()
    
    # 2. For each candidate, find best lines and evaluate
    for i, pts in enumerate(candidates):
        if time.time() > deadline - safety_margin:
            break
            
        # Check validity of points first (uniqueness)
        if len(set(map(tuple, pts.tolist()))) != 54:
            continue
            
        lines = _find_best_lines_for_points(pts)
        
        # Check validity of full config
        if not _is_valid(pts, lines):
            continue
            
        count = _count_incidences(pts, lines)
        
        if count > best_count:
            best_count = count
            best_points = pts.copy()
            best_lines = lines.copy()
            
        # 3. Local Search on the best found so far? 
        # Or on each candidate? 
        # Local search is expensive. Let's run it on the top few candidates.
        # To save time, we'll just run it on the current best if we have time.
        
    # 4. Local Search Refinement
    if best_points is not None and time.time() < deadline - safety_margin:
        # Run local search for the remaining time
        ls_time_limit = deadline - safety_margin
        best_points, best_lines, final_count = _local_search(best_points, best_lines, ls_time_limit)
        if final_count > best_count:
            best_count = final_count
            # Update bests if improved (already done in _local_search return, but ensure variables are synced)
            # _local_search returns the improved points/lines
            pass # best_points and best_lines are already the improved ones from _local_search
            
    return best_points, best_lines

# EVOLVE-BLOCK-END
