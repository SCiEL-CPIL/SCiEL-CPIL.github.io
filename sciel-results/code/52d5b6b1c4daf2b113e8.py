"""
Program to construct 30 distinct integer coordinate pairs in [-30,30]^2
maximizing the largest multiplicity of a single positive doubled area.
"""

import time
import itertools
import numpy as np
from collections import Counter
import random

def _compute_all_areas(pts):
    """Compute all C(30,3) doubled areas. Returns array of 4060 values."""
    n = 30
    areas = np.zeros(4060, dtype=np.int64)
    idx = 0
    for i in range(28):
        xi, yi = pts[i]
        for j in range(i+1, 29):
            xj, yj = pts[j]
            dx1, dy1 = xj - xi, yj - yi
            for k in range(j+1, 30):
                xk, yk = pts[k]
                dx2, dy2 = xk - xi, yk - yi
                areas[idx] = abs(dx1 * dy2 - dy1 * dx2)
                idx += 1
    return areas

def _compute_area_counts(pts):
    """Compute Counter of positive doubled areas."""
    areas = _compute_all_areas(pts)
    positive = areas[areas > 0]
    if len(positive) == 0:
        return Counter()
    unique, counts = np.unique(positive, return_counts=True)
    return Counter(dict(zip(unique.tolist(), counts.tolist())))

def _max_mult_from_counts(counts):
    if not counts:
        return 0
    return max(counts.values())

def _valid(points):
    """Check 30 distinct integer points in [-30,30]^2."""
    if len(points) != 30:
        return False
    if len(set(points)) != 30:
        return False
    for x, y in points:
        if not (isinstance(x, int) and isinstance(y, int)):
            return False
        if x < -30 or x > 30 or y < -30 or y > 30:
            return False
    return True

def _score(points):
    """
    Compute the maximum multiplicity of a single positive doubled area.
    """
    if not _valid(points):
        return 0
    counts = _compute_area_counts(points)
    return _max_mult_from_counts(counts)

def _get_area_counts(points):
    """
    Returns a Counter of all positive doubled areas.
    """
    if not _valid(points):
        return Counter()
    return _compute_area_counts(points)

def _score_target(points, target_area):
    """
    Count the number of triangles with exactly target_area.
    """
    if not _valid(points):
        return 0
    counts = _get_area_counts(points)
    return counts.get(target_area, 0)

def _update_counts_for_move(counts, pts, moved_idx, old_pt, new_pt):
    """
    Update the area Counter after moving point at moved_idx from old_pt to new_pt.
    Only C(29,2)=406 areas change.
    """
    n = 30
    # Areas involving moved_idx: for each pair (a,b) from remaining 29 points
    # The triangle (a, moved_idx, b) changes
    # Need to handle ordering: triple (i,j,k) with i<j<k
    # If moved_idx is at position p, then:
    # - Triples where p is the smallest: (p, j, k) for j>k>p
    # - Triples where p is the middle: (i, p, k) for i<p<k
    # - Triples where p is the largest: (i, j, p) for i<j<p
    
    old_x, old_y = old_pt
    new_x, new_y = new_pt
    p = moved_idx
    
    # For each pair (a, b) from the other 29 points, compute old and new area
    # Triple ordering: sort the three indices
    for a in range(n):
        if a == p:
            continue
        for b in range(a+1, n):
            if b == p:
                continue
            # Triple is (min, mid, max) of {a, b, p}
            # Compute area with old point
            ix, iy = pts[a] if a < p else (pts[a])
            # Actually let's just compute directly
            # Points: a, b, and p (with old_pt or new_pt)
            ax, ay = pts[a]
            bx, by = pts[b]
            
            # Old area: triangle (a, b, old_pt)
            old_area = abs((bx - ax) * (old_y - ay) - (by - ay) * (old_x - ax))
            # New area: triangle (a, b, new_pt)
            new_area = abs((bx - ax) * (new_y - ay) - (by - ay) * (new_x - ax))
            
            if old_area > 0:
                counts[old_area] -= 1
                if counts[old_area] <= 0:
                    del counts[old_area]
            if new_area > 0:
                counts[new_area] = counts.get(new_area, 0) + 1
    
    return counts

def _update_counts_for_swap(counts, pts, idx1, idx2, pt1, pt2, new_pt1, new_pt2):
    """
    Update the area Counter after swapping points at idx1 and idx2.
    pt1 is at idx1, pt2 is at idx2. They swap to new_pt1 (at idx1) and new_pt2 (at idx2).
    Areas that change: all triples involving idx1 OR idx2.
    """
    n = 30
    p1, p2 = idx1, idx2
    if p1 > p2:
        p1, p2 = p2, p1
        pt1, pt2 = pt2, pt1
        new_pt1, new_pt2 = new_pt2, new_pt1
    
    old1x, old1y = pt1
    old2x, old2y = pt2
    new1x, new1y = new_pt1
    new2x, new2y = new_pt2
    
    # Triples involving p1 or p2 (but not both counted twice)
    # Triples with p1 only (not p2): pairs from remaining 28 points excluding p2
    # Triples with p2 only (not p1): pairs from remaining 28 points excluding p1
    # Triples with both p1 and p2: one triple per third point
    
    for c in range(n):
        if c == p1 or c == p2:
            continue
        cx, cy = pts[c]
        
        # Triple (p1, p2, c) - both change
        # Old area
        old_area = abs((old2x - old1x) * (cy - old1y) - (old2y - old1y) * (cx - old1x))
        # New area
        new_area = abs((new2x - new1x) * (cy - new1y) - (new2y - new1y) * (cx - new1x))
        
        if old_area > 0:
            counts[old_area] -= 1
            if counts[old_area] <= 0:
                del counts[old_area]
        if new_area > 0:
            counts[new_area] = counts.get(new_area, 0) + 1
    
    # Triples with p1 but not p2: pairs (a, b) where a,b != p1, p2 and a < b
    # The triple is sorted: (min(a,b,p1), mid, max)
    # But we just need to compute the area of triangle (a, b, p1)
    for a in range(n):
        if a == p1 or a == p2:
            continue
        for b in range(a+1, n):
            if b == p1 or b == p2:
                continue
            ax, ay = pts[a]
            bx, by = pts[b]
            
            # Old: triangle (a, b, old1)
            old_area = abs((bx - ax) * (old1y - ay) - (by - ay) * (old1x - ax))
            # New: triangle (a, b, new1)
            new_area = abs((bx - ax) * (new1y - ay) - (by - ay) * (new1x - ax))
            
            if old_area > 0:
                counts[old_area] -= 1
                if counts[old_area] <= 0:
                    del counts[old_area]
            if new_area > 0:
                counts[new_area] = counts.get(new_area, 0) + 1
    
    # Triples with p2 but not p1: pairs (a, b) where a,b != p1, p2 and a < b
    for a in range(n):
        if a == p1 or a == p2:
            continue
        for b in range(a+1, n):
            if b == p1 or b == p2:
                continue
            ax, ay = pts[a]
            bx, by = pts[b]
            
            # Old: triangle (a, b, old2)
            old_area = abs((bx - ax) * (old2y - ay) - (by - ay) * (old2x - ax))
            # New: triangle (a, b, new2)
            new_area = abs((bx - ax) * (new2y - ay) - (by - ay) * (new2x - ax))
            
            if old_area > 0:
                counts[old_area] -= 1
                if counts[old_area] <= 0:
                    del counts[old_area]
            if new_area > 0:
                counts[new_area] = counts.get(new_area, 0) + 1
    
    return counts

def _local_search_target(points, target_area, time_limit=10.0):
    """
    Local search using incremental area counting.
    Accepts moves that increase target_area count, or (with annealing) slightly decrease.
    Uses both single-point moves and swaps.
    """
    start_time = time.time()
    best_points = list(points)
    counts = _compute_area_counts(best_points)
    best_count = counts.get(target_area, 0)
    
    if best_count == 0:
        return best_points
    
    occupied = set(best_points)
    no_improve_count = 0
    max_no_improve = 150
    temperature = 1.5  # Annealing temperature
    best_global_count = best_count  # Track best seen overall for final polish
    
    while time.time() - start_time < time_limit:
        improved = False
        move_type = random.random()
        
        if move_type < 0.65:
            # Single point move
            random_idx = random.randint(0, 29)
            px, py = best_points[random_idx]
            
            # Try local moves first (small steps), then medium, then large
            for radius in [1, 2, 3, 4, 5, 6, 8]:
                for _ in range(20):
                    dx = random.randint(-radius, radius)
                    dy = random.randint(-radius, radius)
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = px + dx, py + dy
                    
                    if nx < -30 or nx > 30 or ny < -30 or ny > 30:
                        continue
                    
                    new_point = (nx, ny)
                    if new_point in occupied:
                        continue
                    
                    # Apply move incrementally
                    old_point = best_points[random_idx]
                    new_counts = _update_counts_for_move(counts, best_points, random_idx, old_point, new_point)
                    new_count = new_counts.get(target_area, 0)
                    
                    if new_count > best_count:
                        # Accept
                        best_points[random_idx] = new_point
                        occupied.discard(old_point)
                        occupied.add(new_point)
                        counts = new_counts
                        best_count = new_count
                        improved = True
                        no_improve_count = 0
                        break
                    elif new_count >= best_count - int(temperature):
                        # Annealing: accept with probability
                        delta = new_count - best_count
                        if delta < 0:
                            prob = 1.0 / (1.0 + (-delta) / max(temperature, 0.1))
                            if random.random() < prob:
                                best_points[random_idx] = new_point
                                occupied.discard(old_point)
                                occupied.add(new_point)
                                counts = new_counts
                                best_count = new_count
                                improved = True
                                no_improve_count = 0
                                break
                    # Revert (do nothing to best_points)
                if improved:
                    break
                
        else:
            # Swap two points
            idx1 = random.randint(0, 29)
            idx2 = random.randint(0, 29)
            if idx1 == idx2:
                continue
            p1 = best_points[idx1]
            p2 = best_points[idx2]
            
            for _ in range(40):
                # Try moving p1 to a random position near its current location
                radius = random.choice([2, 3, 4, 5])
                nx = p1[0] + random.randint(-radius, radius)
                ny = p1[1] + random.randint(-radius, radius)
                if nx < -30 or nx > 30 or ny < -30 or ny > 30:
                    continue
                new_p1 = (nx, ny)
                if new_p1 in occupied and new_p1 != p1 and new_p1 != p2:
                    continue
                if new_p1 == p2:
                    continue
                # Also try p2 to a random position near its current location
                radius2 = random.choice([2, 3, 4, 5])
                nx2 = p2[0] + random.randint(-radius2, radius2)
                ny2 = p2[1] + random.randint(-radius2, radius2)
                if nx2 < -30 or nx2 > 30 or ny2 < -30 or ny2 > 30:
                    continue
                new_p2 = (nx2, ny2)
                if new_p2 in occupied and new_p2 != p1 and new_p2 != p2:
                    continue
                if new_p2 == p1:
                    continue
                
                # Check both new positions are valid and distinct
                if new_p1 == new_p2:
                    continue
                
                # Apply swap incrementally
                new_counts = _update_counts_for_swap(counts, best_points, idx1, idx2, p1, p2, new_p1, new_p2)
                new_count = new_counts.get(target_area, 0)
                
                if new_count > best_count:
                    best_points[idx1] = new_p1
                    best_points[idx2] = new_p2
                    occupied.discard(p1)
                    occupied.discard(p2)
                    occupied.add(new_p1)
                    occupied.add(new_p2)
                    counts = new_counts
                    best_count = new_count
                    improved = True
                    no_improve_count = 0
                    break
                elif new_count >= best_count - int(temperature):
                    delta = new_count - best_count
                    if delta < 0:
                        prob = 1.0 / (1.0 + (-delta) / max(temperature, 0.1))
                        if random.random() < prob:
                            best_points[idx1] = new_p1
                            best_points[idx2] = new_p2
                            occupied.discard(p1)
                            occupied.discard(p2)
                            occupied.add(new_p1)
                            occupied.add(new_p2)
                            counts = new_counts
                            best_count = new_count
                            improved = True
                            no_improve_count = 0
                            break
        
        if not improved:
            no_improve_count += 1
            temperature *= 0.99  # Cool down faster
            if no_improve_count > max_no_improve:
                # Aggressive restart: perturb multiple points
                for _ in range(5):
                    ri = random.randint(0, 29)
                    old_pt = best_points[ri]
                    np_x = random.randint(-30, 30)
                    np_y = random.randint(-30, 30)
                    new_pt = (np_x, np_y)
                    if new_pt not in occupied:
                        new_counts = _update_counts_for_move(counts, best_points, ri, old_pt, new_pt)
                        nc = new_counts.get(target_area, 0)
                        if nc >= best_count - 2:
                            best_points[ri] = new_pt
                            occupied.discard(old_pt)
                            occupied.add(new_pt)
                            counts = new_counts
                            best_count = nc
                            no_improve_count = 0
                            break
                if no_improve_count > max_no_improve:
                    break
                    
    return best_points

def _generate_candidate(type_id, params=None):
    """
    Generate structured candidates known to have high symmetry.
    """
    if params is None:
        params = {}
        
    points = []
    
    if type_id == 'regular_polygon_concentric':
        # Concentric regular polygons
        ring_sizes = params.get('ring_sizes', [10, 10, 10])
        radii = params.get('radii', [10, 20, 30])
        
        for r, size in zip(radii, ring_sizes):
            for k in range(size):
                angle = 2 * np.pi * k / size
                x = int(round(r * np.cos(angle)))
                y = int(round(r * np.sin(angle)))
                points.append((x, y))
                
    elif type_id == 'triangular_lattice':
        # Simpler structured grid with shifts
        pts = []
        for r in range(5):
            y = (r - 2) * 4 # -8, -4, 0, 4, 8
            for c in range(6):
                x = (c - 2.5) * 4 # -10, -6, -2, 2, 6, 10
                if r % 2 == 1:
                    x += 2 
                pts.append((int(x), y))
        points = pts

    elif type_id == 'circle_grid':
        # Points on concentric circles
        for r, count in [(10, 10), (20, 10), (30, 10)]:
            for k in range(count):
                angle = 2 * np.pi * k / count
                x = int(round(r * np.cos(angle)))
                y = int(round(r * np.sin(angle)))
                points.append((x, y))
                
    elif type_id == 'square_grid':
        # Standard 5x6 grid
        pts = []
        for x in range(-10, 11, 4):
            for y in range(-8, 9, 4):
                pts.append((x, y))
        points = pts
        
    elif type_id == 'dense_grid_1':
        # Dense 5x6 grid with smaller spacing
        pts = []
        for x in range(-8, 9, 3):
            for y in range(-7, 8, 3):
                pts.append((x, y))
        # Take first 30
        points = pts[:30]
        
    elif type_id == 'dense_grid_2':
        # 6x5 grid with different spacing
        pts = []
        for x in range(-9, 10, 4):
            for y in range(-10, 11, 5):
                pts.append((x, y))
        points = pts[:30]
        
    elif type_id == 'rect_grid_3x10':
        # 3x10 rectangle
        pts = []
        for x in range(-15, 16, 3):
            for y in range(-5, 6, 10//3):
                pts.append((x, y))
        # Adjust to get exactly 30
        while len(pts) < 30:
            pts.append((0, 0))
        points = pts[:30]
        
    elif type_id == 'hexagonal':
        # Hexagonal lattice pattern
        pts = []
        # Center
        pts.append((0, 0))
        # First ring (6 points)
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1)]:
            pts.append((dx*2, dy*2))
        # Second ring (12 points)
        for dx, dy in [(2,0),(-2,0),(0,2),(0,-2),(2,2),(-2,-2),(2,-2),(-2,2),(1,2),(-1,-2),(1,-2),(-1,2)]:
            pts.append((dx*2, dy*2))
        # Third ring (11 points to reach 30)
        for dx, dy in [(3,0),(-3,0),(0,3),(0,-3),(3,3),(-3,-3),(3,-3),(-3,3),(2,3),(-2,-3),(2,-3)]:
            pts.append((dx*2, dy*2))
        points = pts[:30]

    return points

def construct():
    """
    Construct 30 points maximizing the multiplicity of a single doubled area.
    Strategy: 
    1. Generate diverse initial candidates.
    2. Identify top candidate areas (targets) from these candidates.
    3. Run specialized local search with incremental scoring for each target area.
    4. Return the best configuration found.
    """
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s margin
    
    candidates = []
    
    # 1. Generate diverse candidates
    # Regular polygons
    candidates.append(_generate_candidate('regular_polygon_concentric', {'ring_sizes': [15, 15], 'radii': [15, 30]}))
    candidates.append(_generate_candidate('regular_polygon_concentric', {'ring_sizes': [10, 10, 10], 'radii': [10, 20, 30]}))
    candidates.append(_generate_candidate('regular_polygon_concentric', {'ring_sizes': [6, 6, 6, 6, 6], 'radii': [6, 12, 18, 24, 30]}))
    
    # Grids - more diverse
    candidates.append(_generate_candidate('square_grid'))
    candidates.append(_generate_candidate('triangular_lattice'))
    candidates.append(_generate_candidate('circle_grid'))
    candidates.append(_generate_candidate('dense_grid_1'))
    candidates.append(_generate_candidate('dense_grid_2'))
    candidates.append(_generate_candidate('hexagonal'))
    
    # Random
    for _ in range(12):
        pts = []
        while len(pts) < 30:
            x = random.randint(-30, 30)
            y = random.randint(-30, 30)
            if (x, y) not in pts:
                pts.append((x, y))
        candidates.append(pts)
        
    # Filter valid candidates
    valid_candidates = [c for c in candidates if _valid(c)]
    
    if not valid_candidates:
        return [(x, y) for x in range(6) for y in range(5)]
        
    # 2. Analyze candidates to find promising target areas
    scored_candidates = []
    for cand in valid_candidates:
        s = _score(cand)
        if s > 0:
            scored_candidates.append((s, cand))
            
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Take top 5 candidates (or fewer if not enough)
    top_candidates = [c[1] for c in scored_candidates[:5]]
    
    # Collect frequent areas from these top candidates
    area_counts_aggregate = Counter()
    for cand in top_candidates:
        counts = _get_area_counts(cand)
        area_counts_aggregate.update(counts)
        
    # Select top 6 most frequent areas as targets
    if area_counts_aggregate:
        top_areas = [area for area, _ in area_counts_aggregate.most_common(6)]
    else:
        top_areas = [2, 4, 6, 8, 10, 12]
        
    # 3. Run specialized local search for each target
    best_points = None
    best_global_score = 0
    
    for target in top_areas:
        if time.time() > deadline - 8:
            break
            
        # Pick the best initial candidate for this specific target
        best_init_for_target = None
        max_count_for_target = 0
        for cand in valid_candidates:
            c = _score_target(cand, target)
            if c > max_count_for_target:
                max_count_for_target = c
                best_init_for_target = cand
                
        if best_init_for_target is None:
            best_init_for_target = valid_candidates[0]
            
        # Run local search
        remaining = deadline - time.time()
        search_time = min(15.0, remaining / 2.5)
        if search_time < 1.0:
            break
            
        result_points = _local_search_target(best_init_for_target, target, time_limit=search_time)
        
        # Evaluate the result with the GLOBAL score
        global_s = _score(result_points)
        
        if global_s > best_global_score:
            best_global_score = global_s
            best_points = result_points
            
    # If no search was run or failed, use the best initial candidate
    if best_points is None:
        best_points = valid_candidates[0]
        
    # Final polish: try to improve the best configuration found
    if time.time() < deadline - 3:
        # Run one more round of local search on the best result
        remaining = deadline - time.time() - 2
        if remaining > 2.0:
            # Find the target area that gives the best count for best_points
            counts = _compute_area_counts(best_points)
            if counts:
                top_target = max(counts, key=counts.get)
                polished = _local_search_target(best_points, top_target, time_limit=min(8.0, remaining))
                polished_score = _score(polished)
                if polished_score > best_global_score:
                    best_points = polished
                    best_global_score = polished_score
    
    # Final check
    if not _valid(best_points):
        return [(x, y) for x in range(6) for y in range(5)]
        
    return best_points

# EVOLVE-BLOCK-END
