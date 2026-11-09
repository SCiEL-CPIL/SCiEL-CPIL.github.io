import time
import random
import numpy as np
from collections import Counter
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill
    
    # Precompute all triples (i, j, k) with i < j < k for 30 points
    triples = list(combinations(range(30), 3))
    i_idx = np.array([t[0] for t in triples], dtype=np.int64)
    j_idx = np.array([t[1] for t in triples], dtype=np.int64)
    k_idx = np.array([t[2] for t in triples], dtype=np.int64)
    
    def score(points):
        """Compute the max multiplicity of any positive doubled area."""
        xs = np.array([p[0] for p in points], dtype=np.int64)
        ys = np.array([p[1] for p in points], dtype=np.int64)
        
        xi = xs[i_idx]
        yi = ys[i_idx]
        xj = xs[j_idx]
        yj = ys[j_idx]
        xk = xs[k_idx]
        yk = ys[k_idx]
        
        area2 = np.abs((xj - xi) * (yk - yi) - (yj - yi) * (xk - xi))
        
        mask = area2 > 0
        if not mask.any():
            return 0
        
        unique, counts = np.unique(area2[mask], return_counts=True)
        return int(counts.max())
    
    def score_with_areas(points):
        """Compute max multiplicity and return the dominant area value."""
        xs = np.array([p[0] for p in points], dtype=np.int64)
        ys = np.array([p[1] for p in points], dtype=np.int64)
        
        xi = xs[i_idx]
        yi = ys[i_idx]
        xj = xs[j_idx]
        yj = ys[j_idx]
        xk = xs[k_idx]
        yk = ys[k_idx]
        
        area2 = np.abs((xj - xi) * (yk - yi) - (yj - yi) * (xk - xi))
        
        mask = area2 > 0
        if not mask.any():
            return 0, 0
        
        unique, counts = np.unique(area2[mask], return_counts=True)
        max_idx = np.argmax(counts)
        return int(counts[max_idx]), int(unique[max_idx])
    
    def fast_score(pts):
        """Fast score computation using precomputed indices."""
        xs = np.array([p[0] for p in pts], dtype=np.int64)
        ys = np.array([p[1] for p in pts], dtype=np.int64)
        
        xi = xs[i_idx]
        yi = ys[i_idx]
        xj = xs[j_idx]
        yj = ys[j_idx]
        xk = xs[k_idx]
        yk = ys[k_idx]
        
        area2 = np.abs((xj - xi) * (yk - yi) - (yj - yi) * (xk - xi))
        
        mask = area2 > 0
        if not mask.any():
            return 0
        
        unique, counts = np.unique(area2[mask], return_counts=True)
        return int(counts.max())
    
    def count_target_area(pts, target_area):
        """Count how many triangles have exactly target_area as doubled area."""
        xs = np.array([p[0] for p in pts], dtype=np.int64)
        ys = np.array([p[1] for p in pts], dtype=np.int64)
        
        xi = xs[i_idx]
        yi = ys[i_idx]
        xj = xs[j_idx]
        yj = ys[j_idx]
        xk = xs[k_idx]
        yk = ys[k_idx]
        
        area2 = np.abs((xj - xi) * (yk - yi) - (yj - yi) * (xk - xi))
        
        return int(np.sum(area2 == target_area))
    
    def score_target(pts, target_area):
        """Score based on how many triangles match target_area."""
        return count_target_area(pts, target_area)
    
    # Generate structured candidates with focus on lattice patterns
    def make_grid(w, h, x0, y0, dx, dy):
        """Create a w x h grid starting at (x0, y0) with spacing (dx, dy)."""
        points = []
        for i in range(w):
            for j in range(h):
                x = x0 + i * dx
                y = y0 + j * dy
                if -30 <= x <= 30 and -30 <= y <= 30:
                    points.append((x, y))
        return points[:30] if len(points) >= 30 else None
    
    def make_staggered_grid(w, h, x0, y0, dx, dy):
        """Staggered (brick) grid."""
        points = []
        for j in range(h):
            offset = 0 if j % 2 == 0 else dx // 2
            for i in range(w):
                x = x0 + offset + i * dx
                y = y0 + j * dy
                if -30 <= x <= 30 and -30 <= y <= 30:
                    points.append((x, y))
        return points[:30] if len(points) >= 30 else None
    
    def make_hexagonal(rows, cols, x0, y0, dx, dy):
        """Hexagonal-like packing."""
        points = []
        for j in range(rows):
            offset = 0 if j % 2 == 0 else dx // 2
            for i in range(cols):
                x = x0 + offset + i * dx
                y = y0 + j * dy
                if -30 <= x <= 30 and -30 <= y <= 30:
                    points.append((x, y))
        return points[:30] if len(points) >= 30 else None
    
    # Generate diverse initial candidates
    candidates = []
    
    # Standard grids - focus on rectangular lattices
    for w in range(4, 11):
        for h in range(4, 11):
            if w * h >= 30:
                for x0 in range(-30, 10, 3):
                    for y0 in range(-30, 10, 3):
                        for dx in [2, 3, 4, 5, 6]:
                            for dy in [2, 3, 4, 5, 6]:
                                pts = make_grid(w, h, x0, y0, dx, dy)
                                if pts and len(set(pts)) == 30:
                                    candidates.append(pts)
    
    # Staggered grids
    for w in range(4, 11):
        for h in range(4, 11):
            if w * h >= 30:
                for x0 in range(-30, 10, 4):
                    for y0 in range(-30, 10, 4):
                        for dx in [3, 4, 5, 6]:
                            for dy in [3, 4, 5, 6]:
                                pts = make_staggered_grid(w, h, x0, y0, dx, dy)
                                if pts and len(set(pts)) == 30:
                                    candidates.append(pts)
    
    # Hexagonal patterns
    for rows in range(4, 11):
        for cols in range(4, 11):
            if rows * cols >= 30:
                for x0 in range(-30, 10, 4):
                    for y0 in range(-30, 10, 4):
                        for dx in [3, 4, 5, 6]:
                            for dy in [3, 4, 5, 6]:
                                pts = make_hexagonal(rows, cols, x0, y0, dx, dy)
                                if pts and len(set(pts)) == 30:
                                    candidates.append(pts)
    
    # Limit candidates to avoid too much evaluation time
    random.shuffle(candidates)
    candidates = candidates[:500]
    
    # Evaluate all candidates and find the best
    best_points = None
    best_score = 0
    best_target_area = 0
    
    for pts in candidates:
        if time.time() > deadline - 20:
            break
        s, ta = score_with_areas(pts)
        if s > best_score:
            best_score = s
            best_points = pts[:]
            best_target_area = ta
    
    # If no good candidate found, use the 6x5 grid as fallback
    if best_points is None:
        best_points = [(x, y) for x in range(6) for y in range(5)]
        best_score, best_target_area = score_with_areas(best_points)
    
    # Constraint-satisfaction local search: focus on matching a target area
    def local_search_cs(points, target_area, max_iters, time_limit):
        """Constraint-satisfaction local search focusing on target area."""
        current = points[:]
        current_score = fast_score(current)
        current_target_count = count_target_area(current, target_area)
        
        for it in range(max_iters):
            if time.time() > time_limit:
                break
            
            # Try moving a random point to a nearby location
            idx = random.randint(0, 29)
            
            found_improvement = False
            # Try several nearby positions
            for _ in range(15):
                # Random offset in a small neighborhood
                ox = random.randint(-4, 4)
                oy = random.randint(-4, 4)
                new_x = current[idx][0] + ox
                new_y = current[idx][1] + oy
                
                if not (-30 <= new_x <= 30 and -30 <= new_y <= 30):
                    continue
                
                new_point = (new_x, new_y)
                
                # Check distinctness
                if new_point in current[:idx] + current[idx+1:]:
                    continue
                
                candidate = current[:]
                candidate[idx] = new_point
                
                # Check if this improves the target area count
                new_target_count = count_target_area(candidate, target_area)
                new_general_score = fast_score(candidate)
                
                # Accept if it improves target count or general score
                if new_target_count > current_target_count or new_general_score > current_score:
                    current = candidate
                    current_score = new_general_score
                    current_target_count = new_target_count
                    found_improvement = True
                    break
                elif new_target_count == current_target_count and new_general_score >= current_score:
                    # Accept equal or slightly better moves with small probability
                    if random.random() < 0.15:
                        current = candidate
                        current_score = new_general_score
                        current_target_count = new_target_count
                        found_improvement = True
                        break
            
            if not found_improvement:
                # Try a larger jump
                new_x = random.randint(-30, 30)
                new_y = random.randint(-30, 30)
                new_point = (new_x, new_y)
                
                if new_point not in current[:idx] + current[idx+1:]:
                    candidate = current[:]
                    candidate[idx] = new_point
                    new_target_count = count_target_area(candidate, target_area)
                    new_general_score = fast_score(candidate)
                    if new_target_count > current_target_count or new_general_score > current_score:
                        current = candidate
                        current_score = new_general_score
                        current_target_count = new_target_count
        
        return current, current_score
    
    # Run constraint-satisfaction local search from the best candidate
    if time.time() < deadline - 5:
        best_points, best_score = local_search_cs(best_points, best_target_area, 5000, deadline - 2)
    
    # Try a few restarts from different structured candidates
    for restart in range(5):
        if time.time() > deadline - 5:
            break
        
        # Pick a random candidate or generate a new one
        if restart < len(candidates):
            start_pts = candidates[restart]
        else:
            # Generate a random configuration
            start_pts = []
            while len(start_pts) < 30:
                x = random.randint(-30, 30)
                y = random.randint(-30, 30)
                if (x, y) not in start_pts:
                    start_pts.append((x, y))
        
        if len(start_pts) != 30:
            continue
        
        # Get the dominant area for this candidate
        _, ta = score_with_areas(start_pts)
        
        result_pts, result_score = local_search_cs(start_pts, ta, 2000, deadline - 2)
        
        if result_score > best_score:
            best_score = result_score
            best_points = result_pts
    
    # Final validation
    assert len(best_points) == 30
    assert len(set(best_points)) == 30
    for x, y in best_points:
        assert -30 <= x <= 30 and -30 <= y <= 30
    
    return best_points
    
    # Generate structured candidates
    def make_grid(w, h, x0, y0, dx, dy):
        """Create a w x h grid starting at (x0, y0) with spacing (dx, dy)."""
        points = []
        for i in range(w):
            for j in range(h):
                x = x0 + i * dx
                y = y0 + j * dy
                if -30 <= x <= 30 and -30 <= y <= 30:
                    points.append((x, y))
        return points[:30] if len(points) >= 30 else None
    
    def make_lattice(rows, cols, x0, y0, dx, dy):
        """Alternative lattice generator."""
        points = []
        for i in range(rows):
            for j in range(cols):
                x = x0 + i * dx
                y = y0 + j * dy
                if -30 <= x <= 30 and -30 <= y <= 30:
                    points.append((x, y))
        return points[:30] if len(points) >= 30 else None
    
    def make_staggered_grid(w, h, x0, y0, dx, dy):
        """Staggered (brick) grid."""
        points = []
        for j in range(h):
            offset = 0 if j % 2 == 0 else dx // 2
            for i in range(w):
                x = x0 + offset + i * dx
                y = y0 + j * dy
                if -30 <= x <= 30 and -30 <= y <= 30:
                    points.append((x, y))
        return points[:30] if len(points) >= 30 else None
    
    def make_hexagonal(rows, cols, x0, y0, dx, dy):
        """Hexagonal-like packing."""
        points = []
        for j in range(rows):
            offset = 0 if j % 2 == 0 else dx // 2
            for i in range(cols):
                x = x0 + offset + i * dx
                y = y0 + j * dy
                if -30 <= x <= 30 and -30 <= y <= 30:
                    points.append((x, y))
        return points[:30] if len(points) >= 30 else None
    
    # Generate diverse initial candidates
    candidates = []
    
    # Standard grids
    for w in range(3, 11):
        for h in range(3, 11):
            if w * h >= 30:
                for x0 in range(-30, 10, 3):
                    for y0 in range(-30, 10, 3):
                        for dx in [2, 3, 4, 5, 6]:
                            for dy in [2, 3, 4, 5, 6]:
                                pts = make_grid(w, h, x0, y0, dx, dy)
                                if pts and len(set(pts)) == 30:
                                    candidates.append(pts)
    
    # Staggered grids
    for w in range(3, 11):
        for h in range(3, 11):
            if w * h >= 30:
                for x0 in range(-30, 10, 4):
                    for y0 in range(-30, 10, 4):
                        for dx in [3, 4, 5, 6]:
                            for dy in [3, 4, 5, 6]:
                                pts = make_staggered_grid(w, h, x0, y0, dx, dy)
                                if pts and len(set(pts)) == 30:
                                    candidates.append(pts)
    
    # Limit candidates to avoid too much evaluation time
    random.shuffle(candidates)
    candidates = candidates[:500]
    
    # Evaluate all candidates and find the best
    best_points = None
    best_score = 0
    
    for pts in candidates:
        if time.time() > deadline - 20:
            break
        s = score(pts)
        if s > best_score:
            best_score = s
            best_points = pts[:]
    
    # If no good candidate found, use the 6x5 grid as fallback
    if best_points is None:
        best_points = [(x, y) for x in range(6) for y in range(5)]
        best_score = score(best_points)
    
    # Local search: hill climbing with restarts
    def local_search(points, max_iters, time_limit):
        """Hill climbing local search."""
        current = points[:]
        current_score = score(current)
        
        # Precompute triples indices for speed
        i_idx = np.array([t[0] for t in triples], dtype=np.int64)
        j_idx = np.array([t[1] for t in triples], dtype=np.int64)
        k_idx = np.array([t[2] for t in triples], dtype=np.int64)
        
        def fast_score(pts):
            xs = np.array([p[0] for p in pts], dtype=np.int64)
            ys = np.array([p[1] for p in pts], dtype=np.int64)
            xi = xs[i_idx]
            yi = ys[i_idx]
            xj = xs[j_idx]
            yj = ys[j_idx]
            xk = xs[k_idx]
            yk = ys[k_idx]
            area2 = np.abs((xj - xi) * (yk - yi) - (yj - yi) * (xk - xi))
            mask = area2 > 0
            if not mask.any():
                return 0
            unique, counts = np.unique(area2[mask], return_counts=True)
            return int(counts.max())
        
        for it in range(max_iters):
            if time.time() > time_limit:
                break
            
            # Try moving a random point to a nearby location
            idx = random.randint(0, 29)
            
            # Try several nearby positions
            found_improvement = False
            for _ in range(10):
                # Random offset in a small neighborhood
                ox = random.randint(-5, 5)
                oy = random.randint(-5, 5)
                new_x = current[idx][0] + ox
                new_y = current[idx][1] + oy
                
                if not (-30 <= new_x <= 30 and -30 <= new_y <= 30):
                    continue
                
                new_point = (new_x, new_y)
                
                # Check distinctness
                if new_point in current[:idx] + current[idx+1:]:
                    continue
                
                candidate = current[:]
                candidate[idx] = new_point
                
                s = fast_score(candidate)
                
                if s > current_score:
                    current = candidate
                    current_score = s
                    found_improvement = True
                    break
                elif s == current_score and random.random() < 0.1:
                    # Accept equal-score moves with small probability to escape plateaus
                    current = candidate
                    found_improvement = True
                    break
            
            if not found_improvement:
                # Try a larger jump
                new_x = random.randint(-30, 30)
                new_y = random.randint(-30, 30)
                new_point = (new_x, new_y)
                
                if new_point not in current[:idx] + current[idx+1:]:
                    candidate = current[:]
                    candidate[idx] = new_point
                    s = fast_score(candidate)
                    if s > current_score:
                        current = candidate
                        current_score = s
        
        return current, current_score
    
    # Run local search from the best candidate
    if time.time() < deadline - 5:
        best_points, best_score = local_search(best_points, 5000, deadline - 2)
    
    # Try a few restarts from different structured candidates
    for restart in range(5):
        if time.time() > deadline - 5:
            break
        
        # Pick a random candidate or generate a new one
        if restart < len(candidates):
            start_pts = candidates[restart]
        else:
            # Generate a random configuration
            start_pts = []
            while len(start_pts) < 30:
                x = random.randint(-30, 30)
                y = random.randint(-30, 30)
                if (x, y) not in start_pts:
                    start_pts.append((x, y))
        
        if len(start_pts) != 30:
            continue
        
        result_pts, result_score = local_search(start_pts, 2000, deadline - 2)
        
        if result_score > best_score:
            best_score = result_score
            best_points = result_pts
    
    # Final validation
    assert len(best_points) == 30
    assert len(set(best_points)) == 30
    for x, y in best_points:
        assert -30 <= x <= 30 and -30 <= y <= 30
    
    return best_points
