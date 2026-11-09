"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import Counter
import numpy as np

def _doubled_areas(points):
    """Compute all doubled absolute areas for triples of points."""
    n = len(points)
    areas = []
    for i in range(n):
        xi, yi = points[i]
        for j in range(i+1, n):
            xj, yj = points[j]
            for k in range(j+1, n):
                xk, yk = points[k]
                det = abs(xi*(yj-yk) + xj*(yk-yi) + xk*(yi-yj))
                if det > 0:
                    areas.append(det)
    return areas

def _score(points):
    """Return the maximum multiplicity of any single positive doubled area."""
    areas = _doubled_areas(points)
    if not areas:
        return 0
    return Counter(areas).most_common(1)[0][1]

def _fast_score(points):
    """Incremental-friendly scoring: compute all areas and return max multiplicity."""
    n = len(points)
    area_counts = {}
    for i in range(n):
        xi, yi = points[i]
        for j in range(i+1, n):
            xj, yj = points[j]
            for k in range(j+1, n):
                xk, yk = points[k]
                det = abs(xi*(yj-yk) + xj*(yk-yi) + xk*(yi-yj))
                if det > 0:
                    area_counts[det] = area_counts.get(det, 0) + 1
    if not area_counts:
        return 0
    return max(area_counts.values())

def _delta_score(points, old_pt, new_pt, old_idx):
    """
    Compute the change in max multiplicity when replacing points[old_idx] with new_pt.
    Returns (new_score, area_counts_after) for efficiency in search.
    """
    n = len(points)
    # Compute areas NOT involving old_idx (these stay the same)
    # Areas involving old_idx need recomputation
    area_counts = {}
    # First, count all areas not involving old_idx
    indices = [i for i in range(n) if i != old_idx]
    for a in range(len(indices)):
        ia = indices[a]
        xa, ya = points[ia]
        for b in range(a+1, len(indices)):
            ib = indices[b]
            xb, yb = points[ib]
            for c in range(b+1, len(indices)):
                ic = indices[c]
                xc, yc = points[ic]
                det = abs(xa*(yb-yc) + xb*(yc-ya) + xc*(ya-yb))
                if det > 0:
                    area_counts[det] = area_counts.get(det, 0) + 1
    # Now add areas involving old_idx (with new point)
    nx, ny = new_pt
    for j in range(n):
        if j == old_idx:
            continue
        xj, yj = points[j]
        for k in range(j+1, n):
            if k == old_idx:
                continue
            xk, yk = points[k]
            det = abs(nx*(yj-yk) + xj*(yk-ny) + xk*(ny-yj))
            if det > 0:
                area_counts[det] = area_counts.get(det, 0) + 1
    if not area_counts:
        return 0, area_counts
    return max(area_counts.values()), area_counts

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s to be safe within 90s limit
    rng = random.Random(42)
    
    # Start with a reasonable initial configuration
    # Try a grid-like structure but with some perturbation
    best_points = [(x, y) for x in range(6) for y in range(5)]
    best_score = _score(best_points)
    
    # Local search: try moving individual points
    max_coord = 30
    min_coord = -30
    
    while time.time() < deadline:
        improved = False
        # Try random moves
        for _ in range(200):
            if time.time() > deadline:
                break
            
            idx = rng.randint(0, 29)
            old_pt = best_points[idx]
            
            # Random new position
            new_x = rng.randint(min_coord, max_coord)
            new_y = rng.randint(min_coord, max_coord)
            new_pt = (new_x, new_y)
            
            if new_pt == old_pt:
                continue
            
            # Check distinctness
            if any(best_points[i] == new_pt for i in range(30) if i != idx):
                continue
            
            # Compute new score
            new_score, _ = _delta_score(best_points, old_pt, new_pt, idx)
            
            if new_score > best_score:
                best_points[idx] = new_pt
                best_score = new_score
                improved = True
                break
            elif new_score == best_score and rng.random() < 0.3:
                # Accept equal score moves with some probability to escape local optima
                best_points[idx] = new_pt
                improved = True
                break
        
        if improved:
            continue
        
        # If no improvement, try a swap move
        for _ in range(50):
            if time.time() > deadline:
                break
            
            idx1 = rng.randint(0, 29)
            idx2 = rng.randint(0, 29)
            if idx1 == idx2:
                continue
            
            # Try swapping with a random new point for idx1
            new_x = rng.randint(min_coord, max_coord)
            new_y = rng.randint(min_coord, max_coord)
            new_pt = (new_x, new_y)
            
            if new_pt == best_points[idx1] or new_pt == best_points[idx2]:
                continue
            if any(best_points[i] == new_pt for i in range(30) if i not in (idx1, idx2)):
                continue
            
            # Temporarily set and compute
            old_pt1 = best_points[idx1]
            best_points[idx1] = new_pt
            new_score, _ = _score(best_points)
            
            if new_score > best_score:
                best_score = new_score
                improved = True
                break
            elif new_score == best_score and rng.random() < 0.2:
                improved = True
                break
            else:
                best_points[idx1] = old_pt1
        
        if improved:
            continue
        
        # If still no improvement, try a larger perturbation
        for _ in range(20):
            if time.time() > deadline:
                break
            
            # Move 2-3 random points
            num_moves = rng.randint(2, 3)
            moved_indices = rng.sample(range(30), num_moves)
            old_pts = [best_points[i] for i in moved_indices]
            
            for i in moved_indices:
                new_x = rng.randint(min_coord, max_coord)
                new_y = rng.randint(min_coord, max_coord)
                new_pt = (new_x, new_y)
                if any(best_points[j] == new_pt for j in range(30) if j != i and j not in moved_indices):
                    # Try a few more times
                    for _ in range(5):
                        new_x = rng.randint(min_coord, max_coord)
                        new_y = rng.randint(min_coord, max_coord)
                        new_pt = (new_x, new_y)
                        if not any(best_points[j] == new_pt for j in range(30) if j != i and j not in moved_indices):
                            break
                    if any(best_points[j] == new_pt for j in range(30) if j != i and j not in moved_indices):
                        break
                best_points[i] = new_pt
            else:
                new_score = _score(best_points)
                if new_score > best_score:
                    best_score = new_score
                    improved = True
                    break
                elif new_score == best_score and rng.random() < 0.1:
                    improved = True
                    break
                else:
                    # Revert
                    for i, op in zip(moved_indices, old_pts):
                        best_points[i] = op
        
        if improved:
            continue
        
        # If stuck for a while, try restarting with a different seed
        # But keep the best found so far
        # For now, just continue with smaller steps
    
    return best_points
# EVOLVE-BLOCK-END
