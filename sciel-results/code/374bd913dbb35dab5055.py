"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random

def _dist_sq(a1, b1, a2, b2):
    da, db = a1 - a2, b1 - b2
    return da * da + da * db + db * db

def _count_edges(points):
    """Count pairs at distance exactly 1 in the triangular lattice."""
    pts = set(points)
    count = 0
    for a, b in pts:
        for da, db in [(1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]:
            if (a+da, b+db) in pts:
                count += 1
    return count // 2

def _generate_candidates(max_coord=20):
    """Generate candidate points in a bounding box."""
    candidates = []
    for a in range(-max_coord, max_coord + 1):
        for b in range(-max_coord, max_coord + 1):
            candidates.append((a, b))
    return candidates

def _greedy_construct(candidates, n_points=32, seed=42):
    """Greedy construction: add points one at a time, maximizing new edges."""
    random.seed(seed)
    
    # Start with a random point near origin
    placed = []
    placed_set = set()
    
    # First point: origin
    first = (0, 0)
    placed.append(first)
    placed_set.add(first)
    
    for _ in range(n_points - 1):
        best_point = None
        best_new_edges = -1
        best_score = -1
        
        for pt in candidates:
            if pt in placed_set:
                continue
            
            # Check minimum distance constraint: dist >= 1 from all placed points
            valid = True
            for pa, pb in placed:
                if _dist_sq(pt[0], pt[1], pa, pb) < 1:
                    valid = False
                    break
            if not valid:
                continue
            
            # Count new edges formed with already-placed points
            new_edges = 0
            for pa, pb in placed:
                if _dist_sq(pt[0], pt[1], pa, pb) == 1:
                    new_edges += 1
            
            # Score: primarily new edges, secondary: prefer points closer to cluster center
            # Compute distance from centroid of placed points
            if placed:
                cx = sum(p[0] for p in placed) / len(placed)
                cy = sum(p[1] for p in placed) / len(placed)
                # Approximate distance from centroid
                dx = pt[0] - cx
                dy = pt[1] - cy
                # Use lattice distance approximation
                dist_to_center = abs(dx) + abs(dy) + abs(dx + dy)
            else:
                dist_to_center = 0
            
            score = (new_edges, -dist_to_center)
            
            if score > (best_new_edges, -best_score):
                best_new_edges = new_edges
                best_score = dist_to_center
                best_point = pt
        
        if best_point is None:
            # No valid point found, break
            break
        
        placed.append(best_point)
        placed_set.add(best_point)
    
    return placed

def _local_improve(points, candidates, max_iterations=500):
    """Local improvement: try swapping points to increase edge count."""
    best_points = list(points)
    best_set = set(best_points)
    best_count = _count_edges(best_points)
    
    # Precompute candidate list excluding current points
    candidate_list = [c for c in candidates if c not in best_set]
    
    improved = True
    iteration = 0
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        for i in range(len(best_points)):
            old_pt = best_points[i]
            
            for new_pt in candidate_list:
                if new_pt == old_pt or new_pt in best_set:
                    continue
                
                # Check minimum distance constraint
                valid = True
                for j in range(len(best_points)):
                    if i == j:
                        continue
                    if _dist_sq(new_pt[0], new_pt[1], best_points[j][0], best_points[j][1]) < 1:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Compute new edge count efficiently
                # Remove old_pt, add new_pt
                new_set = best_set - {old_pt} | {new_pt}
                
                # Count edges: for each point in new_set, count its neighbors in new_set
                new_count = 0
                for a, b in new_set:
                    for da, db in [(1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]:
                        if (a+da, b+db) in new_set:
                            new_count += 1
                new_count //= 2
                
                if new_count > best_count:
                    best_points[i] = new_pt
                    best_set = new_set
                    best_count = new_count
                    # Update candidate list
                    candidate_list = [c for c in candidates if c not in best_set]
                    improved = True
                    break
            
            if improved:
                break
    
    return best_points, best_count

def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    
    candidates = _generate_candidates(max_coord=15)
    
    # Try multiple random seeds for greedy construction
    best_points = None
    best_count = -1
    
    for seed in range(20):
        if time.time() > deadline:
            break
        
        points = _greedy_construct(candidates, n_points=32, seed=seed)
        if len(points) < 32:
            continue
        
        count = _count_edges(points)
        if count > best_count:
            best_count = count
            best_points = list(points)
    
    if best_points is None:
        # Fallback: just take first 32 from a simple spiral
        points = _greedy_construct(candidates, n_points=32, seed=0)
        return [[a, b] for a, b in points]
    
    # Local improvement
    if time.time() < deadline - 10:
        best_points, best_count = _local_improve(best_points, candidates, max_iterations=200)
    
    return [[a, b] for a, b in best_points]
# EVOLVE-BLOCK-END
