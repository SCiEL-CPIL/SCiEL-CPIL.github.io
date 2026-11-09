import math
import time

_NEIGH = [(1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1)]

def _dist_sq(a1, b1, a2, b2):
    da, db = a1 - a2, b1 - b2
    return da * da + da * db + db * db

def _hexdist(a,b):
    return (abs(a)+abs(b)+abs(a+b))//2

def _count_edges(pts):
    s = set(pts)
    cnt = 0
    for (a,b) in s:
        for da,db in _NEIGH:
            if (a+da, b+db) in s:
                cnt += 1
    return cnt // 2

def _greedy_build(n=32, max_coord=8):
    """Greedy construction: build a dense cluster minimizing perimeter growth."""
    placed = [(0, 0)]
    placed_set = {(0, 0)}
    
    # Generate candidates in a bounding box
    candidates = []
    for a in range(-max_coord, max_coord + 1):
        for b in range(-max_coord, max_coord + 1):
            if (a, b) not in placed_set:
                candidates.append((a, b))
    
    for _ in range(n - 1):
        best_pt = None
        best_score = None  # (new_edges, -perimeter_growth, -dist_to_centroid)
        
        # Compute centroid
        cx = sum(p[0] for p in placed) / len(placed)
        cy = sum(p[1] for p in placed) / len(placed)
        
        # Compute current perimeter (approximate: number of boundary edges)
        # Actually, let's use a simpler metric: prefer points that maximize new edges
        # and minimize distance from centroid
        
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
            
            # Distance from centroid (approximate lattice distance)
            dx = pt[0] - cx
            dy = pt[1] - cy
            dist_to_center = abs(dx) + abs(dy) + abs(dx + dy)
            
            # Score: primarily new edges, secondarily closeness to centroid
            # This minimizes perimeter growth
            score = (new_edges, -dist_to_center)
            
            if best_score is None or score > best_score:
                best_score = score
                best_pt = pt
        
        if best_pt is None:
            break
        
        placed.append(best_pt)
        placed_set.add(best_pt)
        # Remove from candidates for efficiency
        # (we'll just check membership in placed_set)
    
    return placed

def _local_improve(points, candidates, deadline):
    """Local improvement: try swapping points to increase edge count."""
    best = list(points)
    best_set = set(best)
    current_edges = _count_edges(best)
    
    improved = True
    iteration = 0
    max_iterations = 100
    
    while improved and iteration < max_iterations:
        if time.time() > deadline:
            break
        improved = False
        iteration += 1
        
        best_delta = 0
        best_swap_i = -1
        best_swap_new = None
        
        for i in range(len(best)):
            if time.time() > deadline:
                break
            old_pt = best[i]
            
            for new_pt in candidates:
                if new_pt == old_pt or new_pt in best_set:
                    continue
                
                # Check minimum distance constraint: dist >= 1 from all other placed points
                valid = True
                for j in range(len(best)):
                    if i == j:
                        continue
                    if _dist_sq(new_pt[0], new_pt[1], best[j][0], best[j][1]) < 1:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Calculate edge change
                edges_lost = 0
                for da, db in _NEIGH:
                    if (old_pt[0]+da, old_pt[1]+db) in best_set:
                        edges_lost += 1
                        
                edges_gained = 0
                for da, db in _NEIGH:
                    nb = (new_pt[0]+da, new_pt[1]+db)
                    if nb in best_set and nb != old_pt:
                        edges_gained += 1
                        
                delta = edges_gained - edges_lost
                if delta > best_delta:
                    best_delta = delta
                    best_swap_i = i
                    best_swap_new = new_pt
                    
        if best_delta > 0:
            # Perform the best swap found
            old_pt = best[best_swap_i]
            best[best_swap_i] = best_swap_new
            best_set.remove(old_pt)
            best_set.add(best_swap_new)
            current_edges += best_delta
            improved = True
    
    return best, current_edges

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    n = 32
    
    # Generate candidate pool
    max_coord = 8
    all_candidates = []
    for a in range(-max_coord, max_coord + 1):
        for b in range(-max_coord, max_coord + 1):
            all_candidates.append((a, b))
    
    # Try multiple greedy constructions with different starting points
    best_points = None
    best_count = -1
    
    # Start from origin and a few other central points
    start_points = [(0, 0), (1, 0), (0, 1), (1, -1)]
    
    for start_pt in start_points:
        if time.time() > deadline - 10:
            break
        
        # Build greedily from this start
        placed = [start_pt]
        placed_set = {start_pt}
        
        for _ in range(n - 1):
            best_pt = None
            best_score = None
            
            # Compute centroid
            cx = sum(p[0] for p in placed) / len(placed)
            cy = sum(p[1] for p in placed) / len(placed)
            
            for pt in all_candidates:
                if pt in placed_set:
                    continue
                
                # Check minimum distance constraint
                valid = True
                for pa, pb in placed:
                    if _dist_sq(pt[0], pt[1], pa, pb) < 1:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Count new edges
                new_edges = 0
                for pa, pb in placed:
                    if _dist_sq(pt[0], pt[1], pa, pb) == 1:
                        new_edges += 1
                
                # Distance from centroid
                dx = pt[0] - cx
                dy = pt[1] - cy
                dist_to_center = abs(dx) + abs(dy) + abs(dx + dy)
                
                score = (new_edges, -dist_to_center)
                
                if best_score is None or score > best_score:
                    best_score = score
                    best_pt = pt
            
            if best_pt is None:
                break
            
            placed.append(best_pt)
            placed_set.add(best_pt)
        
        if len(placed) < n:
            continue
        
        count = _count_edges(placed)
        if count > best_count:
            best_count = count
            best_points = list(placed)
    
    if best_points is None:
        # Fallback: use spiral
        best_points = _greedy_build(n)
        best_count = _count_edges(best_points)
    
    # Local improvement
    if time.time() < deadline - 5:
        best_points, best_count = _local_improve(best_points, all_candidates, deadline)
    
    # Final verification
    if len(best_points) != n:
        # Regenerate if something went wrong
        best_points = _greedy_build(n)
    
    return [[a, b] for (a, b) in best_points]
