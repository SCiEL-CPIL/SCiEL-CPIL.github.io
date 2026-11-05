import time
from collections import Counter
import random
import numpy as np

# EVOLVE-BLOCK-START

def _doubled_areas(points):
    """Compute doubled areas (abs determinant) for all triples of points."""
    n = len(points)
    areas = []
    for i in range(n):
        xi, yi = points[i]
        for j in range(i+1, n):
            xj, yj = points[j]
            for k in range(j+1, n):
                xk, yk = points[k]
                d = (xj - xi) * (yk - yi) - (yj - yi) * (xk - xi)
                d = abs(d)
                if d > 0:
                    areas.append(d)
    return areas

def _score(points):
    """Return the largest multiplicity of a single positive doubled area."""
    areas = _doubled_areas(points)
    if not areas:
        return 0
    c = Counter(areas)
    return max(c.values())

def _initial_grid():
    # 5x6 grid centered roughly at 0
    # x in -2..3, y in -2..2
    return [(x, y) for x in range(-2, 4) for y in range(-2, 3)]

def _local_search_incremental(points, time_limit):
    """
    Local search with incremental area histogram updates.
    This is O(N^2) per move evaluation instead of O(N^3).
    """
    start_time = time.time()
    pts = list(points)
    n = len(pts)
    
    # Initial histogram
    areas = _doubled_areas(pts)
    if not areas:
        return pts, 0
    hist = Counter(areas)
    best_score = max(hist.values())
    
    # Precompute list of all valid points
    all_points = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
    
    # To speed up, we can keep a set of occupied points for O(1) lookup
    occupied = set(pts)
    
    no_improve_count = 0
    max_no_improve = 200
    
    while time.time() - start_time < time_limit:
        # Pick a random point index
        i = random.randint(0, n-1)
        old_pt = pts[i]
        
        # Generate candidate new point
        # We try a few candidates and pick the best one
        best_candidate = None
        best_candidate_score = best_score
        best_candidate_hist = None
        
        # Try 3 random candidates
        for _ in range(3):
            new_pt = random.choice(all_points)
            if new_pt in occupied:
                continue
            
            # Calculate delta in histogram
            # Triples involving i: (i, j, k) where j, k != i
            # We iterate over all pairs (j, k) from the other n-1 points
            
            # To avoid modifying the main hist, we calculate the changes
            # and apply them to a copy? No, that's O(4000) which is similar to full recalc.
            # But we can do it in place and revert.
            
            # However, we need to know the NEW MAX.
            # If we update hist, we can check max(hist).
            
            # Let's update in place.
            changes = []
            
            # Get indices of other points
            # We can just iterate j, k from 0 to n-1, skipping i
            
            # For speed, let's precompute the other points list?
            # No, just loop.
            
            # Optimization:
            # We can precompute the vector from i to all other points?
            # Let's just do the loops.
            
            for j in range(n):
                if j == i: continue
                xj, yj = pts[j]
                for k in range(j+1, n):
                    if k == i: continue
                    xk, yk = pts[k]
                    
                    # Old area
                    old_d = abs((xj - old_pt[0]) * (yk - old_pt[1]) - (yj - old_pt[1]) * (xk - old_pt[0]))
                    # New area
                    new_d = abs((xj - new_pt[0]) * (yk - new_pt[1]) - (yj - new_pt[1]) * (xk - new_pt[0]))
                    
                    if old_d == new_d:
                        continue
                    
                    if old_d > 0:
                        hist[old_d] -= 1
                        changes.append((old_d, 1))
                        if hist[old_d] == 0:
                            del hist[old_d]
                    
                    if new_d > 0:
                        hist[new_d] = hist.get(new_d, 0) + 1
                        changes.append((new_d, -1))
            
            # Check new score
            new_score = max(hist.values()) if hist else 0
            
            if new_score > best_candidate_score:
                best_candidate_score = new_score
                best_candidate = new_pt
                # We need to save the state of hist?
                # If we reject, we need to revert.
                # If we accept, we keep the changes.
                # This is tricky if we try multiple candidates.
                # Simplest: Only try one candidate per iteration?
                # Or revert after each candidate check.
                
                # Let's revert for now to keep it simple and safe.
                # Revert changes
                for area, delta in changes:
                    if delta == 1: # We removed 1, so add 1 back
                        hist[area] = hist.get(area, 0) + 1
                    else: # We added 1, so remove 1
                        hist[area] -= 1
                        if hist[area] == 0:
                            del hist[area]
                
                # If we found a better candidate, we should apply it immediately?
                # Or save the best one and apply at the end of the candidate loop?
                # If we save, we need to re-apply the changes for the best one.
                # That's expensive.
                
                # Better strategy:
                # Try candidates one by one.
                # If a candidate improves, apply it and break the candidate loop.
                # If not, revert and try next.
                
                # But I already reverted.
                # So I need to re-apply if this is the best so far?
                # No, just accept the first improvement found.
                
                # Let's change the logic:
                # Try candidate.
                # If it improves, accept it (keep changes) and break.
                # If not, revert.
                
                # So, if new_score > best_score (current global best):
                #    Accept.
                
                if new_score > best_score:
                    # Re-apply changes (since I reverted above)
                    for j in range(n):
                        if j == i: continue
                        xj, yj = pts[j]
                        for k in range(j+1, n):
                            if k == i: continue
                            xk, yk = pts[k]
                            old_d = abs((xj - old_pt[0]) * (yk - old_pt[1]) - (yj - old_pt[1]) * (xk - old_pt[0]))
                            new_d = abs((xj - new_pt[0]) * (yk - new_pt[1]) - (yj - new_pt[1]) * (xk - new_pt[0]))
                            if old_d == new_d: continue
                            if old_d > 0:
                                hist[old_d] -= 1
                                if hist[old_d] == 0: del hist[old_d]
                            if new_d > 0:
                                hist[new_d] = hist.get(new_d, 0) + 1
                    
                    # Update state
                    pts[i] = new_pt
                    occupied.remove(old_pt)
                    occupied.add(new_pt)
                    best_score = new_score
                    no_improve_count = 0
                    break
                else:
                    # Already reverted above
                    pass
        
        else:
            # No improvement found in this iteration
            no_improve_count += 1
            
            if no_improve_count > max_no_improve:
                # Random kick: move a random point to a random location
                # Even if it worsens the score?
                # Or just restart local search?
                # Let's do a random kick that might worsen score but changes landscape.
                # We need to update hist correctly.
                
                i = random.randint(0, n-1)
                old_pt = pts[i]
                new_pt = random.choice(all_points)
                if new_pt in occupied:
                    continue
                
                # Apply move and update hist
                for j in range(n):
                    if j == i: continue
                    xj, yj = pts[j]
                    for k in range(j+1, n):
                        if k == i: continue
                        xk, yk = pts[k]
                        old_d = abs((xj - old_pt[0]) * (yk - old_pt[1]) - (yj - old_pt[1]) * (xk - old_pt[0]))
                        new_d = abs((xj - new_pt[0]) * (yk - new_pt[1]) - (yj - new_pt[1]) * (xk - new_pt[0]))
                        if old_d == new_d: continue
                        if old_d > 0:
                            hist[old_d] -= 1
                            if hist[old_d] == 0: del hist[old_d]
                        if new_d > 0:
                            hist[new_d] = hist.get(new_d, 0) + 1
                
                pts[i] = new_pt
                occupied.remove(old_pt)
                occupied.add(new_pt)
                best_score = max(hist.values()) if hist else 0
                no_improve_count = 0

    return pts, best_score

def construct():
    deadline = time.time() + 55
    
    # Start with grid
    pts = _initial_grid()
    
    # Run local search
    # We split the time into chunks to allow restarts if stuck?
    # Or just run one long search with kicks.
    
    # Let's run one long search with the incremental method.
    pts, score = _local_search_incremental(pts, time_limit=50)
    
    # If we have time, try another restart?
    # The local search has kicks, so it should explore.
    # But maybe a fresh start from a different grid is better?
    
    # For now, return the result.
    return pts

# EVOLVE-BLOCK-END
