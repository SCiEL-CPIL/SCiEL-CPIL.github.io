import numpy as np
import time
from collections import Counter

def construct():
    start_time = time.time()
    limit = 55.0
    
    # Initial 6x5 grid
    points = [(x, y) for x in range(-2, 4) for y in range(-2, 3)]
    n = len(points)
    
    # Precompute indices for triples involving each point?
    # Or just calculate on the fly.
    
    def get_areas_involving(idx, pts):
        # Returns list of doubled areas for all triples involving idx
        # idx is 0..n-1
        # Other points are all j != idx
        # Triples are (idx, j, k) with j < k and j, k != idx
        areas = []
        pi = np.array(pts[idx])
        others = [pts[j] for j in range(n) if j != idx]
        # We need pairs from 'others'
        # Number of others is 29
        # Pairs: 29*28/2 = 406
        for j in range(len(others)):
            for k in range(j+1, len(others)):
                pj = np.array(others[j])
                pk = np.array(others[k])
                v1 = pj - pi
                v2 = pk - pi
                det = v1[0]*v2[1] - v1[1]*v2[0]
                a = abs(det)
                if a > 0:
                    areas.append(a)
        return areas

    # Initial histogram
    hist = Counter()
    for i in range(n):
        areas = get_areas_involving(i, points)
        # Each triangle (i, j, k) is counted 3 times in this loop (once for i, once for j, once for k)?
        # No, if we iterate i, and for each i we count triples (i, j, k),
        # then triangle (0, 1, 2) is counted when i=0 (as 0,1,2), i=1 (as 1,0,2 -> sorted? no, my loop doesn't sort), i=2.
        # My `get_areas_involving` generates triples (idx, j, k).
        # If I sum over all idx, I will count each triangle 3 times.
        # So I should divide the final counts by 3?
        # Or better, just build the histogram once correctly.
        
    # Let's just build the full histogram once for initialization
    def build_full_hist(pts):
        h = Counter()
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    pi = pts[i]
                    pj = pts[j]
                    pk = pts[k]
                    # det
                    det = (pj[0]-pi[0])*(pk[1]-pi[1]) - (pj[1]-pi[1])*(pk[0]-pi[0])
                    a = abs(det)
                    if a > 0:
                        h[a] += 1
        return h

    hist = build_full_hist(points)
    best_score = max(hist.values()) if hist else 0
    best_points = points[:]
    
    # Helper to update hist for a move
    def try_move(idx, new_pos):
        nonlocal hist, best_score, best_points
        
        # 1. Remove areas of triples involving idx
        old_areas = get_areas_involving(idx, best_points)
        for a in old_areas:
            hist[a] -= 1
            if hist[a] == 0:
                del hist[a]
                
        # 2. Update point
        old_pos = best_points[idx]
        best_points[idx] = new_pos
        
        # 3. Add areas of triples involving idx
        new_areas = get_areas_involving(idx, best_points)
        for a in new_areas:
            hist[a] += 1
            
        # 4. Check score
        current_score = max(hist.values()) if hist else 0
        
        if current_score > best_score:
            best_score = current_score
            return True
        else:
            # Revert
            # Remove new areas
            for a in new_areas:
                hist[a] -= 1
                if hist[a] == 0:
                    del hist[a]
            # Revert point
            best_points[idx] = old_pos
            # Add old areas
            for a in old_areas:
                hist[a] += 1
            return False

    # Hill Climbing Loop
    improved = True
    while improved and (time.time() - start_time) < limit:
        improved = False
        for idx in range(n):
            orig_x, orig_y = best_points[idx]
            # Try moves in a 3x3 or 5x5 neighborhood
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if dx == 0 and dy == 0:
                        continue
                    new_x = orig_x + dx
                    new_y = orig_y + dy
                    if new_x < -30 or new_x > 30 or new_y < -30 or new_y > 30:
                        continue
                    new_pos = (new_x, new_y)
                    if new_pos in best_points:
                        continue
                    
                    if try_move(idx, new_pos):
                        improved = True
                        # If we improved, we might want to restart the scan or just continue
                        # Continuing is fine, but the neighborhood of the new position might be different
                        # Since we modified best_points, the next iterations use the new state.
                        
            if (time.time() - start_time) > limit:
                break

    return best_points
