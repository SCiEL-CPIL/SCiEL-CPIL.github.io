import random
import time

def construct():
    start_time = time.time()
    timeout = 85.0  # Safety margin
    
    n_points = 30
    bounds = 30
    
    def get_initial_points():
        pts = set()
        while len(pts) < n_points:
            x = random.randint(-bounds, bounds)
            y = random.randint(-bounds, bounds)
            pts.add((x, y))
        return list(pts)

    def compute_areas(points):
        counts = {}
        for i in range(n_points):
            pi = points[i]
            for j in range(i+1, n_points):
                pj = points[j]
                for k in range(j+1, n_points):
                    pk = points[k]
                    area = abs(pi[0]*(pj[1]-pk[1]) + pj[0]*(pk[1]-pi[1]) + pk[0]*(pi[1]-pj[1]))
                    if area > 0:
                        counts[area] = counts.get(area, 0) + 1
        return counts

    def max_count(counts):
        return max(counts.values()) if counts else 0

    # Initial state
    points = get_initial_points()
    area_counts = compute_areas(points)
    best_count = max_count(area_counts)
    best_points = points.copy()
    
    # Precompute pairs for each index to speed up updates
    # pairs_for[i] = list of (j, k) such that j, k != i and j < k
    pairs_for = [[] for _ in range(n_points)]
    for i in range(n_points):
        for j in range(n_points):
            if j == i: continue
            for k in range(j+1, n_points):
                if k == i: continue
                pairs_for[i].append((j, k))
                
    # Note: The above precomputation is O(N^3) which is fine.
    # Actually, simpler to just loop in the update step. 406 iterations is nothing.
    
    no_improve_steps = 0
    max_no_improve = 5000 # Reset if stuck
    
    while time.time() - start_time < timeout:
        # Pick random index
        i = random.randint(0, n_points - 1)
        old_pi = points[i]
        
        # Pick random new point
        new_pi = None
        for _ in range(100): # Try to find a distinct point
            x = random.randint(-bounds, bounds)
            y = random.randint(-bounds, bounds)
            cand = (x, y)
            if cand != old_pi and cand not in points:
                new_pi = cand
                break
        if new_pi is None:
            continue # Grid too full? Unlikely with 30 points in 37x37=1369 cells
            
        # Update area counts
        # We need to decrement old areas and increment new areas for triples involving i
        # To avoid modifying the dict while checking max, we can just modify and recompute max
        # Since dict size is small, recomputing max is cheap.
        
        old_areas_to_dec = []
        new_areas_to_inc = []
        
        pi = points[i]
        for j in range(n_points):
            if j == i: continue
            pj = points[j]
            for k in range(j+1, n_points):
                if k == i: continue
                pk = points[k]
                
                old_area = abs(pi[0]*(pj[1]-pk[1]) + pj[0]*(pk[1]-pi[1]) + pk[0]*(pi[1]-pj[1]))
                new_area = abs(new_pi[0]*(pj[1]-pk[1]) + pj[0]*(pk[1]-new_pi[1]) + pk[0]*(new_pi[1]-pj[1]))
                
                if old_area > 0:
                    old_areas_to_dec.append(old_area)
                if new_area > 0:
                    new_areas_to_inc.append(new_area)
                    
        # Apply changes
        for a in old_areas_to_dec:
            area_counts[a] -= 1
            if area_counts[a] == 0:
                del area_counts[a]
                
        for a in new_areas_to_inc:
            area_counts[a] = area_counts.get(a, 0) + 1
            
        # Calculate new max
        # Optimization: We can track max more carefully, but max(dict.values()) is fast enough
        current_max = max(area_counts.values()) if area_counts else 0
        
        if current_max > best_count:
            best_count = current_max
            best_points = points.copy()
            no_improve_steps = 0
            # Accept move
            points[i] = new_pi
        else:
            # Reject move, revert changes
            for a in new_areas_to_inc:
                area_counts[a] -= 1
                if area_counts[a] == 0:
                    del area_counts[a]
            for a in old_areas_to_dec:
                area_counts[a] = area_counts.get(a, 0) + 1
            
            no_improve_steps += 1
            if no_improve_steps > max_no_improve:
                # Random restart
                points = get_initial_points()
                area_counts = compute_areas(points)
                current_max = max(area_counts.values()) if area_counts else 0
                if current_max > best_count:
                    best_count = current_max
                    best_points = points.copy()
                no_improve_steps = 0

    return best_points

