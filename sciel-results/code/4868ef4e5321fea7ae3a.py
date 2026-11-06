import time
import random
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Leave 5s buffer for safety
    
    # Helper to compute areas for triples involving a specific point index
    def get_areas_involving(points, idx):
        n = len(points)
        x1, y1 = points[idx]
        areas = []
        # Iterate over all pairs (j, k) from the other points
        others = [i for i in range(n) if i != idx]
        for u in range(len(others)):
            i = others[u]
            xi, yi = points[i]
            for v in range(u + 1, len(others)):
                j = others[v]
                xj, yj = points[j]
                # Determinant: (xi - x1)(yj - y1) - (yi - y1)(xj - x1)
                area = abs((xi - x1) * (yj - y1) - (yi - y1) * (xj - x1))
                if area > 0:
                    areas.append(area)
        return areas

    def compute_full_counter(points):
        areas = []
        n = len(points)
        for i in range(n):
            # We can just sum up areas involving i, but we must divide by 3? 
            # No, each triple {i,j,k} is counted 3 times if we sum for all i.
            # It's easier to just do the O(N^3) full calculation once for initialization.
            x1, y1 = points[i]
            for j in range(i + 1, n):
                x2, y2 = points[j]
                for k in range(j + 1, n):
                    x3, y3 = points[k]
                    area = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
                    if area > 0:
                        areas.append(area)
        return Counter(areas)

    # Initial configuration: 6x5 grid
    x_coords = [-10, -6, -2, 2, 6, 10]
    y_coords = [-8, -4, 0, 4, 8]
    initial_pts = [(x, y) for x in x_coords for y in y_coords]
    
    current_pts = [p for p in initial_pts]
    current_counter = compute_full_counter(current_pts)
    current_score = max(current_counter.values()) if current_counter else 0
    
    best_pts = [p for p in current_pts]
    best_score = current_score
    
    occupied = set(current_pts)
    all_coords = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
    
    # Simulated Annealing parameters
    T = 10.0
    cooling_rate = 0.999
    
    stuck_count = 0
    max_stuck = 500
    
    while time.time() < deadline:
        # Pick a random point to move
        idx = random.randint(0, 29)
        old_p = current_pts[idx]
        
        # Pick a random empty point
        new_p = None
        attempts = 0
        while attempts < 100:
            cand = random.choice(all_coords)
            if cand not in occupied:
                new_p = cand
                break
            attempts += 1
        
        if new_p is None:
            continue

        # Calculate old areas involving idx
        old_areas = get_areas_involving(current_pts, idx)
        
        # Temporarily move point to calculate new areas
        current_pts[idx] = new_p
        new_areas = get_areas_involving(current_pts, idx)
        # Restore point for now, we'll decide later
        current_pts[idx] = old_p
        
        # Update counter to simulate the move
        # We need to apply deltas to a copy or update in place and revert if rejected.
        # Updating in place and reverting is faster if we accept often, 
        # but copying a Counter of size ~1000 is also fast.
        # Let's update in place and track changes to revert.
        
        changes = []
        for a in old_areas:
            if a in current_counter:
                current_counter[a] -= 1
                if current_counter[a] <= 0:
                    del current_counter[a]
                changes.append((a, -1))
            # if a not in counter, it was 0, now -1? No, we only store positive counts.
            # If a was not in counter, it means count was 0. Decrementing makes it -1? 
            # No, get_areas_involving returns areas that exist. 
            # Wait, if area > 0, it was added to counter. So it MUST be in counter.
            # Unless the counter was built incorrectly. 
            # compute_full_counter adds all positive areas. 
            # get_areas_involving returns positive areas. 
            # So yes, they should be in the counter.
            
        for a in new_areas:
            current_counter[a] = current_counter.get(a, 0) + 1
            changes.append((a, 1))
            
        new_score = max(current_counter.values()) if current_counter else 0
        
        # Decide acceptance
        accepted = False
        if new_score > current_score:
            accepted = True
        else:
            delta = new_score - current_score
            # Simulated Annealing acceptance
            if random.random() < pow(2.71828, delta / T):
                accepted = True
        
        if accepted:
            current_pts[idx] = new_p
            occupied.remove(old_p)
            occupied.add(new_p)
            current_score = new_score
            stuck_count = 0
            
            if new_score > best_score:
                best_score = new_score
                best_pts = [p for p in current_pts]
        else:
            # Revert changes
            for a, delta_val in changes:
                if delta_val == -1:
                    # We decremented, so we need to increment back
                    current_counter[a] = current_counter.get(a, 0) + 1
                else:
                    # We incremented, so we need to decrement back
                    if a in current_counter:
                        current_counter[a] -= 1
                        if current_counter[a] <= 0:
                            del current_counter[a]
            
            stuck_count += 1
            if stuck_count > max_stuck:
                # Restart with perturbation
                # Reset to best state
                current_pts = [p for p in best_pts]
                occupied = set(current_pts)
                current_counter = compute_full_counter(current_pts)
                current_score = best_score
                stuck_count = 0
                
                # Perturb: move 2-3 random points to random spots
                for _ in range(3):
                    idx_p = random.randint(0, 29)
                    old_p_p = current_pts[idx_p]
                    while True:
                        new_p_p = random.choice(all_coords)
                        if new_p_p not in occupied:
                            break
                    # Apply move directly
                    old_areas_p = get_areas_involving(current_pts, idx_p)
                    current_pts[idx_p] = new_p_p
                    new_areas_p = get_areas_involving(current_pts, idx_p)
                    current_pts[idx_p] = old_p_p # restore for counter update logic? 
                    # Actually, just update counter
                    for a in old_areas_p:
                        if a in current_counter:
                            current_counter[a] -= 1
                            if current_counter[a] <= 0: del current_counter[a]
                    for a in new_areas_p:
                        current_counter[a] = current_counter.get(a, 0) + 1
                    
                    current_pts[idx_p] = new_p_p
                    occupied.remove(old_p_p)
                    occupied.add(new_p_p)
                
                current_score = max(current_counter.values()) if current_counter else 0
                # Note: best_score remains the same, we are just exploring from a perturbed best state
                # If we find something better, it will update best_score.
        
        # Cool down
        T *= cooling_rate
        if T < 0.01:
            T = 0.01

    return best_pts
