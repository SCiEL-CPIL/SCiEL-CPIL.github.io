import time
import random
from collections import Counter
import numpy as np

def construct():
    deadline = time.time() + 55.0  # safety margin under 60s
    random.seed(42)
    
    N = 30
    
    def doubled_area(p1, p2, p3):
        return abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
    
    def full_count(points):
        areas = []
        for i in range(N):
            for j in range(i+1, N):
                for k in range(j+1, N):
                    a = doubled_area(points[i], points[j], points[k])
                    if a > 0:
                        areas.append(a)
        if not areas:
            return Counter(), 0, 0
        c = Counter(areas)
        max_area, max_count = c.most_common(1)[0]
        return c, max_area, max_count

    def get_pairs_excluding(idx):
        pairs = []
        for j in range(N):
            if j == idx: continue
            for k in range(j+1, N):
                if k == idx: continue
                pairs.append((j, k))
        return pairs

    all_pairs = [get_pairs_excluding(i) for i in range(N)]

    # Initial grid
    coords = [(x, y) for x in range(6) for y in range(5)]
    counter, best_area, best_count = full_count(coords)
    best_overall = best_count
    best_points = list(coords)
    best_target = best_area
    target = best_target
    current_count = best_count

    no_improve = 0
    max_no_improve = 100
    iterations = 0
    
    # Strategy:
    # 1. Fast local search using delta evaluation for a fixed target area.
    # 2. Periodically re-evaluate the full configuration to find the true best area/count.
    # 3. If stuck, switch target to the current best area found in the full re-eval.
    # 4. Occasional random restarts or larger jumps if completely stuck.
    
    while time.time() < deadline:
        if no_improve > max_no_improve:
            # Re-evaluate full config
            counter, ba, bc = full_count(coords)
            if bc > best_overall:
                best_overall = bc
                best_points = list(coords)
                best_target = ba
            
            # Switch target to the most frequent area in the current config
            target = ba
            current_count = bc
            no_improve = 0
            max_no_improve += 50 # Increase patience slightly
            
            # If we've been stuck for a long time (many restarts of target), try a random perturbation
            if max_no_improve > 500:
                # Random restart with a few points moved significantly
                for _ in range(random.randint(3, 8)):
                    idx = random.randint(0, N-1)
                    new_pt = (random.randint(-30, 30), random.randint(-30, 30))
                    if new_pt not in coords:
                        coords[idx] = new_pt
                # Re-evaluate after perturbation
                counter, ba, bc = full_count(coords)
                if bc > best_overall:
                    best_overall = bc
                    best_points = list(coords)
                target = ba
                current_count = bc
                no_improve = 0
                max_no_improve = 100
            continue
        
        # Pick a random point to move
        idx = random.randint(0, N-1)
        old_pt = coords[idx]
        pairs = all_pairs[idx]
        
        improved = False
        # Try a few random positions, plus some local ones
        candidates = []
        # Local moves
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                if dx == 0 and dy == 0: continue
                nx, ny = old_pt[0] + dx, old_pt[1] + dy
                if -30 <= nx <= 30 and -30 <= ny <= 30:
                    candidates.append((nx, ny))
        # Random distant moves
        for _ in range(3):
            candidates.append((random.randint(-30, 30), random.randint(-30, 30)))
        
        random.shuffle(candidates)
        
        best_delta = -1
        best_new_pt = None
        
        for new_pt in candidates:
            if new_pt == old_pt:
                continue
            if new_pt in coords:
                continue
            
            # Calculate delta for the current target area
            old_matches = 0
            new_matches = 0
            for j, k in pairs:
                p_j = coords[j]
                p_k = coords[k]
                if doubled_area(old_pt, p_j, p_k) == target:
                    old_matches += 1
                if doubled_area(new_pt, p_j, p_k) == target:
                    new_matches += 1
            
            delta = new_matches - old_matches
            
            if delta > best_delta:
                best_delta = delta
                best_new_pt = new_pt
        
        if best_new_pt is not None and (best_delta > 0 or (best_delta == 0 and random.random() < 0.1)):
            # Accept move
            # Update current_count estimate (approximate, will be corrected on full re-eval)
            current_count += best_delta
            coords[idx] = best_new_pt
            improved = True
            no_improve = 0
            if current_count > best_overall:
                best_overall = current_count
                best_points = list(coords)
        else:
            no_improve += 1
            
        iterations += 1
        
        # Periodic full re-evaluation to correct drift in current_count and find new targets
        if iterations % 20 == 0:
             counter, ba, bc = full_count(coords)
             if bc > best_overall:
                 best_overall = bc
                 best_points = list(coords)
             # Sync current state
             target = ba
             current_count = bc
             # If the target changed significantly or count dropped, reset no_improve counter logic slightly?
             # Actually, if we just re-evaluated, our state is fresh.
             # If no_improve was high, the block above handles it.

    # Final verification
    if len(best_points) != N or len(set(best_points)) != N:
        return [(x,y) for x in range(6) for y in range(5)]
    for p in best_points:
        if not (-30 <= p[0] <= 30 and -30 <= p[1] <= 30):
            return [(x,y) for x in range(6) for y in range(5)]
            
    return best_points
