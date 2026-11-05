import time
import random
from collections import Counter

def construct():
    deadline = time.time() + 50.0
    random.seed(12345)
    
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
    
    # Strategy: Start with a dense grid and use local search to maximize multiplicity.
    # The current program already does this but we can improve the initialization and search.
    
    # Initial grid: 6x5
    coords = [(x, y) for x in range(6) for y in range(5)]
    
    counter, best_area, best_count = full_count(coords)
    best_overall = best_count
    best_points = list(coords)
    best_target = best_area
    
    target = best_target
    current_count = best_count
    
    # Precompute pairs excluding idx
    def get_pairs_excluding(idx):
        pairs = []
        for j in range(N):
            if j == idx: continue
            for k in range(j+1, N):
                if k == idx: continue
                pairs.append((j, k))
        return pairs
    
    all_pairs = [get_pairs_excluding(i) for i in range(N)]
    
    no_improve = 0
    max_no_improve = 200
    iterations = 0
    
    # Try multiple restarts with different initial grids
    restarts = 0
    max_restarts = 3
    
    while time.time() < deadline:
        if no_improve > max_no_improve:
            # Re-evaluate
            counter, ba, bc = full_count(coords)
            if bc > best_overall:
                best_overall = bc
                best_points = list(coords)
                best_target = ba
            
            # Switch target
            target = ba
            current_count = bc
            no_improve = 0
            max_no_improve += 100
            
            # If stuck, try a random restart
            restarts += 1
            if restarts > max_restarts:
                # Random restart
                coords = [(random.randint(-10, 10), random.randint(-10, 10)) for _ in range(N)]
                # Ensure distinct
                seen = set()
                for i in range(N):
                    while coords[i] in seen:
                        coords[i] = (random.randint(-10, 10), random.randint(-10, 10))
                    seen.add(coords[i])
                restarts = 0
                no_improve = 0
                continue
        
        # Pick a random point to move
        idx = random.randint(0, N-1)
        old_pt = coords[idx]
        pairs = all_pairs[idx]
        
        improved = False
        
        # Try several random positions
        for _ in range(10):
            new_pt = (random.randint(-30, 30), random.randint(-30, 30))
            if new_pt == old_pt:
                continue
            if new_pt in coords:
                continue
            
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
            
            if delta > 0:
                current_count += delta
                coords[idx] = new_pt
                improved = True
                no_improve = 0
                if current_count > best_overall:
                    best_overall = current_count
                    best_points = list(coords)
                break
            elif delta == 0 and random.random() < 0.03:
                coords[idx] = new_pt
                no_improve += 1
                break
        
        if not improved:
            no_improve += 1
        
        iterations += 1
    
    # Final verification
    if len(best_points) != N or len(set(best_points)) != N:
        return [(x,y) for x in range(6) for y in range(5)]
    for p in best_points:
        if not (-30 <= p[0] <= 30 and -30 <= p[1] <= 30):
            return [(x,y) for x in range(6) for y in range(5)]
            
    return best_points
