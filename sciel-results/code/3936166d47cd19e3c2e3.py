import time
import random
from collections import Counter

def construct():
    deadline = time.time() + 80.0  # safety margin before 90s kill
    random.seed(42)
    
    N = 30
    coords = [(x, y) for x in range(6) for y in range(5)]
    
    def doubled_area(p1, p2, p3):
        return abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
    
    def full_count(points):
        """Compute all non-zero doubled areas and return (counter, max_area, max_count)."""
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
    
    def count_for_target(points, target):
        """Count how many triples have doubled area == target."""
        cnt = 0
        for i in range(N):
            for j in range(i+1, N):
                for k in range(j+1, N):
                    if doubled_area(points[i], points[j], points[k]) == target:
                        cnt += 1
        return cnt
    
    def incremental_delta(points, idx, new_pt, target):
        """
        Compute change in count of target area when moving point idx to new_pt.
        Returns (delta, new_count) where new_count = old_count + delta.
        We need to know old_count; instead compute new_count directly for triples involving idx.
        """
        # Triples involving idx: (idx, a, b) for a < b, a != idx, b != idx
        # Old contribution: count of target among triples with idx
        # New contribution: count of target among triples with new_pt at idx
        old_cnt = 0
        new_cnt = 0
        for a in range(N):
            if a == idx:
                continue
            for b in range(a+1, N):
                if b == idx:
                    continue
                # triple is (idx, a, b) in some order
                old_cnt += 1 if doubled_area(points[idx], points[a], points[b]) == target else 0
                new_cnt += 1 if doubled_area(new_pt, points[a], points[b]) == target else 0
        return new_cnt - old_cnt, new_cnt
    
    # Initial evaluation
    counter, best_area, best_count = full_count(coords)
    best_overall = best_count
    best_points = list(coords)
    best_target = best_area
    
    # Local search
    # We'll alternate between optimizing for the current best area and trying new areas
    target = best_target
    current_count = best_count
    
    # Precompute: for efficiency, we'll track which triples have which area
    # But for simplicity, use incremental delta
    
    iterations = 0
    no_improve = 0
    max_no_improve = 500
    
    while time.time() < deadline:
        if no_improve > max_no_improve:
            # Try a different target area
            # Re-evaluate to find current best area
            counter, best_area, best_count = full_count(coords)
            if best_count > best_overall:
                best_overall = best_count
                best_points = list(coords)
                best_target = best_area
            target = best_area
            current_count = best_count
            no_improve = 0
            max_no_improve += 200  # be more patient
            continue
        
        # Pick a random point to move
        idx = random.randint(0, N-1)
        old_pt = coords[idx]
        
        # Try random new positions
        improved = False
        for _ in range(20):  # try 20 random positions per point
            new_pt = (random.randint(-30, 30), random.randint(-30, 30))
            if new_pt == old_pt:
                continue
            # Check no duplicate
            if new_pt in coords:
                continue
            
            # Incremental delta
            delta, new_cnt = incremental_delta(coords, idx, new_pt, target)
            
            if new_cnt > current_count:
                # Accept
                coords[idx] = new_pt
                current_count = new_cnt
                improved = True
                no_improve = 0
                if current_count > best_overall:
                    best_overall = current_count
                    best_points = list(coords)
                    best_target = target
                break
            elif delta == 0 and random.random() < 0.1:
                # Accept neutral move with some probability (simulated annealing-like)
                coords[idx] = new_pt
                # current_count unchanged
                no_improve += 1
                break
        
        if not improved:
            no_improve += 1
        
        iterations += 1
        
        # Periodically re-evaluate to check if a different target is better
        if iterations % 200 == 0:
            counter, ba, bc = full_count(coords)
            if bc > best_overall:
                best_overall = bc
                best_points = list(coords)
                best_target = ba
            if ba != target:
                # Switch target if it's better
                if bc > current_count:
                    target = ba
                    current_count = bc
                    no_improve = 0
    
    # Final: return best_points
    # Verify
    if len(best_points) != N or len(set(best_points)) != N:
        return [(x,y) for x in range(6) for y in range(5)]
    for p in best_points:
        if not (-30 <= p[0] <= 30 and -30 <= p[1] <= 30):
            return [(x,y) for x in range(6) for y in range(5)]
    
    return best_points
