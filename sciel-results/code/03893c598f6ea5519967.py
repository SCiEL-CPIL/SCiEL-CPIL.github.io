import time
import random
import itertools
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Leave 15s margin for safety to avoid timeout
    
    # Precompute triple indices
    # n=30, triples = 30*29*28/6 = 4060
    n = 30
    triples = list(itertools.combinations(range(n), 3))
    triple_count = len(triples)
    
    # Precompute which triples involve which point
    # point_triples[i] is a list of (j, k, tri_idx) where tri_idx is index in triples
    # Actually, for a move of point i, we need to know the other two points.
    # Let's store for each point i, a list of (j, k) pairs.
    point_pairs = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                # This is O(N^3) precomp, which is fine.
                pass # We don't need this specific structure if we just iterate triples.
                # But iterating all triples for each move is O(4060).
                # Iterating only triples containing i is O(29*28/2) = 406.
                # Let's precompute for each i, the list of (j, k) such that {i,j,k} is a triple.
    
    # Better precomputation:
    # For each i, list of (j, k) where j,k != i
    # We can just generate them on the fly or precompute.
    # Precompute for speed.
    pairs_for_point = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j: continue
            for k in range(n):
                if i == k or j == k: continue
                # To avoid duplicates, we can just store unique pairs.
                # But for determinant calculation, order doesn't matter for absolute value.
                # Let's just store (j, k) for all j<k, j!=i, k!=i.
        # Simpler:
        pairs_for_point[i] = [(j, k) for j in range(n) for k in range(j+1, n) if j != i and k != i]
        
    # Initial configuration: 6x5 grid
    # Points: (x, y) for x in 0..5, y in 0..4
    # This fits in [-30, 30]
    points = [(x, y) for x in range(6) for y in range(5)]
    
    def calc_areas(pts):
        """Calculate doubled areas for all triples."""
        areas = [0] * triple_count
        for idx, (i, j, k) in enumerate(triples):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            x3, y3 = pts[k]
            det = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
            areas[idx] = det
        return areas

    def get_max_freq(areas):
        counts = Counter(areas)
        # Exclude zero area? The task says "exclude zero-area triples".
        # "maximize the largest multiplicity of a single positive doubled area"
        # So we ignore area 0.
        if 0 in counts:
            del counts[0]
        if not counts:
            return 0
        return max(counts.values())

    # Initial evaluation
    current_areas = calc_areas(points)
    current_max_freq = get_max_freq(current_areas)
    best_points = points[:]
    best_max_freq = current_max_freq
    
    # Local Search
    # We will try to move points to increase the max frequency.
    # Since calculating full areas is expensive, we will update incrementally.
    
    # For incremental update, we need to know the contribution of each triple to the global counter.
    # We maintain a global Counter `global_counts`.
    # When moving point i:
    # 1. For all triples containing i, decrement their area count.
    # 2. Update point i.
    # 3. For all triples containing i, calculate new area, increment count.
    # 4. Check new max.
    
    # Let's implement this.
    
    def init_counts(pts):
        counts = Counter()
        for idx, (i, j, k) in enumerate(triples):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            x3, y3 = pts[k]
            det = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
            if det > 0:
                counts[det] += 1
        return counts

    global_counts = init_counts(points)
    current_max_freq = max(global_counts.values()) if global_counts else 0
    
    # Helper to update counts for a move of point i
    # Returns the delta in max_freq? No, just update the counter.
    
    def try_move(pts, i, new_pos):
        # Check if new_pos is valid and distinct
        if new_pos in pts:
            return False, 0
        if not (-30 <= new_pos[0] <= 30 and -30 <= new_pos[1] <= 30):
            return False, 0
            
        old_pos = pts[i]
        
        # 1. Remove old contributions
        # We need to iterate over all triples containing i
        # And decrement the count of their area
        
        # To make this fast, we precomputed pairs_for_point[i] which is (j,k)
        # But we also need to know which triple index it is? No, we just need to update the global counter.
        # We don't need the triple index if we just maintain a Counter of areas.
        
        # However, we must be careful: if two triples have the same area, and we move i,
        # we are just changing the area of specific triples.
        
        # Step 1: Decrement
        for j, k in pairs_for_point[i]:
            x1, y1 = old_pos
            x2, y2 = pts[j]
            x3, y3 = pts[k]
            det = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
            if det > 0:
                global_counts[det] -= 1
                if global_counts[det] == 0:
                    del global_counts[det]
                    
        # Step 2: Update point
        pts[i] = new_pos
        
        # Step 3: Increment
        for j, k in pairs_for_point[i]:
            x1, y1 = new_pos
            x2, y2 = pts[j]
            x3, y3 = pts[k]
            det = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
            if det > 0:
                global_counts[det] += 1
                
        return True, 0 # We'll check max separately

    # Main search loop
    # We use a greedy approach with random restarts.
    
    # We'll do multiple restarts.
    max_restarts = 20
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Initialize points for this restart
        if restart == 0:
            pts = [(x, y) for x in range(6) for y in range(5)]
        else:
            # Perturb best_points
            pts = best_points[:]
            for _ in range(3):
                idx = random.randint(0, n-1)
                nx = random.randint(-30, 30)
                ny = random.randint(-30, 30)
                if (nx, ny) not in pts:
                    pts[idx] = (nx, ny)
                    
        # Recalculate counts for this restart
        global_counts = init_counts(pts)
        if not global_counts:
            current_max_freq = 0
            current_target_area = 0
        else:
            current_max_freq = max(global_counts.values())
            # Pick the area with max frequency as target
            current_target_area = [a for a, c in global_counts.items() if c == current_max_freq][0]
        
        if current_max_freq > best_max_freq:
            best_max_freq = current_max_freq
            best_points = pts[:]
            
        # Local search: Try to increase the count of the current target area
        improved = True
        no_improve_count = 0
        max_no_improve = 5
        
        while improved and time.time() < deadline:
            improved = False
            indices = list(range(n))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() > deadline:
                    break
                
                # Try a few random moves for point i
                for _ in range(5):
                    nx = random.randint(-30, 30)
                    ny = random.randint(-30, 30)
                    new_pos = (nx, ny)
                    
                    if new_pos in pts:
                        continue
                    
                    old_pos = pts[i]
                    
                    # Decrement old contributions
                    for j, k in pairs_for_point[i]:
                        x1, y1 = old_pos
                        x2, y2 = pts[j]
                        x3, y3 = pts[k]
                        det = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
                        if det > 0:
                            global_counts[det] -= 1
                            if global_counts[det] == 0:
                                del global_counts[det]
                            
                    pts[i] = new_pos
                    
                    # Increment new contributions
                    new_target_count = global_counts.get(current_target_area, 0)
                    for j, k in pairs_for_point[i]:
                        x1, y1 = new_pos
                        x2, y2 = pts[j]
                        x3, y3 = pts[k]
                        det = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
                        if det > 0:
                            global_counts[det] += 1
                            if det == current_target_area:
                                new_target_count += 1

                    # Check if we increased the target area count or found a new max
                    # Strategy: Accept if target area count increased, OR if some area count exceeds current_max_freq
                    # For simplicity and speed, we primarily focus on increasing the current target.
                    # But we must also handle if the target area count drops (it shouldn't in this move logic unless we revert, but we are moving forward).
                    # Actually, moving point i changes 406 triples. The count of target area can go up or down.
                    
                    # Let's check the new max frequency efficiently.
                    # We only care if new_target_count > current_max_freq OR if new_target_count == current_max_freq and we are "stable"?
                    # No, the objective is max multiplicity.
                    # If new_target_count > current_max_freq, we have a strict improvement.
                    # If new_target_count < current_max_freq, we might have worsened the target, but maybe another area became dominant?
                    # Calculating full max is O(K). K is small (likely < 100 unique areas).
                    # Let's just calculate max if new_target_count is close to current_max_freq.
                    
                    # Optimization: If new_target_count > current_max_freq, accept immediately.
                    # Else, check if any other area exceeded current_max_freq.
                    # To avoid O(K) check every time, we can just revert if new_target_count < current_max_freq.
                    # This is a greedy heuristic: "Always try to increase the current best bucket".
                    
                    if new_target_count > current_max_freq:
                        current_max_freq = new_target_count
                        improved = True
                        no_improve_count = 0
                        if current_max_freq > best_max_freq:
                            best_max_freq = current_max_freq
                            best_points = pts[:]
                        break # Move to next point
                    else:
                        # Revert
                        pts[i] = old_pos
                        # Re-increment old
                        for j, k in pairs_for_point[i]:
                            x1, y1 = old_pos
                            x2, y2 = pts[j]
                            x3, y3 = pts[k]
                            det = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
                            if det > 0:
                                global_counts[det] += 1
                        # Re-decrement new
                        for j, k in pairs_for_point[i]:
                            x1, y1 = new_pos
                            x2, y2 = pts[j]
                            x3, y3 = pts[k]
                            det = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
                            if det > 0:
                                global_counts[det] -= 1
                                if global_counts[det] == 0:
                                    del global_counts[det]
                
                if not improved:
                    no_improve_count += 1
                    if no_improve_count > max_no_improve:
                        break

        if time.time() > deadline:
            break

    return best_points

# EVOLVE-BLOCK-END
