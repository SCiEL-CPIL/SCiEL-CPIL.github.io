import time
import numpy as np
import random
from collections import Counter

def construct():
    t0 = time.time()
    DEADLINE = 85.0  # Safety margin before 90s limit

    def elapsed():
        return time.time() - t0

    def compute_areas_and_counts(pts):
        """
        Compute all non-zero doubled areas and return:
        1. A Counter of area -> count
        2. The maximum count
        3. The area value with the maximum count (ties broken by smaller area)
        """
        n = len(pts)
        P = np.array(pts, dtype=np.int64)
        
        # Vectorized computation of all triple cross products
        # To save memory and time, we iterate over the first point
        # but use numpy for the remaining 29x28/2 pairs? 
        # Actually, 30C3 = 4060. Vectorizing the whole thing is easy.
        i_idx, j_idx, k_idx = np.meshgrid(
            np.arange(n), np.arange(n), np.arange(n), indexing='ij'
        )
        # Filter i < j < k
        mask = (i_idx < j_idx) & (j_idx < k_idx)
        i_idx = i_idx[mask]
        j_idx = j_idx[mask]
        k_idx = k_idx[mask]
        
        P_i = P[i_idx]
        P_j = P[j_idx]
        P_k = P[k_idx]
        
        # Cross product (P_j - P_i) x (P_k - P_i)
        # (x_j - x_i)*(y_k - y_i) - (y_j - y_i)*(x_k - x_i)
        areas = (P_j[:, 0] - P_i[:, 0]) * (P_k[:, 1] - P_i[:, 1]) - \
                (P_j[:, 1] - P_i[:, 1]) * (P_k[:, 0] - P_i[:, 0])
        
        abs_areas = np.abs(areas)
        # Exclude zero area
        non_zero = abs_areas[abs_areas > 0]
        
        if len(non_zero) == 0:
            return Counter(), 0, 0
            
        # Convert to list of ints for Counter
        # Using np.unique is faster for counting
        vals, cnts = np.unique(non_zero, return_counts=True)
        
        max_count = int(cnts.max())
        # Find the area with max count. If multiple, pick the one that appears first in sorted order?
        # Or just the one with max count.
        max_idx = np.argmax(cnts)
        target_area = int(vals[max_idx])
        
        # Build a counter for the specific target area? 
        # The task asks to maximize the largest multiplicity of a single positive doubled area.
        # So we just need the max count.
        # However, for the "area-locking" strategy, we need to know WHICH area is the target.
        # We return the full unique values and counts to allow re-targeting if needed.
        
        return vals, cnts, max_count, target_area

    def score_of(pts):
        """
        Returns the maximum multiplicity of any non-zero doubled area.
        """
        try:
            vals, cnts, max_count, target_area = compute_areas_and_counts(pts)
            return max_count
        except:
            return 0

    def incremental_score(pts, idx, new_pt, current_areas_dict, current_target_area, current_max_count):
        """
        Fast approximation: 
        1. Remove old point's contributions.
        2. Add new point's contributions.
        3. Check if the count for the current target area increased.
        4. If not, check if any other area's count exceeded current_max_count.
        
        To keep it fast, we only track the counts for a few "candidate" areas:
        - The current target area
        - Areas that are close in count to the max
        
        Actually, a full recount is ~4000 ops. 
        Let's just do a full recount but optimized in numpy. 
        4000 ops is very fast in numpy. 
        The bottleneck is the python loop overhead in the current program's local search 
        if it calls a slow scorer.
        
        Let's verify speed of compute_areas_and_counts.
        """
        # We'll just call the full scorer. It's fast enough.
        return score_of(pts)

    # 1. Initialization: Generate strong structured candidates
    # The "Two lines" configuration is known to be strong.
    # Let's generate variations of parallel lines.
    
    candidates = []
    
    # Candidate 1: Two lines, 15 points each, y=0 and y=1
    cand1 = [(x, 0) for x in range(15)] + [(x, 1) for x in range(15)]
    candidates.append(cand1)
    
    # Candidate 2: Three lines, 10 points each
    cand2 = [(x, 0) for x in range(10)] + [(x, 1) for x in range(10)] + [(x, 2) for x in range(10)]
    candidates.append(cand2)
    
    # Candidate 3: Five lines, 6 points each
    cand3 = [(x, 0) for x in range(6)] + [(x, 1) for x in range(6)] + \
            [(x, 2) for x in range(6)] + [(x, 3) for x in range(6)] + \
            [(x, 4) for x in range(6)]
    candidates.append(cand3)
    
    # Candidate 4: Grid 6x5
    cand4 = [(x, y) for x in range(6) for y in range(5)]
    candidates.append(cand4)
    
    # Candidate 5: Two lines with offset x-range
    cand5 = [(x, 0) for x in range(15)] + [(x, 1) for x in range(1, 16)]
    candidates.append(cand5)
    
    # Candidate 6: Two lines, wider spacing
    cand6 = [(x, 0) for x in range(15)] + [(x, 2) for x in range(15)]
    candidates.append(cand6)

    # Evaluate candidates
    best_points = None
    best_score = 0
    best_target_area = 0
    
    for cand in candidates:
        # Ensure distinct and in bounds
        if len(set(cand)) != 30:
            continue
        if any(abs(x) > 30 or abs(y) > 30 for x, y in cand):
            continue
            
        vals, cnts, max_count, target_area = compute_areas_and_counts(cand)
        if max_count > best_score:
            best_score = max_count
            best_points = list(cand)
            best_target_area = target_area

    # 2. Targeted Local Search (Area-Locking)
    # Strategy: 
    # We focus on increasing the count of triangles with area == best_target_area.
    # We allow switching the target area if we find a better one.
    
    current_points = list(best_points)
    current_score = best_score
    current_target_area = best_target_area
    
    random.seed(42)
    
    # Helper to get current area counts
    def get_area_counts(pts):
        try:
            vals, cnts, max_count, target_area = compute_areas_and_counts(pts)
            # Return a dict for O(1) lookup
            d = dict(zip(vals, cnts))
            return d, max_count, target_area
        except:
            return {}, 0, 0

    # Initial state
    area_counts_dict, cur_max, cur_target = get_area_counts(current_points)
    
    # We will maintain a "hot" set of areas to check.
    # Actually, just checking the current target and the global max is enough.
    
    # To speed up, we pre-compute the "impact" of moving a point?
    # Moving point i affects 29*28/2 = 406 triangles? No, 29*28/2 is wrong.
    # Point i is in 29*28/2 triangles? 
    # Number of triangles containing point i is C(29, 2) = 406.
    # So we need to update 406 area counts per move.
    # This is very fast.
    
    def update_counts_on_move(pts, idx, old_pt, new_pt, area_counts_dict):
        """
        Update area_counts_dict when moving pts[idx] from old_pt to new_pt.
        Returns updated dict.
        """
        n = len(pts)
        P = np.array(pts, dtype=np.int64)
        P[idx] = new_pt
        
        # We need to recalculate areas for all triples involving idx.
        # Triples: (idx, a, b) where a, b are in 0..n-1, a!=idx, b!=idx, a<b or similar.
        
        # Simpler: Just recount everything. 
        # 4060 ops in numpy is ~10-20 microseconds.
        # 1000 iterations/sec is easy.
        # Let's benchmark.
        pass

    # Let's stick to the full recount for simplicity and robustness, 
    # but optimize the loop structure.
    
    # Optimization: 
    # Instead of random moves, we can try "greedy" moves that specifically 
    # create the target area.
    
    # For now, let's do a fast local search with a tighter deadline check.
    
    max_iters = 100000 # Cap to prevent infinite loops if time check fails
    iter_count = 0
    no_improve = 0
    step_size = 2
    
    while elapsed() < DEADLINE and iter_count < max_iters:
        iter_count += 1
        
        # Pick a random point
        idx = random.randint(0, 29)
        old_pt = current_points[idx]
        
        # Generate neighbor
        # Bias towards moves that might create the target area?
        # Hard to do analytically. Stick to random walk but with smaller steps.
        
        r = random.random()
        if r < 0.05:
            # Medium jump
            dx = random.randint(-3, 3)
            dy = random.randint(-3, 3)
        else:
            # Small step
            dx = random.randint(-1, 1)
            dy = random.randint(-1, 1)
        
        new_x = old_pt[0] + dx
        new_y = old_pt[1] + dy
        
        if new_x < -30 or new_x > 30 or new_y < -30 or new_y > 30:
            continue
            
        new_pt = (new_x, new_y)
        if new_pt == old_pt:
            continue
            
        # Check distinctness
        # This is O(N) in python, which is slow.
        # Use a set.
        # We maintain a set of current points.
        # Let's maintain a set outside the loop.
        
        # For now, simple check
        if new_pt in current_points:
            continue
            
        # Apply move
        current_points[idx] = new_pt
        
        # Score
        new_score = score_of(current_points)
        
        if new_score > current_score:
            current_score = new_score
            no_improve = 0
            # Update target area if it changed or if we want to track it
            _, _, _, new_target = compute_areas_and_counts(current_points)
            current_target_area = new_target
            
            if current_score > best_score:
                best_score = current_score
                best_points = list(current_points)
                best_target_area = current_target_area
        else:
            # Revert
            current_points[idx] = old_pt
            no_improve += 1
            
        # If stuck, restart or increase step size
        if no_improve > 1000:
            no_improve = 0
            # Restart from best
            current_points = list(best_points)
            current_score = best_score
            current_target_area = best_target_area
            # Increase step size to escape
            step_size = random.randint(2, 5)
            # Or try a random restart
            if random.random() < 0.1:
                # Generate a new random candidate
                rand_pts = []
                used = set()
                while len(rand_pts) < 30:
                    x = random.randint(-30, 30)
                    y = random.randint(-30, 30)
                    if (x, y) not in used:
                        used.add((x, y))
                        rand_pts.append((x, y))
                s = score_of(rand_pts)
                if s > current_score:
                    current_points = rand_pts
                    current_score = s
                    if s > best_score:
                        best_score = s
                        best_points = list(rand_pts)

    # 3. Final Polish: 
    # Try to fix any points that are "wasted" (not contributing to the target area)
    # by moving them to positions that create the target area.
    
    # This is complex to implement efficiently without a full solver.
    # The local search above should be sufficient if fast.
    
    return best_points
