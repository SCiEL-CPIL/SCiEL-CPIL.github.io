import time
import numpy as np
from collections import defaultdict
import random
import itertools

def construct():
    t0 = time.time()
    DEADLINE = 58.0  # Safety margin before 60s limit

    def elapsed():
        return time.time() - t0

    def compute_max_area_count_fast(pts):
        """
        Compute the maximum multiplicity of any non-zero doubled area.
        Optimized using pairwise difference vectors and numpy.
        """
        n = len(pts)
        if n < 3:
            return 0

        P = np.array(pts, dtype=np.int64)

        # Build all pairwise difference vectors (i, j) with i < j
        i_idx, j_idx = np.triu_indices(n, k=1)
        d = P[j_idx] - P[i_idx]  # shape (M, 2), M = n*(n-1)/2
        dx = d[:, 0]
        dy = d[:, 1]
        M = len(dx)

        # For each base vector a (index u), compute cross products with all later vectors v
        area_counts = defaultdict(int)
        for u in range(M - 1):
            cu = dx[u]
            cv = dy[u]
            if cu == 0 and cv == 0:
                continue
            cross = cu * dy[u+1:] - cv * dx[u+1:]
            nz = cross[cross != 0]
            if len(nz) == 0:
                continue
            nz_abs = np.abs(nz)
            vals, cnts = np.unique(nz_abs, return_counts=True)
            for v, c in zip(vals, cnts):
                area_counts[int(v)] += int(c)

        if not area_counts:
            return 0
        return max(area_counts.values())

    def score_of(pts):
        return compute_max_area_count_fast(pts)

    # 1. Initialize with strong structured candidates
    
    # Candidate 1: Two lines, 15 points each
    cand1 = [(x, 0) for x in range(15)] + [(x, 1) for x in range(15)]
    
    # Candidate 2: Grid 6x5
    cand2 = [(x, y) for x in range(6) for y in range(5)]
    
    # Candidate 3: Three lines, 10 points each
    cand3 = [(x, 0) for x in range(10)] + [(x, 1) for x in range(10)] + [(x, 2) for x in range(10)]
    
    # Candidate 4: Random
    random.seed(42)
    cand4 = []
    used = set()
    while len(cand4) < 30:
        x = random.randint(-30, 30)
        y = random.randint(-30, 30)
        if (x, y) not in used:
            used.add((x, y))
            cand4.append((x, y))
            
    # Candidate 5: Four lines with varying spacing
    cand5 = [(x, 0) for x in range(8)] + [(x, 2) for x in range(8)] + \
            [(x, 4) for x in range(7)] + [(x, 6) for x in range(7)]
    
    # Candidate 6: Concentric-like structure
    cand6 = [(x, 0) for x in range(10)] + [(x, 1) for x in range(10)] + \
            [(x, -1) for x in range(5)] + [(x, -2) for x in range(5)]
    
    # Candidate 7: 5 lines, 6 points each
    cand7 = [(x, 0) for x in range(6)] + [(x, 1) for x in range(6)] + \
            [(x, 2) for x in range(6)] + [(x, 3) for x in range(6)] + \
            [(x, 4) for x in range(6)]
    
    # Candidate 8: 2 lines with offset
    cand8 = [(x, 0) for x in range(15)] + [(x+1, 1) for x in range(14)] + [(14, 1)]
    # Ensure distinctness and bounds for cand8
    cand8 = []
    used8 = set()
    for x in range(15):
        pt = (x, 0)
        if pt not in used8:
            cand8.append(pt)
            used8.add(pt)
    for x in range(15):
        pt = (x, 1)
        if pt not in used8:
            cand8.append(pt)
            used8.add(pt)
    # If less than 30, fill randomly
    if len(cand8) < 30:
        while len(cand8) < 30:
            x = random.randint(-30, 30)
            y = random.randint(-30, 30)
            if (x, y) not in used8:
                cand8.append((x, y))
                used8.add((x, y))

    candidates = [cand1, cand2, cand3, cand4, cand5, cand6, cand7, cand8]
    
    best_points = None
    best_score = 0
    
    for cand in candidates:
        s = score_of(cand)
        if s > best_score:
            best_score = s
            best_points = list(cand)
            
    # 2. Targeted Simulated Annealing with Incremental Scoring and Swap Moves
    # Strategy: 
    # 1. Use incremental area count updates to speed up evaluation.
    # 2. Include swap moves (exchange two points) to escape local minima.
    # 3. Larger neighborhood (±5) and more iterations.
    
    current_points = list(best_points)
    current_score = best_score
    
    # Precompute area counts for all areas to enable incremental updates
    def build_area_counts(pts):
        n = len(pts)
        P = np.array(pts, dtype=np.int64)
        i_idx, j_idx = np.triu_indices(n, k=1)
        d = P[j_idx] - P[i_idx]
        dx = d[:, 0]
        dy = d[:, 1]
        M = len(dx)
        area_counts = defaultdict(int)
        for u in range(M - 1):
            cu = dx[u]
            cv = dy[u]
            if cu == 0 and cv == 0:
                continue
            cross = cu * dy[u+1:] - cv * dx[u+1:]
            nz = cross[cross != 0]
            if len(nz) == 0:
                continue
            nz_abs = np.abs(nz)
            vals, cnts = np.unique(nz_abs, return_counts=True)
            for v, c in zip(vals, cnts):
                area_counts[int(v)] += int(c)
        return area_counts

    current_area_counts = build_area_counts(current_points)
    current_max_count = max(current_area_counts.values()) if current_area_counts else 0
    
    # Get the dominant area from the initial best
    def get_dominant_area(pts, area_counts):
        if not area_counts:
            return 1, 0
        max_count = max(area_counts.values())
        for a, c in area_counts.items():
            if c == max_count:
                return a, c
        return 1, 0

    target_area, target_count = get_dominant_area(current_points, current_area_counts)
    
    # Helper to incrementally update area counts when moving point idx from old_pt to new_pt
    def update_area_counts_move(area_counts, pts, idx, old_pt, new_pt):
        """
        Update area_counts when point at idx changes from old_pt to new_pt.
        Returns updated area_counts (in-place modification for speed).
        """
        n = len(pts)
        P = np.array(pts, dtype=np.int64)
        
        # Compute areas involving point idx with old_pt
        for i in range(n):
            if i == idx:
                continue
            for j in range(i + 1, n):
                if j == idx:
                    continue
                # Triangle (i, idx, j)
                p1 = P[i]
                p2 = P[j]
                # Old area: |det(p1 - old_pt, p2 - old_pt)|
                v1_old = p1 - np.array(old_pt)
                v2_old = p2 - np.array(old_pt)
                area_old = abs(int(v1_old[0] * v2_old[1] - v1_old[1] * v2_old[0]))
                if area_old > 0:
                    area_counts[area_old] -= 1
                    if area_counts[area_old] == 0:
                        del area_counts[area_old]
                
                # New area: |det(p1 - new_pt, p2 - new_pt)|
                v1_new = p1 - np.array(new_pt)
                v2_new = p2 - np.array(new_pt)
                area_new = abs(int(v1_new[0] * v2_new[1] - v1_new[1] * v2_new[0]))
                if area_new > 0:
                    area_counts[area_new] = area_counts.get(area_new, 0) + 1

    # Helper to incrementally update area counts when swapping points i and j
    def update_area_counts_swap(area_counts, pts, i, j):
        """
        Update area_counts when points at i and j are swapped.
        Returns updated area_counts (in-place modification for speed).
        """
        n = len(pts)
        P = np.array(pts, dtype=np.int64)
        
        # Areas involving i or j (but not both, as they are swapped)
        # For any k != i, j:
        #   Old: triangles (i, j, k) -> new: same (since i and j are just labels, the set of points is same)
        #   Actually, swapping i and j doesn't change the set of points, so all areas remain the same!
        # Wait, this is correct: swapping two points in the list doesn't change the geometry.
        # So area_counts remains unchanged.
        pass

    random.seed(42)
    temperature = 10.0
    cooling_rate = 0.9995
    iteration = 0
    
    # We will run SA for a fixed number of iterations or until time runs out
    # We focus on moves that increase the count of the target area.
    
    while elapsed() < DEADLINE:
        iteration += 1
        # Occasionally re-evaluate the global score to see if we've beaten the best
        # and potentially update the target area
        if iteration % 100 == 0:
            global_score = max(current_area_counts.values()) if current_area_counts else 0
            if global_score > best_score:
                best_score = global_score
                best_points = list(current_points)
                # Update target area to the new dominant one
                target_area, target_count = get_dominant_area(current_points, current_area_counts)
        
        # Decide move type: 80% single point move, 20% swap
        if random.random() < 0.8:
            # Single point move
            idx = random.randint(0, 29)
            old_x, old_y = current_points[idx]
            
            # Generate a new position
            step_size = random.randint(1, 5)
            new_x = old_x + random.randint(-step_size, step_size)
            new_y = old_y + random.randint(-step_size, step_size)
            
            # Ensure bounds
            if new_x < -30 or new_x > 30 or new_y < -30 or new_y > 30:
                continue
                
            # Check distinctness
            new_pt = (new_x, new_y)
            if new_pt == (old_x, old_y):
                continue
            is_valid = True
            for j in range(30):
                if j == idx:
                    continue
                if current_points[j] == new_pt:
                    is_valid = False
                    break
            
            if not is_valid:
                continue
                
            # Try the move
            old_pt = (old_x, old_y)
            current_points[idx] = new_pt
            
            # Update area counts
            old_counts = dict(current_area_counts)
            update_area_counts_move(current_area_counts, current_points, idx, old_pt, new_pt)
            
            new_target_count = current_area_counts.get(target_area, 0)
            delta_target = new_target_count - target_count
            
            # Acceptance criterion
            if delta_target > 0:
                accept = True
            elif delta_target == 0:
                accept = random.random() < 0.5
            else:
                accept_prob = np.exp(delta_target / temperature)
                accept = random.random() < accept_prob
            
            if accept:
                target_count = new_target_count
            else:
                # Revert
                current_points[idx] = old_pt
                current_area_counts = old_counts
                
        else:
            # Swap move
            i, j = random.sample(range(30), 2)
            # Swapping doesn't change geometry, so skip for now
            # But we can use it to reorganize for future moves
            continue
            
        # Cool down
        temperature *= cooling_rate
        if temperature < 0.1:
            temperature = 1.0 # Reheat occasionally to escape deep local minima

    # Final check
    final_score = max(current_area_counts.values()) if current_area_counts else 0
    if final_score > best_score:
        best_score = final_score
        best_points = list(current_points)

    return best_points
