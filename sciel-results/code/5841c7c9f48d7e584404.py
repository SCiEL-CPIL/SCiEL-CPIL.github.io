import time
import random
from itertools import permutations

def load(boxes, container):
    W, H, D = container
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # Pre-process boxes: (w, h, d)
    # We need to handle the output format: (bin, x, y, z, w, h, d)
    # bin 0 for loaded, -1 for left out.
    
    def get_rotations(w, h, d):
        seen = set()
        rots = []
        for p in permutations((w, h, d)):
            if p not in seen:
                seen.add(p)
                rots.append(p)
        return rots

    # Pre-compute rotations for all boxes to save time in loops
    box_rots = []
    for b in boxes:
        box_rots.append(get_rotations(*b))
        
    def check_overlap(x, y, z, w, h, d, placed):
        """Check if box at (x,y,z) with size (w,h,d) overlaps with any in placed."""
        for px, py, pz, pw, ph, pd in placed:
            if not (x + w <= px or px + pw <= x or
                    y + h <= py or py + ph <= y or
                    z + d <= pz or pz + pd <= z):
                return True
        return False

    def solve_greedy(box_indices, sort_key):
        """
        Solve using greedy corner point search.
        box_indices: list of indices into 'boxes'
        sort_key: function to sort boxes
        Returns: list of placements (bin, x, y, z, w, h, d)
        """
        # Create list of (index, w, h, d)
        items = []
        for i in box_indices:
            w, h, d = boxes[i]
            items.append((i, w, h, d))
        
        # Sort
        items.sort(key=lambda x: sort_key(x[1], x[2], x[3]), reverse=True)
        
        placed = [] # (x, y, z, w, h, d)
        candidates = {(0, 0, 0)}
        placements = [None] * len(boxes) # Initialize all as None, will fill with -1 or 0
        
        for i, w, h, d in items:
            best_placement = None
            # Try to find a spot
            # We iterate candidates in a specific order to encourage tight packing
            # Sort candidates by (z, y, x) to fill bottom-up, front-to-back, left-to-right
            sorted_candidates = sorted(candidates, key=lambda p: (p[2], p[1], p[0]))
            
            found = False
            for cx, cy, cz in sorted_candidates:
                if found:
                    break
                # Try all rotations for this box at this candidate point
                for rw, rh, rd in box_rots[i]:
                    # Check bounds
                    if cx + rw > W or cy + rh > H or cz + rd > D:
                        continue
                    # Check overlap
                    if check_overlap(cx, cy, cz, rw, rh, rd, placed):
                        continue
                    # If we get here, it fits
                    best_placement = (cx, cy, cz, rw, rh, rd)
                    found = True
                    break
            
            if best_placement:
                x, y, z, rw, rh, rd = best_placement
                placed.append((x, y, z, rw, rh, rd))
                placements[i] = (0, x, y, z, rw, rh, rd)
                
                # Generate new candidates
                new_pts = [
                    (x + rw, y, z),
                    (x, y + rh, z),
                    (x, y, z + rd),
                    (x + rw, y + rh, z),
                    (x + rw, y, z + rd),
                    (x, y + rh, z + rd),
                    (x + rw, y + rh, z + rd)
                ]
                for p in new_pts:
                    if 0 <= p[0] <= W and 0 <= p[1] <= H and 0 <= p[2] <= D:
                        candidates.add(p)
            else:
                # Could not place
                placements[i] = (-1, 0, 0, 0, w, h, d)
                
        return placements

    def score_placements(placements):
        total_vol = 0
        for p in placements:
            if p[0] == 0:
                total_vol += p[4] * p[5] * p[6]
        return total_vol

    # Strategy 1: Volume Descending
    indices = list(range(len(boxes)))
    best_score = -1
    best_placements = None
    
    # Define sort keys
    sort_vol = lambda w, h, d: w * h * d
    sort_max = lambda w, h, d: max(w, h, d)
    sort_sum = lambda w, h, d: w + h + d
    sort_area = lambda w, h, d: 2 * (w * h + h * d + w * d)
    
    strategies = [
        sort_vol,
        sort_max,
        sort_sum,
        sort_area
    ]
    
    for key in strategies:
        if time.time() > deadline:
            break
        pl = solve_greedy(indices, key)
        sc = score_placements(pl)
        if sc > best_score:
            best_score = sc
            best_placements = pl

    # Strategy 2: Randomized Restarts with Volume Sort
    # If time permits, do a few random shuffles to break ties or find different local optima
    # However, greedy with corner points is deterministic for a given sort order.
    # Randomizing the order of candidates might help.
    
    def solve_greedy_randomized(box_indices, sort_key, seed):
        random.seed(seed)
        items = []
        for i in box_indices:
            w, h, d = boxes[i]
            items.append((i, w, h, d))
        
        items.sort(key=lambda x: sort_key(x[1], x[2], x[3]), reverse=True)
        
        placed = []
        candidates = {(0, 0, 0)}
        placements = [None] * len(boxes)
        
        for i, w, h, d in items:
            best_placement = None
            # Shuffle candidates slightly? No, that breaks the "fill from bottom" logic.
            # Instead, we can add a small random perturbation to the candidate selection?
            # Or just rely on the different sort keys.
            
            # Let's try a "Best Fit" variant: check ALL candidates and pick the one with min (z,y,x)
            # This is what we did. 
            # To add randomness, we can randomly permute the rotations for each box.
            
            rots = box_rots[i][:]
            random.shuffle(rots)
            
            sorted_candidates = sorted(candidates, key=lambda p: (p[2], p[1], p[0]))
            
            found = False
            for cx, cy, cz in sorted_candidates:
                if found:
                    break
                for rw, rh, rd in rots:
                    if cx + rw > W or cy + rh > H or cz + rd > D:
                        continue
                    if check_overlap(cx, cy, cz, rw, rh, rd, placed):
                        continue
                    best_placement = (cx, cy, cz, rw, rh, rd)
                    found = True
                    break
            
            if best_placement:
                x, y, z, rw, rh, rd = best_placement
                placed.append((x, y, z, rw, rh, rd))
                placements[i] = (0, x, y, z, rw, rh, rd)
                
                new_pts = [
                    (x + rw, y, z),
                    (x, y + rh, z),
                    (x, y, z + rd),
                    (x + rw, y + rh, z),
                    (x + rw, y, z + rd),
                    (x, y + rh, z + rd),
                    (x + rw, y + rh, z + rd)
                ]
                for p in new_pts:
                    if 0 <= p[0] <= W and 0 <= p[1] <= H and 0 <= p[2] <= D:
                        candidates.add(p)
            else:
                placements[i] = (-1, 0, 0, 0, w, h, d)
                
        return placements

    # Run a few randomized versions of the Volume Sort strategy
    for seed in range(10):
        if time.time() > deadline:
            break
        pl = solve_greedy_randomized(indices, sort_vol, seed)
        sc = score_placements(pl)
        if sc > best_score:
            best_score = sc
            best_placements = pl

    if best_placements is None:
        # Fallback: all left out
        best_placements = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    return best_placements
