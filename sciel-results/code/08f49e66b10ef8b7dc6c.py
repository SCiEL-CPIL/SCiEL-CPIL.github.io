import time
import random
from itertools import permutations
import numpy as np

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
        
    class SkylineIndex:
        """Spatial index using skyline representation for fast overlap detection."""
        def __init__(self, W, H, D):
            self.W, self.H, self.D = W, H, D
            # Store placed boxes as list of (x, y, z, w, h, d)
            self.placed = []
            # For fast overlap checking, we'll use a simple approach:
            # Group boxes by z-level and check only relevant ones
            self.z_levels = {}  # z -> list of boxes at that z-level
            # Pre-compute z ranges for fast filtering
            self.z_ranges = []  # list of (z_min, z_max) for each placed box
        
        def add(self, x, y, z, w, h, d):
            """Add a placed box to the index."""
            self.placed.append((x, y, z, w, h, d))
            if z not in self.z_levels:
                self.z_levels[z] = []
            self.z_levels[z].append((x, y, z, w, h, d))
            self.z_ranges.append((z, z + d))
        
        def check_overlap(self, x, y, z, w, h, d):
            """Check if box at (x,y,z) with size (w,h,d) overlaps with any placed box."""
            z_min = z
            z_max = z + d
            
            # Quick check: iterate through all placed boxes but filter by z-range first
            # Use the z_ranges list for fast filtering
            for idx, (rz_min, rz_max) in enumerate(self.z_ranges):
                # Check z overlap: rz_min < z_max and rz_max > z_min
                if rz_min >= z_max or rz_max <= z_min:
                    continue
                
                # Get the box at this index
                px, py, pz, pw, ph, pd = self.placed[idx]
                
                # Check x overlap: px < x+w and px+pw > x
                if px + pw <= x or px >= x + w:
                    continue
                # Check y overlap: py < y+h and py+ph > y
                if py + ph <= y or py >= y + h:
                    continue
                # All three dimensions overlap
                return True
            return False

    def solve_greedy(box_indices, sort_key):
        """
        Solve using greedy corner point search with skyline index.
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
        
        index = SkylineIndex(W, H, D)
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
                    if index.check_overlap(cx, cy, cz, rw, rh, rd):
                        continue
                    # If we get here, it fits
                    best_placement = (cx, cy, cz, rw, rh, rd)
                    found = True
                    break
            
            if best_placement:
                x, y, z, rw, rh, rd = best_placement
                index.add(x, y, z, rw, rh, rd)
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
        
        index = SkylineIndex(W, H, D)
        candidates = {(0, 0, 0)}
        placements = [None] * len(boxes)
        
        for i, w, h, d in items:
            best_placement = None
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
                    if index.check_overlap(cx, cy, cz, rw, rh, rd):
                        continue
                    best_placement = (cx, cy, cz, rw, rh, rd)
                    found = True
                    break
            
            if best_placement:
                x, y, z, rw, rh, rd = best_placement
                index.add(x, y, z, rw, rh, rd)
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

    # Run more randomized versions since we have faster overlap checking
    num_restarts = 50
    for seed in range(num_restarts):
        if time.time() > deadline:
            break
        pl = solve_greedy_randomized(indices, sort_vol, seed)
        sc = score_placements(pl)
        if sc > best_score:
            best_score = sc
            best_placements = pl
    
    # Also try randomized with other sort keys
    for key in strategies:
        for seed in range(5):
            if time.time() > deadline:
                break
            pl = solve_greedy_randomized(indices, key, seed + 100)
            sc = score_placements(pl)
            if sc > best_score:
                best_score = sc
                best_placements = pl

    if best_placements is None:
        # Fallback: all left out
        best_placements = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    return best_placements
