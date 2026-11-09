"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from math import floor, ceil

# EVOLVE-BLOCK-START
def _rotations(w, h, d):
    """Generate all 6 unique axis rotations of a box."""
    dims = [w, h, d]
    seen = set()
    out = []
    for i in range(3):
        for j in range(3):
            if j == i:
                continue
            k = 6 - i - j
            t = (dims[i], dims[j], dims[k])
            if t not in seen:
                seen.add(t)
                out.append(t)
    return out

def _try_place(heights, W, H, D, bw, bh, bd, step):
    """
    Try to place a box of size (bw, bh, bd) on the height-field.
    Returns (x, y, z) if placed, else None.
    Uses a bottom-left-front most-fit heuristic.
    """
    # We scan candidate positions. To keep it fast, we use a coarse step
    # but also try the exact grid-aligned positions.
    # For efficiency, we limit the search space.
    best = None
    best_score = float('inf')
    
    # Candidate x, y positions
    # We only need to check positions where the box can start
    # To speed up, we can skip positions that are clearly invalid
    for x in range(0, W - bw + 1, step):
        for y in range(0, H - bd + 1, step):
            # Check if the box fits in this column range
            # The box occupies x to x+bw, y to y+bd
            # We need the max height in that range
            # For speed, we sample the corners and center, or use a fast max
            # A full max is O(bw*bd) which can be slow.
            # Let's approximate by checking the 4 corners and the center.
            # If the box is small, check all.
            
            # Fast check: if any corner is too high, skip
            if heights[x, y] + bh > D:
                continue
            if heights[x + bw - 1, y] + bh > D:
                continue
            if heights[x, y + bd - 1] + bh > D:
                continue
            if heights[x + bw - 1, y + bd - 1] + bh > D:
                continue
            
            # More detailed check: max height in the footprint
            # To be exact, we need the max.
            # For now, let's do a quick max check.
            # If this is too slow, we will need to optimize.
            # Given 60 boxes, and container sizes likely small integers,
            # a full check might be okay if step is 1.
            
            # Let's do the full max check for correctness.
            # Optimization: if bw*bd is large, this is slow.
            # But 60 boxes * 6 rotations * positions * max_check...
            # Let's assume container dims are < 100.
            
            max_h = 0
            # Unroll for small sizes? No, just numpy max.
            # But we are in a loop, so numpy overhead is high.
            # Let's use a simple loop if dims are small, or numpy if large.
            # Actually, let's just use numpy max on the slice.
            # heights[x:x+bw, y:y+bd]
            
            # To avoid numpy overhead in tight loops, we can do:
            # If the box is 1x1, just check heights[x,y]
            if bw == 1 and bd == 1:
                max_h = heights[x, y]
            else:
                # Use numpy
                max_h = np.max(heights[x:x+bw, y:y+bd])
                
            if max_h + bh > D:
                continue
            
            # Score: prefer lower z, then smaller x, then smaller y
            # This is "bottom-left-front"
            z = max_h
            # Score for sorting: lower is better
            # We want the position with the lowest z, then lowest x, then lowest y
            # To make this a "min" problem, we can use a tuple
            # But we are in a loop, so let's compute a score
            # score = z * 10000 + x * 100 + y
            # This assumes W, H < 100. If not, we need larger multipliers.
            # Let's use a tuple comparison instead for robustness.
            
            curr = (z, x, y)
            if best is None or curr < best:
                best = curr
                best_score = curr
    
    if best is None:
        return None
    
    x, y, z = best
    # Update height field
    heights[x:x+bw, y:y+bd] = np.maximum(heights[x:x+bw, y:y+bd], z + bh)
    return (x, y, z)

def load(boxes, container):
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # Time budget
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Pre-compute rotations for each box
    box_rots = []
    for i, (w, h, d) in enumerate(boxes):
        rots = _rotations(w, h, d)
        box_rots.append(rots)
    
    # Sort boxes by volume descending
    box_indices = sorted(range(len(boxes)), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    
    best_placements = None
    best_vol = 0
    
    # Determine step size based on container size
    # If container is large, use step > 1 to speed up
    # But we need to be careful not to miss optimal positions.
    # For now, let's use step 1 for small containers, and maybe 2 for large.
    # Let's start with step 1 and see if it's fast enough.
    
    num_restarts = 50  # We will adjust based on time
    
    for restart in range(num_restarts):
        if time.time() > deadline - 2.0:
            break
        
        # Shuffle the box order slightly for diversity
        # But keep large boxes first?
        # Let's try: sort by volume, but add some noise to the sort key
        # Or just shuffle the small boxes.
        
        # Strategy:
        # 1. Sort by volume descending
        # 2. For each box, try all rotations
        # 3. Place at best-fit position
        
        heights = np.zeros((W, H), dtype=int)
        placements = [None] * len(boxes)
        placed_vol = 0
        
        # We will iterate through boxes in sorted order
        # But to add diversity, we can randomize the order of boxes with similar volumes
        # Or just shuffle the entire list and rely on the placement heuristic.
        # Let's stick to sorted order for stability, but randomize the tie-breaking.
        
        # To add diversity, let's shuffle the indices of boxes that have the same volume
        # Or just shuffle the whole list and sort by volume with a random tie-breaker.
        
        random.seed(42 + restart)
        
        # Create a list of (volume, random_key, index)
        sort_keys = []
        for i in box_indices:
            vol = boxes[i][0] * boxes[i][1] * boxes[i][2]
            rand_key = random.random()
            sort_keys.append((vol, rand_key, i))
        
        # Sort by volume desc, then random key asc
        sort_keys.sort(key=lambda x: (-x[0], x[1]))
        order = [k[2] for k in sort_keys]
        
        for idx in order:
            if time.time() > deadline - 2.0:
                break
            
            w, h, d = boxes[idx]
            rots = box_rots[idx]
            
            best_pos = None
            best_rot = None
            best_score = None
            
            # Try each rotation
            for rw, rh, rd in rots:
                # Check if rotation fits in container
                if rw > W or rh > D or rd > H:
                    continue
                
                # Try to place
                # We need a copy of heights to test placement
                # But _try_place modifies heights.
                # So we need to save/restore or use a copy.
                # Copying a numpy array is fast.
                h_copy = heights.copy()
                pos = _try_place(h_copy, W, H, D, rw, rh, rd, 1)
                
                if pos is not None:
                    x, y, z = pos
                    # Score: prefer lower z, then lower x, then lower y
                    # Also prefer rotations that leave more space?
                    # For now, just use (z, x, y)
                    score = (z, x, y)
                    if best_score is None or score < best_score:
                        best_score = score
                        best_pos = pos
                        best_rot = (rw, rh, rd)
                        # Update heights with this placement
                        heights = h_copy
            
            if best_pos is not None:
                x, y, z = best_pos
                rw, rh, rd = best_rot
                placements[idx] = (0, x, y, z, rw, rh, rd)
                placed_vol += rw * rh * rd
            else:
                placements[idx] = (-1, 0, 0, 0, w, h, d)
        
        # If we didn't break due to time, we have a complete placement
        if placements[0] is not None:  # Check if all are placed
            if placed_vol > best_vol:
                best_vol = placed_vol
                best_placements = placements.copy()
        
        # If time is running out, break
        if time.time() > deadline - 5.0:
            break
    
    # If no placement was found (e.g., all boxes too large), return all as left out
    if best_placements is None:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    return best_placements
# EVOLVE-BLOCK-END
