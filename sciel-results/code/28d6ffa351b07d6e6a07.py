"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import heapq
from itertools import product

# EVOLVE-BLOCK-START
def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    W, H, D = container
    n = len(boxes)
    
    # Sort boxes by volume descending (best-fit decreasing)
    indexed_boxes = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[1]*x[2]*x[3], reverse=True)
    
    # Each bin stores a list of placed boxes: (x, y, z, w, h, d)
    bins = []
    
    def boxes_overlap(x1, y1, z1, w1, h1, d1, x2, y2, z2, w2, h2, d2):
        return (x1 < x2 + w2 and x2 < x1 + w1 and
                y1 < y2 + h2 and y2 < y1 + h1 and
                z1 < z2 + d2 and z2 < z1 + d1)
    
    def find_placement(bin_boxes, w, h, d, W, H, D):
        """Find a valid (x, y, z) position in the bin. Returns None if impossible."""
        # Generate candidate positions: origin, and corners of existing boxes
        candidates = set()
        candidates.add((0, 0, 0))
        for (bx, by, bz, bw, bh, bd) in bin_boxes:
            # Positions just to the right, above, and in front of each box
            if bx + bw + w <= W:
                candidates.add((bx + bw, by, bz))
            if by + bh + h <= H:
                candidates.add((bx, by + bh, bz))
            if bz + bd + d <= D:
                candidates.add((bx, by, bz + bd))
            # Also try corners
            for dx in [bx + bw]:
                for dy in [by + bh]:
                    for dz in [bz + bd]:
                        if dx + w <= W and dy + h <= H and dz + d <= D:
                            candidates.add((dx, dy, dz))
        
        # Sort candidates by (z, y, x) to prefer lower positions
        candidates = sorted(candidates, key=lambda p: (p[2], p[1], p[0]))
        
        for (x, y, z) in candidates:
            if x + w > W or y + h > H or z + d > D:
                continue
            # Check overlap with all existing boxes
            valid = True
            for (bx, by, bz, bw, bh, bd) in bin_boxes:
                if boxes_overlap(x, y, z, w, h, d, bx, by, bz, bw, bh, bd):
                    valid = False
                    break
            if valid:
                return (x, y, z)
        
        return None
    
    def bin_volume(bin_boxes):
        return sum(w*h*d for (_, _, _, w, h, d) in bin_boxes)
    
    # Place each box
    placements = [None] * n  # (bin_idx, x, y, z, w, h, d)
    
    for orig_idx, w, h, d in indexed_boxes:
        if time.time() > deadline:
            # Return what we have; fill remaining with a simple fallback
            break
        
        best_bin = -1
        best_pos = None
        best_score = float('inf')
        
        # Try existing bins
        for bi, bin_boxes in enumerate(bins):
            pos = find_placement(bin_boxes, w, h, d, W, H, D)
            if pos is not None:
                # Score: prefer bin that results in least wasted space
                # Use the position's z+y+x as a proxy for tightness
                score = pos[2] * W * H + pos[1] * W + pos[0]
                if score < best_score:
                    best_score = score
                    best_bin = bi
                    best_pos = pos
        
        if best_bin == -1:
            # Open a new bin
            best_bin = len(bins)
            bins.append([])
            best_pos = (0, 0, 0)
        
        # Place the box
        x, y, z = best_pos
        bins[best_bin].append((x, y, z, w, h, d))
        placements[orig_idx] = (best_bin, x, y, z, w, h, d)
    
    # Handle any unplaced boxes with a simple shelf fallback
    if any(p is None for p in placements):
        # Simple shelf packing for remaining
        remaining = [i for i in range(n) if placements[i] is None]
        for i in remaining:
            w, h, d = boxes[i]
            # Find a bin or create new
            placed = False
            for bi, bin_boxes in enumerate(bins):
                pos = find_placement(bin_boxes, w, h, d, W, H, D)
                if pos is not None:
                    x, y, z = pos
                    bins[bi].append((x, y, z, w, h, d))
                    placements[i] = (bi, x, y, z, w, h, d)
                    placed = True
                    break
            if not placed:
                bi = len(bins)
                bins.append([(0, 0, 0, w, h, d)])
                placements[i] = (bi, 0, 0, 0, w, h, d)
    
    return placements
# EVOLVE-BLOCK-END
