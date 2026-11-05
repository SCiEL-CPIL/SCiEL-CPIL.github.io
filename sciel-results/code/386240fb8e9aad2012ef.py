"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Deadline guard: 150s wall-clock limit, use 130s as safety margin
    start_time = time.time()
    deadline = start_time + 130.0
    
    # Sort boxes by volume descending (Best Fit Decreasing)
    indexed_boxes = [(i, w, h, d, w*h*d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[4], reverse=True)
    
    # Each bin is a list of placed boxes: (x, y, z, w, h, d)
    bins = []
    
    def get_corners(bin_placements):
        """Generate candidate corner positions for a new box in a bin.
        Corners are derived from existing box faces."""
        if not bin_placements:
            return [(0, 0, 0)]
        
        xs = set([0])
        ys = set([0])
        zs = set([0])
        
        for (x, y, z, w, h, d) in bin_placements:
            xs.add(x)
            xs.add(x + w)
            ys.add(y)
            ys.add(y + h)
            zs.add(z)
            zs.add(z + d)
        
        # Filter to valid positions within container
        xs = [x for x in xs if x + 1 <= W]  # approximate filter
        ys = [y for y in ys if y + 1 <= H]
        zs = [z for z in zs if z + 1 <= D]
        
        corners = set()
        for x in xs:
            for y in ys:
                for z in zs:
                    corners.add((x, y, z))
        
        return list(corners)
    
    def can_place(bin_placements, x, y, z, w, h, d):
        """Check if a box can be placed at (x, y, z) in the given bin."""
        if x + w > W or y + h > H or z + d > D:
            return False
        for (px, py, pz, pw, ph, pd) in bin_placements:
            # Check overlap
            if not (x + w <= px or px + pw <= x or
                    y + h <= py or py + ph <= y or
                    z + d <= pz or pz + pd <= z):
                return False
        return True
    
    def score_position(x, y, z, w, h, d):
        """Score a position - lower is better (tighter fit).
        Prefer positions that minimize the sum of distances from origin."""
        return x + y + z
    
    def best_fit_position(bin_placements, w, h, d):
        """Find the best position for a box in a bin using best-fit."""
        corners = get_corners(bin_placements)
        best_pos = None
        best_score = float('inf')
        
        for (x, y, z) in corners:
            if can_place(bin_placements, x, y, z, w, h, d):
                s = score_position(x, y, z, w, h, d)
                if s < best_score:
                    best_score = s
                    best_pos = (x, y, z)
        
        return best_pos
    
    # Place each box
    for i, w, h, d, vol in indexed_boxes:
        if time.time() > deadline:
            break
        
        best_bin = -1
        best_pos = None
        best_score = float('inf')
        
        # Try to place in existing bins (best fit)
        for b_idx, bin_placements in enumerate(bins):
            pos = best_fit_position(bin_placements, w, h, d)
            if pos is not None:
                s = score_position(pos[0], pos[1], pos[2], w, h, d)
                if s < best_score:
                    best_score = s
                    best_bin = b_idx
                    best_pos = pos
        
        # If no existing bin works, open a new bin
        if best_bin == -1:
            best_bin = len(bins)
            bins.append([])
            best_pos = (0, 0, 0)
        
        # Place the box
        x, y, z = best_pos
        bins[best_bin].append((x, y, z, w, h, d))
    
    # Build result
    # We need to map back to original box indices
    # Create a placement list where each entry is (bin, x, y, z, w, h, d)
    # indexed by original box order
    
    # First, collect placements by original index
    placements_by_index = [None] * n
    
    # Re-process: we need to track which original index went to which bin/position
    # Let's redo this more carefully
    
    # Actually, let me restructure: track original index with each placement
    bins2 = []
    
    indexed_boxes2 = [(i, w, h, d, w*h*d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes2.sort(key=lambda x: x[4], reverse=True)
    
    for i, w, h, d, vol in indexed_boxes2:
        if time.time() > deadline:
            # If we've hit the deadline, place remaining boxes in a simple way
            # to ensure all boxes are placed
            pass
        
        best_bin = -1
        best_pos = None
        best_score = float('inf')
        
        for b_idx, bin_placements in enumerate(bins2):
            pos = best_fit_position(bin_placements, w, h, d)
            if pos is not None:
                s = score_position(pos[0], pos[1], pos[2], w, h, d)
                if s < best_score:
                    best_score = s
                    best_bin = b_idx
                    best_pos = pos
        
        if best_bin == -1:
            best_bin = len(bins2)
            bins2.append([])
            best_pos = (0, 0, 0)
        
        x, y, z = best_pos
        bins2[best_bin].append((i, x, y, z, w, h, d))
    
    # Build final placements list in original order
    result = [None] * n
    for b_idx, bin_placements in enumerate(bins2):
        for (orig_idx, x, y, z, w, h, d) in bin_placements:
            result[orig_idx] = (b_idx, x, y, z, w, h, d)
    
    # If any boxes weren't placed (due to deadline), place them in a new bin
    # using a simple shelf approach as fallback
    for idx in range(n):
        if result[idx] is None:
            w, h, d = boxes[idx]
            # Find a bin where it fits, or create new one
            placed = False
            for b_idx in range(len(bins2)):
                pos = best_fit_position(bins2[b_idx], w, h, d)
                if pos is not None:
                    x, y, z = pos
                    bins2[b_idx].append((idx, x, y, z, w, h, d))
                    result[idx] = (b_idx, x, y, z, w, h, d)
                    placed = True
                    break
            if not placed:
                new_bin = len(bins2)
                bins2.append([(idx, 0, 0, 0, w, h, d)])
                result[idx] = (new_bin, 0, 0, 0, w, h, d)
    
    return result
# EVOLVE-BLOCK-END
