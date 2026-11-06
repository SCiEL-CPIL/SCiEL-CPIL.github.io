"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
# EVOLVE-BLOCK-START

def pack(boxes, container):
    """
    3D Bin Packing using a Best-Fit Decreasing (BFD) heuristic with
    a simplified empty-space management per bin.
    """
    W, H, D = container
    container_volume = W * H * D
    
    # Sort boxes by volume descending
    indexed_boxes = [(w, h, d, i) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[0]*x[1]*x[2], reverse=True)
    
    # Each bin has a list of occupied rectangles and a list of empty spaces
    # We use a simple approach: for each bin, maintain a list of "slots"
    # A slot is (x, y, z, remaining_w, remaining_h, remaining_d)
    # But this is complex. Let's use a simpler approach:
    # For each bin, maintain a list of placed boxes and check for overlap.
    
    bins = []  # list of lists of (x, y, z, w, h, d)
    
    def can_place(bin_boxes, x, y, z, w, h, d):
        # Check if box fits in container
        if x + w > W or y + h > H or z + d > D:
            return False
        # Check overlap with existing boxes
        for bx, by, bz, bw, bh, bd in bin_boxes:
            # Overlap if intervals overlap in all three dimensions
            if (x < bx + bw and x + w > bx and
                y < by + bh and y + h > by and
                z < bz + bd and z + d > bz):
                return False
        return True
    
    def find_position(bin_boxes, w, h, d):
        # Find the best position (minimize x, then y, then z)
        # Try all integer positions? That's too slow.
        # Instead, try to place at origin first, then shift.
        # Better: use a grid of candidate positions.
        # For simplicity, try placing at (0,0,0), then scan.
        # Actually, let's use a heuristic: try to place in the "lowest" position.
        # We'll iterate over possible x, y, z positions.
        # To make this efficient, we'll only consider positions that are
        # "supported" by existing boxes or the container floor/walls.
        
        # Candidate x positions: 0, and x + w for each existing box
        candidate_xs = [0]
        candidate_ys = [0]
        candidate_zs = [0]
        
        for bx, by, bz, bw, bh, bd in bin_boxes:
            if bx + bw <= W:
                candidate_xs.append(bx + bw)
            if by + bh <= H:
                candidate_ys.append(by + bh)
            if bz + bd <= D:
                candidate_zs.append(bz + bd)
        
        # Also add positions that are just inside the container
        candidate_xs = sorted(set(candidate_xs))
        candidate_ys = sorted(set(candidate_ys))
        candidate_zs = sorted(set(candidate_zs))
        
        best_pos = None
        best_score = float('inf')
        
        for x in candidate_xs:
            for y in candidate_ys:
                for z in candidate_zs:
                    if can_place(bin_boxes, x, y, z, w, h, d):
                        # Score: prefer lower z, then lower y, then lower x
                        score = z * 10000 + y * 100 + x
                        if score < best_score:
                            best_score = score
                            best_pos = (x, y, z)
        
        return best_pos
    
    for w, h, d, orig_idx in indexed_boxes:
        placed = False
        # Try to place in existing bins
        for b_idx, bin_boxes in enumerate(bins):
            pos = find_position(bin_boxes, w, h, d)
            if pos is not None:
                x, y, z = pos
                bin_boxes.append((x, y, z, w, h, d))
                placed = True
                break
        
        if not placed:
            # Start a new bin
            bins.append([(0, 0, 0, w, h, d)])
    
    # Build placements in original order
    placements = [None] * len(boxes)
    for b_idx, bin_boxes in enumerate(bins):
        for x, y, z, w, h, d in bin_boxes:
            # Find the original index
            for i, (ow, oh, od) in enumerate(boxes):
                if ow == w and oh == h and od == d and placements[i] is None:
                    placements[i] = (b_idx, x, y, z, w, h, d)
                    break
    
    # If any placement is still None (shouldn't happen), assign to a new bin
    for i in range(len(boxes)):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (len(bins), 0, 0, 0, w, h, d)
            bins.append([(0, 0, 0, w, h, d)])
    
    return placements

# EVOLVE-BLOCK-END
