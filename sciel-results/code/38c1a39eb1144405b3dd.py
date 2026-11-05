"""Improved 3D bin packing with First-Fit Decreasing and deadline guard."""
import time

# EVOLVE-BLOCK-START
def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    deadline = time.time() + 110.0  # 10s safety margin
    
    # Sort boxes by volume descending (FDD heuristic)
    indexed_boxes = list(enumerate(boxes))
    indexed_boxes.sort(key=lambda x: x[1][0] * x[1][1] * x[1][2], reverse=True)
    
    # Bins will store list of (x, y, z, w, h, d)
    bins = []
    placement_map = {}  # original index -> (bin_idx, x, y, z)
    
    for orig_idx, (w, h, d) in indexed_boxes:
        if time.time() > deadline:
            # Fallback: place remaining boxes in a simple shelf pattern in new bins
            for remaining_orig_idx, (rw, rh, rd) in indexed_boxes:
                if remaining_orig_idx not in placement_map:
                    # Find next empty bin or create new
                    placed = False
                    for b_idx, bin_contents in enumerate(bins):
                        # Try shelf placement
                        pos = _try_shelf_place(bin_contents, W, H, D, rw, rh, rd)
                        if pos:
                            x, y, z = pos
                            placement_map[remaining_orig_idx] = (b_idx, x, y, z)
                            bin_contents.append((x, y, z, rw, rh, rd))
                            placed = True
                            break
                    if not placed:
                        b_idx = len(bins)
                        bins.append([(0, 0, 0, rw, rh, rd)])
                        placement_map[remaining_orig_idx] = (b_idx, 0, 0, 0)
            break
        
        placed = False
        for b_idx, bin_contents in enumerate(bins):
            # Try to place in this bin using best-fit position search
            pos = _find_position(bin_contents, W, H, D, w, h, d)
            if pos is not None:
                x, y, z = pos
                placement_map[orig_idx] = (b_idx, x, y, z)
                bin_contents.append((x, y, z, w, h, d))
                placed = True
                break
        
        if not placed:
            # Create new bin
            b_idx = len(bins)
            bins.append([(0, 0, 0, w, h, d)])
            placement_map[orig_idx] = (b_idx, 0, 0, 0)
    
    # Build result in original order
    result = []
    for i in range(n):
        b_idx, x, y, z = placement_map[i]
        w, h, d = boxes[i]
        result.append((b_idx, x, y, z, w, h, d))
    
    return result

def _find_position(bin_contents, W, H, D, w, h, d):
    """Find the best (lowest z, then y, then x) position for a box in a bin."""
    best_pos = None
    best_score = None
    
    # Candidate x positions: 0 and right edges of existing boxes
    x_candidates = [0]
    for (bx, by, bz, bw, bh, bd) in bin_contents:
        x_candidates.append(bx + bw)
    x_candidates = sorted(set(x for x in x_candidates if x <= W - w))
    
    # Candidate y positions: 0 and top edges of existing boxes
    y_candidates = [0]
    for (bx, by, bz, bw, bh, bd) in bin_contents:
        y_candidates.append(by + bh)
    y_candidates = sorted(set(y for y in y_candidates if y <= H - h))
    
    # Candidate z positions: 0 and front edges of existing boxes
    z_candidates = [0]
    for (bx, by, bz, bw, bh, bd) in bin_contents:
        z_candidates.append(bz + bd)
    z_candidates = sorted(set(z for z in z_candidates if z <= D - d))
    
    for x in x_candidates:
        for y in y_candidates:
            for z in z_candidates:
                if _fits(bin_contents, x, y, z, w, h, d, W, H, D):
                    score = (z, y, x)
                    if best_score is None or score < best_score:
                        best_score = score
                        best_pos = (x, y, z)
    
    return best_pos

def _fits(bin_contents, x, y, z, w, h, d, W, H, D):
    """Check if a box at (x,y,z) with dims (w,h,d) fits without overlap."""
    if x + w > W or y + h > H or z + d > D:
        return False
    for (bx, by, bz, bw, bh, bd) in bin_contents:
        # Check overlap
        if not (x + w <= bx or bx + bw <= x or
                y + h <= by or by + bh <= y or
                z + d <= bz or bz + bd <= z):
            return False
    return True

def _try_shelf_place(bin_contents, W, H, D, w, h, d):
    """Simple shelf placement for fallback."""
    x, y, z, row_h, layer_d = 0, 0, 0, 0, 0
    for (bx, by, bz, bw, bh, bd) in bin_contents:
        if x + w > W:
            x, y, row_h = 0, y + row_h, 0
        if y + h > H:
            y, z, layer_d = 0, z + layer_d, 0
            x, row_h = 0, 0
        if z + d > D:
            return None
        # Check if this position works
        if _fits(bin_contents, x, y, z, w, h, d, W, H, D):
            return (x, y, z)
        x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)
    return None
# EVOLVE-BLOCK-END
