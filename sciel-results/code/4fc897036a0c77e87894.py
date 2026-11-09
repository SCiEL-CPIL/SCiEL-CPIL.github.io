"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
import heapq
from collections import defaultdict

def _fits_in_bin(box, x, y, z, W, H, D):
    """Check if box of dimensions (w,h,d) fits at (x,y,z) in container (W,H,D)."""
    w, h, d = box
    return (x + w <= W and y + h <= H and z + d <= D)

def _can_place(box, x, y, z, W, H, D, occupied):
    """Check if box can be placed at (x,y,z) without overlapping any occupied cell."""
    w, h, d = box
    # Quick bounds check
    if x + w > W or y + h > H or z + d > D:
        return False
    # Check overlap with occupied boxes
    for (ox, oy, oz, ow, oh, od) in occupied:
        # Check if the new box [x, x+w) x [y, y+h) x [z, z+d) overlaps with
        # the occupied box [ox, ox+ow) x [oy, oy+oh) x [oz, oz+od)
        if (x < ox + ow and ox < x + w and
            y < oy + oh and oy < y + h and
            z < oz + od and oz < z + d):
            return False
    return True

def _best_fit_position(box, W, H, D, occupied, prefer_low=True):
    """Find the best position for a box in a bin using best-fit (min z, then y, then x)."""
    w, h, d = box
    best_pos = None
    best_score = None
    
    # Generate candidate positions: try all valid (x, y, z) positions
    # For efficiency, we only need to try positions aligned to existing box edges or 0
    # But for correctness with arbitrary sizes, let's try a grid of candidates
    
    # Candidate x positions: 0 and right edges of existing boxes
    x_candidates = [0]
    y_candidates = [0]
    z_candidates = [0]
    
    for (ox, oy, oz, ow, oh, od) in occupied:
        if ox + ow <= W:
            x_candidates.append(ox + ow)
        if oy + oh <= H:
            y_candidates.append(oy + oh)
        if oz + od <= D:
            z_candidates.append(oz + od)
    
    # Also add 0 if not present
    if 0 not in x_candidates:
        x_candidates.append(0)
    if 0 not in y_candidates:
        y_candidates.append(0)
    if 0 not in z_candidates:
        z_candidates.append(0)
    
    # Sort candidates for deterministic ordering
    x_candidates = sorted(set(x_candidates))
    y_candidates = sorted(set(y_candidates))
    z_candidates = sorted(set(z_candidates))
    
    for z in z_candidates:
        if z + d > D:
            continue
        for y in y_candidates:
            if y + h > H:
                continue
            for x in x_candidates:
                if x + w > W:
                    continue
                if _can_place(box, x, y, z, W, H, D, occupied):
                    # Score: prefer lower z, then lower y, then lower x
                    score = (z, y, x)
                    if best_score is None or score < best_score:
                        best_score = score
                        best_pos = (x, y, z)
    
    return best_pos

def _best_fit_bin(box, bins, W, H, D):
    """Find the best bin and position for a box using best-fit across all bins."""
    best_bin = None
    best_pos = None
    best_score = None
    
    for b in range(len(bins)):
        occupied = bins[b]
        pos = _best_fit_position(box, W, H, D, occupied, prefer_low=True)
        if pos is not None:
            x, y, z = pos
            score = (z, y, x)
            if best_score is None or score < best_score:
                best_score = score
                best_bin = b
                best_pos = pos
    
    return best_bin, best_pos

def _try_repack_box(box, bins, W, H, D, exclude_bin=None):
    """Try to find a position for a box in any bin (excluding exclude_bin if set)."""
    for b in range(len(bins)):
        if b == exclude_bin:
            continue
        occupied = bins[b]
        pos = _best_fit_position(box, W, H, D, occupied)
        if pos is not None:
            return b, pos
    return None, None

def _remove_box_from_bin(box, bins, W, H, D):
    """Remove a box from all bins. Returns the bin index it was removed from, or -1 if not found."""
    for b in range(len(bins)):
        for i, (x, y, z, w, h, d) in enumerate(bins[b]):
            if (w, h, d) == box:
                bins[b].pop(i)
                return b
    return -1

def _add_box_to_bin(box, x, y, z, bins, W, H, D, bin_idx):
    """Add a box to a specific bin."""
    bins[bin_idx].append((x, y, z) + box)

def _compute_score(bins, W, H, D):
    """Compute the score: B + 0.1 * (1 - last_bin_occupied_volume / container_volume)."""
    if not bins:
        return 0.0
    
    B = len(bins)
    container_volume = W * H * D
    
    # Find the last bin (highest index that has boxes)
    last_bin_idx = max(b for b in range(len(bins)) if bins[b])
    last_bin_volume = sum(w * h * d for (x, y, z, w, h, d) in bins[last_bin_idx])
    
    return B + 0.1 * (1 - last_bin_volume / container_volume)

def _sort_boxes_ffd(boxes):
    """Sort boxes by First-Fit Decreasing: largest volume first, then largest max dimension."""
    return sorted(boxes, key=lambda b: (-b[0]*b[1]*b[2], -max(b)))

def pack(boxes, container):
    """Pack boxes into bins using best-fit decreasing with local search improvement."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin
    
    W, H, D = container
    n = len(boxes)
    
    if n == 0:
        return []
    
    # Sort boxes using FFD
    sorted_boxes = _sort_boxes_ffd(list(boxes))
    
    # Initial packing using best-fit decreasing
    bins = []  # Each bin is a list of (x, y, z, w, h, d)
    
    for box in sorted_boxes:
        if time.time() > deadline:
            break
        
        best_bin, best_pos = _best_fit_bin(box, bins, W, H, D)
        
        if best_bin is None:
            # Try to place in a new bin
            bins.append([])
            best_bin = len(bins) - 1
            # Find position in the new (empty) bin
            best_pos = _best_fit_position(box, W, H, D, [])
            if best_pos is None:
                # Shouldn't happen for an empty bin if box fits in container
                best_pos = (0, 0, 0)
        
        x, y, z = best_pos
        bins[best_bin].append((x, y, z) + box)
    
    # Local search: try to reduce the number of bins by moving boxes from the last bin
    # into other bins
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            B = len(bins)
            
            # Try to empty the last bin
            if B > 1:
                last_bin_idx = B - 1
                if bins[last_bin_idx]:
                    # Try to move each box from the last bin to another bin
                    boxes_to_try = list(bins[last_bin_idx])
                    for box in boxes_to_try:
                        if time.time() > deadline:
                            break
                        # Remove from last bin
                        bins[last_bin_idx].remove((0, 0, 0) + box if False else None)
                        # Actually, let's do this more carefully
                        # Find and remove the exact entry
                        removed = False
                        for i, entry in enumerate(bins[last_bin_idx]):
                            if entry[3:] == box:
                                bins[last_bin_idx].pop(i)
                                removed = True
                                break
                        if not removed:
                            continue
                        
                        # Try to place in another bin
                        new_bin, new_pos = _try_repack_box(box, bins, W, H, D, exclude_bin=last_bin_idx)
                        if new_bin is not None:
                            x, y, z = new_pos
                            bins[new_bin].append((x, y, z) + box)
                            improved = True
                        else:
                            # Put it back
                            bins[last_bin_idx].append((0, 0, 0) + box)
                    
                    # Remove empty last bin if it's empty
                    if not bins[last_bin_idx]:
                        bins.pop()
                        improved = True
    
    # Final local search: try to consolidate by moving boxes between bins
    if time.time() < deadline:
        # Try to move boxes from higher-index bins to lower-index bins to consolidate
        for _ in range(3):  # Limited passes
            if time.time() > deadline:
                break
            changed = False
            for b in range(len(bins) - 1, 0, -1):
                if time.time() > deadline:
                    break
                if not bins[b]:
                    continue
                boxes_in_bin = list(bins[b])
                for box in boxes_in_bin:
                    if time.time() > deadline:
                        break
                    # Try to move to a lower bin
                    for target_b in range(b):
                        if time.time() > deadline:
                            break
                        pos = _best_fit_position(box, W, H, D, bins[target_b])
                        if pos is not None:
                            # Remove from current bin
                            for i, entry in enumerate(bins[b]):
                                if entry[3:] == box:
                                    bins[b].pop(i)
                                    break
                            # Add to target bin
                            x, y, z = pos
                            bins[target_b].append((x, y, z) + box)
                            changed = True
                            break
                    if time.time() > deadline:
                        break
                # Remove empty bins
                bins = [b for b in bins if b]
            if not changed:
                break
    
    # Build the final placement list
    placements = []
    for b, bin_entries in enumerate(bins):
        for (x, y, z, w, h, d) in bin_entries:
            placements.append((b, x, y, z, w, h, d))
    
    return placements

# EVOLVE-BLOCK-END
