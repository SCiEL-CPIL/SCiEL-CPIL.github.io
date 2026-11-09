"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import heapq

# EVOLVE-BLOCK-START

def _overlaps(p1, p2):
    """Check if two boxes (x, y, z, w, h, d) overlap (strict interior)."""
    x1, y1, z1, w1, h1, d1 = p1
    x2, y2, z2, w2, h2, d2 = p2
    return not (x1 + w1 <= x2 or x2 + w2 <= x1 or
                y1 + h1 <= y2 or y2 + h2 <= y1 or
                z1 + d1 <= z2 or z2 + d2 <= z1)


def _fits_in_container(x, y, z, w, h, d, W, H, D):
    return x >= 0 and y >= 0 and z >= 0 and x + w <= W and y + h <= H and z + d <= D


def _candidate_positions(existing, W, H, D, w, h, d):
    """Generate candidate positions to try, prioritizing bottom-left-front."""
    # Always try origin first
    candidates = [(0, 0, 0)]
    # Generate positions based on existing boxes:
    # For each existing box, the new box can be placed just to the right, above, or in front.
    for (ex, ey, ez, ew, eh, ed) in existing:
        # Right of existing box
        candidates.append((ex + ew, ey, ez))
        # Above existing box
        candidates.append((ex, ey + eh, ez))
        # In front of existing box
        candidates.append((ex, ey, ez + ed))
        # Also consider combinations: right and above, right and front, above and front
        candidates.append((ex + ew, ey + eh, ez))
        candidates.append((ex + ew, ey, ez + ed))
        candidates.append((ex, ey + eh, ez + ed))
        # All three
        candidates.append((ex + ew, ey + eh, ez + ed))
    
    # Deduplicate and filter
    seen = set()
    unique_candidates = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique_candidates.append(c)
    
    # Sort by (z, y, x) to prioritize bottom, then left, then front
    unique_candidates.sort(key=lambda p: (p[2], p[1], p[0]))
    
    return unique_candidates


def _try_place_box(box_idx, boxes, bin_placements, W, H, D, deadline):
    """Try to place a box in an existing bin. Returns (bin_idx, x, y, z) or None."""
    w, h, d = boxes[box_idx]
    
    for bin_idx, existing in bin_placements.items():
        if time.time() > deadline:
            return None
        
        candidates = _candidate_positions(existing, W, H, D, w, h, d)
        
        for (x, y, z) in candidates:
            if not _fits_in_container(x, y, z, w, h, d, W, H, D):
                continue
            
            # Check overlap with all existing boxes in this bin
            overlap = False
            for ep in existing:
                if _overlaps((x, y, z, w, h, d), ep):
                    overlap = True
                    break
            if not overlap:
                return (bin_idx, x, y, z)
    
    return None


def _score_placement(placements, boxes, container):
    """Calculate the score: mean B + 0.1*(1 - last-bin occupied volume fraction)."""
    W, H, D = container
    container_vol = W * H * D
    
    if not placements:
        return float('inf')
    
    # Count occupied bins
    bin_volumes = {}
    for (b, x, y, z, w, h, d) in placements:
        if b not in bin_volumes:
            bin_volumes[b] = 0
        bin_volumes[b] += w * h * d
    
    num_bins = len(bin_volumes)
    
    # Find last bin (largest label)
    last_bin = max(bin_volumes.keys())
    last_bin_vol = bin_volumes[last_bin]
    
    score = num_bins + 0.1 * (1 - last_bin_vol / container_vol)
    return score


def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    
    if n == 0:
        return []
    
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Sort boxes by volume descending for better packing
    indexed_boxes = list(enumerate(boxes))
    indexed_boxes.sort(key=lambda x: x[1][0] * x[1][1] * x[1][2], reverse=True)
    
    # Greedy Best-Fit Decreasing
    bin_placements = {}  # bin_idx -> list of (x, y, z, w, h, d)
    bin_order = []  # order of bins created
    placements = [None] * n  # final placements indexed by original box index
    
    for orig_idx, (w, h, d) in indexed_boxes:
        if time.time() > deadline:
            # Fallback: just place remaining boxes in a new bin each
            for remaining_idx, (rw, rh, rd) in indexed_boxes:
                if placements[remaining_idx] is None:
                    new_bin = len(bin_placements)
                    bin_placements[new_bin] = []
                    bin_order.append(new_bin)
                    bin_placements[new_bin].append((0, 0, 0, rw, rh, rd))
                    placements[remaining_idx] = (new_bin, 0, 0, 0, rw, rh, rd)
            break
        
        # Try to place in existing bins
        result = _try_place_box(orig_idx, boxes, bin_placements, W, H, D, deadline)
        
        if result is not None:
            bin_idx, x, y, z = result
            bin_placements[bin_idx].append((x, y, z, w, h, d))
            placements[orig_idx] = (bin_idx, x, y, z, w, h, d)
        else:
            # Create a new bin
            new_bin = len(bin_placements)
            bin_placements[new_bin] = [(0, 0, 0, w, h, d)]
            bin_order.append(new_bin)
            placements[orig_idx] = (new_bin, 0, 0, 0, w, h, d)
    
    # If we hit the deadline, make sure all placements are filled
    for i in range(n):
        if placements[i] is None:
            # Place in a new bin at origin (may not be valid, but avoids crash)
            new_bin = len(bin_placements)
            bin_placements[new_bin] = []
            bin_order.append(new_bin)
            w, h, d = boxes[i]
            bin_placements[new_bin].append((0, 0, 0, w, h, d))
            placements[i] = (new_bin, 0, 0, 0, w, h, d)
    
    return placements

# EVOLVE-BLOCK-END
