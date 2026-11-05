"""3D Bin Packing: FDD + best-fit with skyline pruning and multi-pass refinement."""
import time

# EVOLVE-BLOCK-START
def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []

    deadline = time.time() + 110.0  # safety margin

    # Sort boxes by volume descending (FDD heuristic)
    indexed_boxes = list(enumerate(boxes))
    indexed_boxes.sort(key=lambda x: x[1][0] * x[1][1] * x[1][2], reverse=True)

    # Bins will store list of (x, y, z, w, h, d)
    bins = []
    placement_map = {}  # original index -> (bin_idx, x, y, z)

    for orig_idx, (w, h, d) in indexed_boxes:
        if time.time() > deadline:
            # Fallback: place remaining boxes greedily in new bins
            for remaining_orig_idx, (rw, rh, rd) in indexed_boxes:
                if remaining_orig_idx not in placement_map:
                    b_idx = len(bins)
                    bins.append([(0, 0, 0, rw, rh, rd)])
                    placement_map[remaining_orig_idx] = (b_idx, 0, 0, 0)
            break

        placed = False
        # Try existing bins first (first-fit with best position)
        for b_idx, bin_contents in enumerate(bins):
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

    # Refinement pass: try to improve by moving boxes to earlier bins or better positions
    if time.time() < deadline - 5.0:
        _refine(bins, placement_map, boxes, W, H, D, deadline)

    # Build result in original order
    result = []
    for i in range(n):
        b_idx, x, y, z = placement_map[i]
        w, h, d = boxes[i]
        result.append((b_idx, x, y, z, w, h, d))

    return result


def _find_position(bin_contents, W, H, D, w, h, d):
    """Find the best (lowest z, then y, then x) position for a box in a bin."""
    if not bin_contents:
        if w <= W and h <= H and d <= D:
            return (0, 0, 0)
        return None

    best_pos = None
    best_score = None

    # Candidate x positions: 0 and right edges of existing boxes
    x_candidates = {0}
    for (bx, by, bz, bw, bh, bd) in bin_contents:
        x_candidates.add(bx + bw)
    x_candidates = sorted(x for x in x_candidates if x + w <= W)

    # Candidate y positions: 0 and top edges of existing boxes
    y_candidates = {0}
    for (bx, by, bz, bw, bh, bd) in bin_contents:
        y_candidates.add(by + bh)
    y_candidates = sorted(y for y in y_candidates if y + h <= H)

    # Candidate z positions: 0 and front edges of existing boxes
    z_candidates = {0}
    for (bx, by, bz, bw, bh, bd) in bin_contents:
        z_candidates.add(bz + bd)
    z_candidates = sorted(z for z in z_candidates if z + d <= D)

    # Prune: only consider positions where the box can actually fit
    # Use a more efficient check by sorting candidates
    for z in z_candidates:
        for y in y_candidates:
            for x in x_candidates:
                if _fits(bin_contents, x, y, z, w, h, d):
                    score = (z, y, x)
                    if best_score is None or score < best_score:
                        best_score = score
                        best_pos = (x, y, z)
                        # Since we iterate z, y, x in ascending order,
                        # the first valid position is optimal
                        return best_pos

    return None


def _fits(bin_contents, x, y, z, w, h, d):
    """Check if a box at (x,y,z) with dims (w,h,d) fits without overlap."""
    for (bx, by, bz, bw, bh, bd) in bin_contents:
        if not (x + w <= bx or bx + bw <= x or
                y + h <= by or by + bh <= y or
                z + d <= bz or bz + bd <= z):
            return False
    return True


def _refine(bins, placement_map, boxes, W, H, D, deadline):
    """Try to move boxes to earlier bins or better positions to reduce bin count."""
    # For each bin from the last, try to move boxes to earlier bins
    max_b = len(bins) - 1
    for b_idx in range(max_b, -1, -1):
        if time.time() > deadline - 3.0:
            break
        if not bins[b_idx]:
            continue

        # Try to move each box in this bin to an earlier bin
        to_move = list(bins[b_idx])
        for box_data in to_move:
            if time.time() > deadline - 3.0:
                break
            bx, by, bz, bw, bh, bd = box_data
            # Find original index
            orig_idx = None
            for idx, (b, x, y, z) in placement_map.items():
                if b == b_idx and x == bx and y == by and z == bz:
                    orig_idx = idx
                    break
            if orig_idx is None:
                continue

            # Try to place in earlier bins
            moved = False
            for new_b in range(b_idx):
                pos = _find_position(bins[new_b], W, H, D, bw, bh, bd)
                if pos is not None:
                    x, y, z = pos
                    # Remove from current bin
                    bins[b_idx].remove(box_data)
                    # Add to new bin
                    bins[new_b].append((x, y, z, bw, bh, bd))
                    placement_map[orig_idx] = (new_b, x, y, z)
                    moved = True
                    break
            if moved:
                continue

            # Try to find a better position in the same bin
            # Remove and re-add
            bins[b_idx].remove(box_data)
            pos = _find_position(bins[b_idx], W, H, D, bw, bh, bd)
            if pos is not None:
                x, y, z = pos
                bins[b_idx].append((x, y, z, bw, bh, bd))
                placement_map[orig_idx] = (b_idx, x, y, z)
            else:
                # Put back
                bins[b_idx].append(box_data)
                placement_map[orig_idx] = (b_idx, bx, by, bz)

    # Remove empty bins from the end (relabel)
    while bins and not bins[-1]:
        bins.pop()
        # Relabel placement_map
        for idx, (b, x, y, z) in placement_map.items():
            if b >= len(bins):
                # Should not happen if we only remove empty last bins
                pass

    # Compact bin labels
    if bins:
        label_map = {}
        for old_label in range(len(bins)):
            if bins[old_label]:
                label_map[old_label] = len(label_map)
        for idx in placement_map:
            b = placement_map[idx][0]
            if b in label_map:
                placement_map[idx] = (label_map[b],) + placement_map[idx][1:]
            else:
                # Box was in an empty bin that got removed - shouldn't happen
                placement_map[idx] = (0,) + placement_map[idx][1:]
# EVOLVE-BLOCK-END
