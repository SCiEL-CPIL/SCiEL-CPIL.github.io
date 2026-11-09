"""3D Bin Packing using Skyline/Maximal Rectangles per layer."""
import time
import math

# EVOLVE-BLOCK-START

def pack(boxes, container):
    """
    Place every box (no rotation) into numbered bins.
    Uses a skyline-based 3D packing heuristic with multiple ordering strategies.
    Returns list of (bin, x, y, z, w, h, d) per box.
    """
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []

    deadline = time.time() + 140.0  # safety margin before 150s kill

    # Precompute box volumes
    vols = [w * h * d for w, h, d in boxes]
    total_vol = sum(vols)
    cont_vol = W * H * D

    # Try several ordering strategies and pick the best result
    # Each strategy returns (placements, score)
    best_placements = None
    best_score = float('inf')

    strategies = [
        ('volume_desc', lambda i: -vols[i]),
        ('dim_sum_desc', lambda i: -(boxes[i][0] + boxes[i][1] + boxes[i][2])),
        ('max_dim_desc', lambda i: -max(boxes[i])),
        ('volume_asc', lambda i: vols[i]),
        ('dim_sum_asc', lambda i: boxes[i][0] + boxes[i][1] + boxes[i][2]),
        ('original', lambda i: i),
    ]

    for name, key_func in strategies:
        if time.time() > deadline:
            break

        order = sorted(range(n), key=key_func)

        placements = _pack_with_order(order, boxes, (W, H, D))

        if placements is None:
            continue

        score = _score(placements, boxes, (W, H, D))
        if score < best_score:
            best_score = score
            best_placements = placements

    if best_placements is None:
        # Fallback: simple shelf
        best_placements = _shelf_pack(boxes, (W, H, D))
        best_score = _score(best_placements, boxes, (W, H, D))

    return best_placements


def _score(placements, boxes, container):
    """Compute the score: mean B + 0.1*(1 - last-bin occupied volume fraction)."""
    W, H, D = container
    cont_vol = W * H * D
    if not placements:
        return 0.0

    bin_vols = {}
    for (b, x, y, z, w, h, d) in placements:
        bin_vols[b] = bin_vols.get(b, 0) + w * h * d

    B = len(bin_vols)
    last_bin = max(bin_vols.keys())
    last_vol = bin_vols[last_bin]
    frac = last_vol / cont_vol if cont_vol > 0 else 0.0
    score = B + 0.1 * (1.0 - frac)
    return score


def _shelf_pack(boxes, container):
    """Simple shelf packing as fallback."""
    W, H, D = container
    placements = []
    b, x, y, z, row_h, layer_d = 0, 0, 0, 0, 0, 0
    for w, h, d in boxes:
        if x + w > W:
            x, y, row_h = 0, y + row_h, 0
        if y + h > H:
            y, z, layer_d = 0, z + layer_d, 0
            x, row_h = 0, 0
        if z + d > D:
            b, x, y, z, row_h, layer_d = b + 1, 0, 0, 0, 0, 0
        placements.append((b, x, y, z, w, h, d))
        x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)
    return placements


def _pack_with_order(order, boxes, container):
    """
    Pack boxes in the given order using a skyline-based 3D approach.
    Each bin maintains a set of 'skyline' segments for efficient placement.
    """
    W, H, D = container
    n = len(boxes)

    # For each bin, maintain a list of placed boxes and a skyline structure
    # Skyline: list of (x_start, x_end, z_height) representing the top surface profile
    # We use a simpler approach: for each bin, keep a list of boxes and find best position
    # by checking all valid (x, y, z) positions derived from existing box boundaries.

    bins = []  # list of lists of (x, y, z, w, h, d)

    def _find_position(bin_boxes, w, h, d):
        """
        Find the best (x, y, z) position for a box of size (w, h, d) in a bin.
        Uses candidate positions derived from existing box edges.
        Returns (x, y, z) or None if no fit.
        """
        if not bin_boxes:
            return (0, 0, 0)

        # Generate candidate x, y, z positions
        # x candidates: 0, and right edges of existing boxes
        x_cands = [0]
        y_cands = [0]
        z_cands = [0]

        for (bx, by, bz, bw, bh, bd) in bin_boxes:
            x_cands.append(bx + bw)
            y_cands.append(by + bh)
            z_cands.append(bz + bd)

        x_cands = sorted(set(x for x in x_cands if x + w <= W))
        y_cands = sorted(set(y for y in y_cands if y + h <= H))
        z_cands = sorted(set(z for z in z_cands if z + d <= D))

        if not x_cands or not y_cands or not z_cands:
            return None

        # Also add 0 to be safe
        if 0 not in x_cands:
            x_cands.insert(0, 0)
        if 0 not in y_cands:
            y_cands.insert(0, 0)
        if 0 not in z_cands:
            z_cands.insert(0, 0)

        best_pos = None
        best_metric = None

        for x in x_cands:
            if x + w > W:
                continue
            for y in y_cands:
                if y + h > H:
                    continue
                for z in z_cands:
                    if z + d > D:
                        continue
                    # Check overlap with all existing boxes
                    if _overlaps(bin_boxes, x, y, z, w, h, d):
                        continue
                    # Metric: prefer low z, then low y, then low x (to keep bins compact)
                    metric = (z, y, x)
                    if best_metric is None or metric < best_metric:
                        best_metric = metric
                        best_pos = (x, y, z)

        return best_pos

    def _overlaps(bin_boxes, x, y, z, w, h, d):
        """Check if a box at (x,y,z) with size (w,h,d) overlaps any box in bin_boxes."""
        for (bx, by, bz, bw, bh, bd) in bin_boxes:
            if (x < bx + bw and x + w > bx and
                y < by + bh and y + h > by and
                z < bz + bd and z + d > bz):
                return True
        return False

    # Pack boxes in order
    placements = []

    for idx in order:
        w, h, d = boxes[idx]
        placed = False

        # Try existing bins first (in order of least fill to encourage consolidation? No, try all)
        for b_idx in range(len(bins)):
            pos = _find_position(bins[b_idx], w, h, d)
            if pos is not None:
                x, y, z = pos
                bins[b_idx].append((x, y, z, w, h, d))
                placements.append((b_idx, x, y, z, w, h, d))
                placed = True
                break

        if not placed:
            # Open a new bin
            b_idx = len(bins)
            bins.append([(0, 0, 0, w, h, d)])
            placements.append((b_idx, 0, 0, 0, w, h, d))

    return placements


# EVOLVE-BLOCK-END
