"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import itertools

# EVOLVE-BLOCK-START
def _2d_shelf_pack(items, W, D):
    """Pack 2D rectangles (w,d) into a W x D area using shelf heuristic.
    items: list of (w, d). Returns list of (w, d, x, z) placed (z is depth coord).
    """
    shelves = []  # each shelf: (y_start, y_end, x_cursor)
    placed = []
    for w, d in items:
        # try to place on an existing shelf
        best_shelf = -1
        best_gap = None
        for i, (ys, ye, xc) in enumerate(shelves):
            if d <= (ye - ys) and xc + w <= W:
                # fits on this shelf
                gap = (ye - ys) - d
                if best_shelf == -1 or gap < best_gap:
                    best_shelf = i
                    best_gap = gap
        if best_shelf != -1:
            ys, ye, xc = shelves[best_shelf]
            placed.append((w, d, xc, ys))
            shelves[best_shelf] = (ys, ye, xc + w)
        else:
            # start a new shelf
            if d > D:
                continue
            ys = 0
            # find highest empty y where a new shelf of height d fits
            # simple: stack shelves from bottom
            top = 0
            for (ys2, ye2, _) in shelves:
                top = max(top, ye2)
            if top + d > D:
                continue
            shelves.append((top, top + d, 0))
            placed.append((w, d, 0, top))
    return placed


def _pack_slab(boxes_for_slab, W, D, slab_h):
    """Try to pack boxes into one slab of height slab_h.
    boxes_for_slab: list of (w, h, d) original dims.
    Returns (placed_list, used_volume) where placed_list is list of (w,h,d,x,y,z)
    with y being the slab-local vertical offset (0..slab_h) and z the depth.
    We choose orientation so that the box height (vertical) <= slab_h.
    """
    # For each box, pick the orientation that minimizes vertical extent while fitting
    # We want to minimize the "height" used in the slab (the dimension along container H).
    # Try all 6 rotations, pick the one with smallest height that is <= slab_h,
    # breaking ties by smallest area.
    oriented = []
    for (w, h, d) in boxes_for_slab:
        best = None
        for (ow, oh, od) in itertools.permutations((w, h, d)):
            if oh <= slab_h:
                # candidate: footprint (ow, od), vertical oh
                area = ow * od
                if best is None or oh < best[0] or (oh == best[0] and area < best[1]):
                    best = (oh, area, ow, od)
        if best is not None:
            oh, area, ow, od = best
            oriented.append((ow, od, oh))
    if not oriented:
        return [], 0.0
    # 2D shelf pack on (W, D) using footprints (ow, od)
    placed2d = _2d_shelf_pack([(ow, od) for (ow, od, oh) in oriented], W, D)
    # Map back: placed2d gives (w, d, x, z). We need to know which box it was.
    # Since _2d_shelf_pack preserves order of items that fit, we need to track.
    # Let's redo with tracking.
    return _pack_slab_tracked(oriented, W, D, slab_h)


def _pack_slab_tracked(oriented, W, D, slab_h):
    """oriented: list of (ow, od, oh). Returns list of (ow, oh, od, x, 0, z) for placed boxes."""
    shelves = []  # (y_start, y_end, x_cursor)
    placed = []
    for idx, (ow, od, oh) in enumerate(oriented):
        best_shelf = -1
        best_gap = None
        for i, (ys, ye, xc) in enumerate(shelves):
            if od <= (ye - ys) and xc + ow <= W:
                gap = (ye - ys) - od
                if best_shelf == -1 or gap < best_gap:
                    best_shelf = i
                    best_gap = gap
        if best_shelf != -1:
            ys, ye, xc = shelves[best_shelf]
            placed.append((ow, oh, od, xc, ys, 0))  # (w, h, d, x, y, z)
            shelves[best_shelf] = (ys, ye, xc + ow)
        else:
            top = 0
            for (ys2, ye2, _) in shelves:
                top = max(top, ye2)
            if top + od > D:
                continue
            shelves.append((top, top + od, 0))
            placed.append((ow, oh, od, 0, top, 0))
    return placed


def _try_slab_height(boxes, W, H, D, slab_h):
    """Pack boxes into slabs of height slab_h. Returns (placements, total_volume)."""
    placements = [(-1, 0, 0, 0, w, h, d) for (w, h, d) in boxes]
    remaining = list(range(len(boxes)))
    total_vol = 0.0
    z_offset = 0
    # Sort remaining by volume descending for better packing
    order = sorted(remaining, key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    # Process in batches of slab height
    idx_in_order = 0
    while z_offset + slab_h <= H and idx_in_order < len(order):
        # Collect boxes for this slab
        slab_boxes = []
        slab_indices = []
        while idx_in_order < len(order):
            bi = order[idx_in_order]
            w, h, d = boxes[bi]
            # Check if box can fit in slab height
            can_fit = False
            for (ow, oh, od) in itertools.permutations((w, h, d)):
                if oh <= slab_h:
                    can_fit = True
                    break
            if can_fit:
                slab_boxes.append((w, h, d))
                slab_indices.append(bi)
                idx_in_order += 1
            else:
                # skip this box for now, mark as left out
                placements[bi] = (-1, 0, 0, 0, w, h, d)
                idx_in_order += 1
        if not slab_boxes:
            break
        # Pack this slab
        oriented = []
        for (w, h, d) in slab_boxes:
            best = None
            for (ow, oh, od) in itertools.permutations((w, h, d)):
                if oh <= slab_h:
                    area = ow * od
                    if best is None or oh < best[0] or (oh == best[0] and area < best[1]):
                        best = (oh, area, ow, od)
            if best is not None:
                oh, area, ow, od = best
                oriented.append((ow, od, oh))
        placed = _pack_slab_tracked(oriented, W, D, slab_h)
        # placed has same length as oriented? No, only placed ones.
        # We need to map placed back to slab_indices. Since _pack_slab_tracked
        # may skip some, we need to track which ones were placed.
        # Let's redo with explicit tracking.
        placed2 = _pack_slab_with_tracking(oriented, W, D, slab_h)
        for j, (ow, oh, od, x, y, z) in enumerate(placed2):
            bi = slab_indices[j]
            placements[bi] = (0, x, y, z_offset + y, ow, oh, od)
            total_vol += ow * oh * od
        z_offset += slab_h
    return placements, total_vol


def _pack_slab_with_tracking(oriented, W, D, slab_h):
    """oriented: list of (ow, od, oh). Returns list of (ow, oh, od, x, y, z) for placed boxes,
    in the same order as input (only those that fit)."""
    shelves = []
    placed = []
    for idx, (ow, od, oh) in enumerate(oriented):
        best_shelf = -1
        best_gap = None
        for i, (ys, ye, xc) in enumerate(shelves):
            if od <= (ye - ys) and xc + ow <= W:
                gap = (ye - ys) - od
                if best_shelf == -1 or gap < best_gap:
                    best_shelf = i
                    best_gap = gap
        if best_shelf != -1:
            ys, ye, xc = shelves[best_shelf]
            placed.append((ow, oh, od, xc, ys, 0))
            shelves[best_shelf] = (ys, ye, xc + ow)
        else:
            top = 0
            for (ys2, ye2, _) in shelves:
                top = max(top, ye2)
            if top + od > D:
                placed.append(None)
            else:
                shelves.append((top, top + od, 0))
                placed.append((ow, oh, od, 0, top, 0))
    return [p for p in placed if p is not None]


def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    # Try a range of slab heights
    # Candidate slab heights: divisors-like values, and box heights
    slab_candidates = set()
    # Add box heights and their permutations
    for (w, h, d) in boxes:
        for dim in (w, h, d):
            if dim <= H:
                slab_candidates.add(dim)
    # Add some standard values
    for sh in [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 100]:
        if sh <= H:
            slab_candidates.add(sh)
    # Also try H itself (single slab)
    slab_candidates.add(H)
    
    best_placements = [(-1, 0, 0, 0, w, h, d) for (w, h, d) in boxes]
    best_vol = 0.0
    
    # Limit number of slab candidates to avoid timeout
    slab_list = sorted(slab_candidates)
    if len(slab_list) > 30:
        # Sample evenly
        step = len(slab_list) / 30
        slab_list = [slab_list[int(i * step)] for i in range(30)]
    
    for slab_h in slab_list:
        placements, vol = _try_slab_height(boxes, W, H, D, slab_h)
        if vol > best_vol:
            best_vol = vol
            best_placements = placements
    
    return best_placements
# EVOLVE-BLOCK-END
