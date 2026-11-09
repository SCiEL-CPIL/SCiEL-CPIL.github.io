"""3D Bin Packing: FFD with best-fit placement + local search refinement."""
import time

# EVOLVE-BLOCK-START
def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []

    container_vol = W * H * D

    # Sort boxes by volume descending (FFD)
    indexed_boxes = [(w * h * d, i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: -x[0])

    # Bins: each bin is a list of (x, y, z, w, h, d, orig_idx)
    bins = []

    def overlaps(x1, y1, z1, w1, h1, d1, x2, y2, z2, w2, h2, d2):
        return (x1 < x2 + w2 and x2 < x1 + w1 and
                y1 < y2 + h2 and y2 < y1 + h1 and
                z1 < z2 + d2 and z2 < z1 + d1)

    def find_best_position(bin_boxes, w, h, d):
        """Find the best (lowest z, then y, then x) valid position for a box in a bin."""
        if not bin_boxes:
            return (0, 0, 0)

        # Generate candidate positions from corners/edges of existing boxes
        candidates = set()
        candidates.add((0, 0, 0))

        # Collect all unique x, y, z coordinates from box edges
        xs = set([0])
        ys = set([0])
        zs = set([0])
        for (px, py, pz, pw, ph, pd, _) in bin_boxes:
            if px + pw <= W:
                xs.add(px + pw)
            if py + ph <= H:
                ys.add(py + ph)
            if pz + pd <= D:
                zs.add(pz + pd)
            xs.add(px)
            ys.add(py)
            zs.add(pz)

        # Limit candidate set size for performance
        # Use all combinations of edge coordinates
        for cx in xs:
            if cx + w > W:
                continue
            for cy in ys:
                if cy + h > H:
                    continue
                for cz in zs:
                    if cz + d > D:
                        continue
                    candidates.add((cx, cy, cz))

        # Also add simple corner-based candidates for robustness
        for (px, py, pz, pw, ph, pd, _) in bin_boxes:
            if px + pw <= W:
                candidates.add((px + pw, py, pz))
            if py + ph <= H:
                candidates.add((px, py + ph, pz))
            if pz + pd <= D:
                candidates.add((px, py, pz + pd))
            if px + pw <= W and py + ph <= H:
                candidates.add((px + pw, py + ph, pz))
            if px + pw <= W and pz + pd <= D:
                candidates.add((px + pw, py, pz + pd))
            if py + ph <= H and pz + pd <= D:
                candidates.add((px, py + ph, pz + pd))
            if px + pw <= W and py + ph <= H and pz + pd <= D:
                candidates.add((px + pw, py + ph, pz + pd))

        # Find best valid position (minimize z, then y, then x)
        best_pos = None
        best_key = None

        for (cx, cy, cz) in candidates:
            if cx + w > W or cy + h > H or cz + d > D:
                continue
            # Check for overlap
            valid = True
            for (px, py, pz, pw, ph, pd, _) in bin_boxes:
                if overlaps(cx, cy, cz, w, h, d, px, py, pz, pw, ph, pd):
                    valid = False
                    break
            if valid:
                key = (cz, cy, cx)
                if best_key is None or key < best_key:
                    best_key = key
                    best_pos = (cx, cy, cz)

        return best_pos

    # Place each box using FFD
    placement_map = [None] * n  # (bin_idx, x, y, z) for each original box

    for vol, orig_idx, w, h, d in indexed_boxes:
        if time.time() > deadline:
            # Emergency: place in new bin at origin
            bin_idx = len(bins)
            bins.append([(0, 0, 0, w, h, d, orig_idx)])
            placement_map[orig_idx] = (bin_idx, 0, 0, 0)
            continue

        placed = False
        # Try existing bins first
        for b_idx in range(len(bins)):
            if time.time() > deadline:
                break
            pos = find_best_position(bins[b_idx], w, h, d)
            if pos is not None:
                bins[b_idx].append((pos[0], pos[1], pos[2], w, h, d, orig_idx))
                placement_map[orig_idx] = (b_idx, pos[0], pos[1], pos[2])
                placed = True
                break

        if not placed:
            # Open a new bin
            bin_idx = len(bins)
            bins.append([(0, 0, 0, w, h, d, orig_idx)])
            placement_map[orig_idx] = (bin_idx, 0, 0, 0)

    # ---- Local Search Refinement ----
    # Goal: try to reduce bin count by moving boxes from the last bin into earlier bins
    # Or improve last-bin fill by rearranging

    def bin_volume(b):
        return sum(bw * bh * bd for (_, _, _, bw, bh, bd, _) in bins[b])

    def try_move_box(bin_from, box_idx_in_bin, bin_to):
        """Try to move a box from bin_from to bin_to. Returns True if successful."""
        if bin_from == bin_to:
            return False
        box = bins[bin_from][box_idx_in_bin]
        bx, by, bz, bw, bh, bd, bidx = box

        # Remove from source
        bins[bin_from].pop(box_idx_in_bin)

        # Try to place in target
        pos = find_best_position(bins[bin_to], bw, bh, bd)
        if pos is not None:
            bins[bin_to].append((pos[0], pos[1], pos[2], bw, bh, bd, bidx))
            placement_map[bidx] = (bin_to, pos[0], pos[1], pos[2])
            return True
        else:
            # Put back
            bins[bin_from].append(box)
            return False

    # Try to eliminate the last bin by moving its boxes to earlier bins
    if len(bins) > 1:
        last_bin_idx = len(bins) - 1
        # Iterate multiple times to allow cascading moves
        for iteration in range(10):
            if time.time() > deadline - 5:
                break
            if len(bins) <= 1:
                break
            last_bin_idx = len(bins) - 1
            moved_any = False
            # Try each box in the last bin (in reverse order to keep indices stable)
            for box_idx in range(len(bins[last_bin_idx]) - 1, -1, -1):
                if time.time() > deadline - 5:
                    break
                for target_bin in range(last_bin_idx):
                    if try_move_box(last_bin_idx, box_idx, target_bin):
                        moved_any = True
                        break
                if moved_any:
                    break
            if not moved_any:
                break

    # Try to improve last-bin fill: move boxes from earlier bins into the last bin
    # to increase its fill fraction (only if it doesn't increase B)
    if len(bins) > 1:
        last_bin_idx = len(bins) - 1
        # Try to fill the last bin more by moving smaller boxes into it
        for box_idx in range(len(bins[last_bin_idx]) - 1, -1, -1):
            if time.time() > deadline - 5:
                break
            # This is optional; skip for now to avoid complexity
            pass

    # Build result in original order
    placements = []
    for i in range(n):
        w, h, d = boxes[i]
        b_idx, x, y, z = placement_map[i]
        placements.append((b_idx, x, y, z, w, h, d))

    return placements
# EVOLVE-BLOCK-END
