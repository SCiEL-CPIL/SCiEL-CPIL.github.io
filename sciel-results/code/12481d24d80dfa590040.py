"""Improved 3D Bin Packing: fast best-fit with corner candidates, relocation, and time guard."""
import time
import random

# EVOLVE-BLOCK-START
def pack(boxes, container):
    """
    Place boxes into bins using a fast best-fit decreasing approach with
    a light time-bounded relocation pass.
    Returns list of (bin, x, y, z, w, h, d) tuples.
    """
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill

    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []

    # Indexed boxes: (orig_idx, w, h, d, vol)
    indexed_boxes = [(i, w, h, d, w * h * d) for i, (w, h, d) in enumerate(boxes)]

    def compute_score(placements):
        """B + 0.1*(1 - last-bin occupied volume / container volume)"""
        if not placements:
            return 0.0
        bins_used = set()
        last_bin = -1
        for (b, x, y, z, w, h, d) in placements:
            bins_used.add(b)
            if b > last_bin:
                last_bin = b
        last_bin_vol = 0
        for (b, x, y, z, w, h, d) in placements:
            if b == last_bin:
                last_bin_vol += w * h * d
        container_vol = W * H * D
        B = len(bins_used)
        occ_frac = last_bin_vol / container_vol if container_vol > 0 else 1.0
        return B + 0.1 * (1.0 - occ_frac)

    def find_placement(placed, w, h, d):
        """
        Find best (x, y, z) for box (w,h,d) in a bin with given placed boxes.
        placed: list of (x, y, z, w, h, d).
        Uses corner candidates from placed boxes plus origin.
        Prefers lowest z, then lowest y, then lowest x (gravity + compactness).
        Returns (x, y, z) or None.
        """
        # Quick fit check
        if w > W or h > H or d > D:
            return None

        # Build candidate x, y, z values from box edges
        xs = {0}
        ys = {0}
        zs = {0}
        for (px, py, pz, pw, ph, pd) in placed:
            xs.add(px)
            xs.add(px + pw)
            ys.add(py)
            ys.add(py + ph)
            zs.add(pz)
            zs.add(pz + pd)

        # Filter to valid positions
        xs = [x for x in xs if x + w <= W]
        ys = [y for y in ys if y + h <= H]
        zs = [z for z in zs if z + d <= D]

        if not xs or not ys or not zs:
            return None

        best_pos = None
        best_metric = None

        # Sort by z (gravity), then y, then x for compactness
        for cz in sorted(zs):
            for cy in sorted(ys):
                for cx in sorted(xs):
                    # Check overlap
                    overlap = False
                    for (px, py, pz, pw, ph, pd) in placed:
                        if not (cx + w <= px or px + pw <= cx or
                                cy + h <= py or py + ph <= cy or
                                cz + d <= pz or pz + pd <= cz):
                            overlap = True
                            break
                    if overlap:
                        continue
                    # Found a valid position
                    metric = (cz, cy, cx)
                    if best_metric is None or metric < best_metric:
                        best_metric = metric
                        best_pos = (cx, cy, cz)
                        if cx == 0 and cy == 0 and cz == 0:
                            return best_pos

        return best_pos

    def solve(sorted_boxes):
        """Best-fit decreasing: try most-filled bins first."""
        bins = []  # each bin: {'placed': [...], 'vol': int}
        placements = []

        for (orig_idx, w, h, d, vol) in sorted_boxes:
            if time.time() > deadline:
                return None

            # Try existing bins, most-filled first (best-fit)
            if bins:
                bin_order = sorted(range(len(bins)), key=lambda bi: -bins[bi]['vol'])
            else:
                bin_order = []

            placed_in_bin = False
            for bi in bin_order:
                pos = find_placement(bins[bi]['placed'], w, h, d)
                if pos is not None:
                    x, y, z = pos
                    bins[bi]['placed'].append((x, y, z, w, h, d))
                    bins[bi]['vol'] += vol
                    placements.append((bi, x, y, z, w, h, d))
                    placed_in_bin = True
                    break

            if not placed_in_bin:
                # Open new bin
                new_bin = {'placed': [(0, 0, 0, w, h, d)], 'vol': vol}
                bins.append(new_bin)
                placements.append((len(bins) - 1, 0, 0, 0, w, h, d))

        return placements

    # Multiple sorting strategies
    strategies = [
        lambda bl: sorted(bl, key=lambda x: -x[4]),
        lambda bl: sorted(bl, key=lambda x: (-(x[4]), -(x[1]+x[2]+x[3]))),
        lambda bl: sorted(bl, key=lambda x: (-(x[1]+x[2]+x[3]), -x[4])),
        lambda bl: sorted(bl, key=lambda x: (-(x[1]*x[2]), -x[4])),
        lambda bl: sorted(bl, key=lambda x: (-(x[2]*x[3]), -x[4])),
        lambda bl: sorted(bl, key=lambda x: (-(x[1]*x[3]), -x[4])),
        lambda bl: sorted(bl, key=lambda x: (-max(x[1],x[2],x[3]), -x[4])),
    ]

    best_placements = None
    best_score = float('inf')

    for strat in strategies:
        if time.time() > deadline - 5.0:
            break
        sorted_boxes = strat(indexed_boxes)
        placements = solve(sorted_boxes)
        if placements is None:
            continue
        score = compute_score(placements)
        if score < best_score:
            best_score = score
            best_placements = placements

    if best_placements is None:
        # Fallback: simple sequential packing
        placements = []
        b = 0
        x = y = z = 0
        row_h = layer_d = 0
        for (orig_idx, w, h, d, vol) in indexed_boxes:
            if x + w > W:
                x = 0
                y += row_h
                row_h = 0
            if y + h > H:
                y = 0
                z += layer_d
                layer_d = 0
                row_h = 0
            if z + d > D:
                b += 1
                x = y = z = 0
                row_h = layer_d = 0
            placements.append((b, x, y, z, w, h, d))
            x += w
            row_h = max(row_h, h)
            layer_d = max(layer_d, d)
        best_placements = placements
        best_score = compute_score(best_placements)

    # Relocation pass: try to move boxes from the last bin into earlier bins
    # to reduce B or improve last-bin fill
    def _overlap(a, b):
        ax, ay, az, aw, ah, ad = a
        bx, by, bz, bw, bh, bd = b
        if ax + aw <= bx or bx + bw <= ax:
            return False
        if ay + ah <= by or by + bh <= ay:
            return False
        if az + ad <= bz or bz + bd <= az:
            return False
        return True

    def _find_corner(box, items, W, H, D):
        """Find lowest (z, then y, then x) corner where box fits without overlap."""
        w, h, d = box
        xs = sorted(set([0] + [it[0] + it[3] for it in items if it[0] + it[3] <= W]))
        ys = sorted(set([0] + [it[1] + it[4] for it in items if it[1] + it[4] <= H]))
        zs = sorted(set([0] + [it[2] + it[5] for it in items if it[2] + it[5] <= D]))
        for z in zs:
            if z + d > D:
                break
            for y in ys:
                if y + h > H:
                    break
                for x in xs:
                    if x + w > W:
                        break
                    ok = True
                    for it in items:
                        if _overlap((x, y, z, w, h, d), it):
                            ok = False
                            break
                    if ok:
                        return (x, y, z)
        return None

    def _bin_volume(bin_items):
        return sum(it[3] * it[4] * it[5] for it in bin_items)

    # Build bins from best_placements for relocation
    bins = {}
    for (b, x, y, z, w, h, d) in best_placements:
        if b not in bins:
            bins[b] = []
        bins[b].append((x, y, z, w, h, d))

    def _rebuild_placements(bins_dict):
        pl = []
        for b in sorted(bins_dict.keys()):
            for (px, py, pz, pw, ph, pd) in bins_dict[b]:
                pl.append((b, px, py, pz, pw, ph, pd))
        return pl

    # Aggressive relocation: repeatedly try to move boxes from highest bin to earlier bins
    # Also try to swap pairs of boxes between bins to improve packing
    attempts = 0
    max_attempts = 5000
    while time.time() < deadline - 1.0 and attempts < max_attempts:
        attempts += 1
        used_bins = sorted(bins.keys())
        if len(used_bins) <= 1:
            break
        last_bin = used_bins[-1]
        if last_bin not in bins or len(bins[last_bin]) == 0:
            break

        # Get boxes in last bin, try each one
        cand_indices = list(range(len(bins[last_bin])))
        random.shuffle(cand_indices)

        moved = False
        for ci in cand_indices:
            item = bins[last_bin][ci]
            x, y, z, w, h, d = item
            box = (w, h, d)

            # Try moving to earlier bins (most filled first for best-fit)
            for tb in sorted(used_bins[:-1], key=lambda b: -_bin_volume(bins.get(b, []))):
                if tb not in bins:
                    continue
                placed = _find_corner(box, bins[tb], W, H, D)
                if placed is None:
                    continue
                # Try the move
                nx, ny, nz = placed
                bins[last_bin].remove(item)
                bins[tb].append((nx, ny, nz, w, h, d))

                if len(bins[last_bin]) == 0:
                    del bins[last_bin]

                new_placements = _rebuild_placements(bins)
                new_score = compute_score(new_placements)
                if new_score < best_score:
                    best_score = new_score
                    best_placements = new_placements
                else:
                    # Revert
                    bins[tb].remove((nx, ny, nz, w, h, d))
                    bins[last_bin].append(item)
                moved = True
                break

            if moved:
                break

        if not moved:
            # Try swap-based local search: swap a box from last bin with a box from an earlier bin
            swapped = False
            for ci in cand_indices:
                if swapped:
                    break
                item_a = bins[last_bin][ci]
                xa, ya, za, wa, ha, da = item_a
                box_a = (wa, ha, da)

                for tb in used_bins[:-1]:
                    if swapped:
                        break
                    if tb not in bins:
                        continue
                    for tj in range(len(bins[tb])):
                        item_b = bins[tb][tj]
                        xb, yb, zb, wb, hb, db = item_b
                        box_b = (wb, hb, db)

                        # Try swapping: put box_a in bin tb (replacing item_b's spot),
                        # and put box_b in last bin (replacing item_a's spot)
                        # First remove both
                        bins[last_bin].remove(item_a)
                        bins[tb].remove(item_b)

                        # Try placing box_a in bin tb
                        pos_a = _find_corner(box_a, bins[tb], W, H, D)
                        if pos_a is not None:
                            # Try placing box_b in last bin
                            pos_b = _find_corner(box_b, bins[last_bin], W, H, D)
                            if pos_b is not None:
                                pa_x, pa_y, pa_z = pos_a
                                pb_x, pb_y, pb_z = pos_b
                                bins[tb].append((pa_x, pa_y, pa_z, wa, ha, da))
                                bins[last_bin].append((pb_x, pb_y, pb_z, wb, hb, db))

                                new_placements = _rebuild_placements(bins)
                                new_score = compute_score(new_placements)
                                if new_score < best_score:
                                    best_score = new_score
                                    best_placements = new_placements
                                    swapped = True
                                else:
                                    # Revert
                                    bins[tb].remove((pa_x, pa_y, pa_z, wa, ha, da))
                                    bins[last_bin].remove((pb_x, pb_y, pb_z, wb, hb, db))
                                    bins[last_bin].append(item_a)
                                    bins[tb].append(item_b)
                            else:
                                # Revert
                                bins[last_bin].append(item_a)
                                bins[tb].append(item_b)
                        else:
                            # Revert
                            bins[last_bin].append(item_a)
                            bins[tb].append(item_b)

            if not swapped:
                break

    # Rebuild final placements from bins if we did relocations
    final_placements = _rebuild_placements(bins)
    final_score = compute_score(final_placements)
    if final_score < best_score:
        best_placements = final_placements

    return best_placements
# EVOLVE-BLOCK-END
