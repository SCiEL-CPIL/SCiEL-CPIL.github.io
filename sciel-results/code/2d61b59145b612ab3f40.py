"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools

# EVOLVE-BLOCK-START
def load(boxes, container):
    W, H, D = container
    deadline = time.time() + 110.0  # safety margin before 120s kill

    # Precompute unique orientations for each box
    def get_orientations(w, h, d):
        opts = set()
        for perm in itertools.permutations((w, h, d)):
            opts.add(perm)
        return list(opts)

    # Try to place a box at position (x,y,z) with orientation (bw,bh,bd)
    # into a shelf structure. Returns (x,y,z,bw,bh,bd) or None.
    def try_place_shelf(bx, by, bz, bw, bh, bd, shelves, W, H, D):
        # Check if box fits within container
        if bx + bw > W or by + bh > H or bz + bd > D:
            return None
        # Check overlap with existing placed boxes
        for (ox, oy, oz, ow, oh, od) in shelves:
            if (bx < ox + ow and bx + bw > ox and
                by < oy + oh and by + bh > oy and
                bz < oz + od and bz + bd > oz):
                return None
        return (bx, by, bz, bw, bh, bd)

    # Shelf-based packing: boxes are placed in rows along X, stacked in Y within a row,
    # and rows stacked in Z. Each box can be rotated.
    def shelf_pack(box_list, W, H, D, deadline):
        """
        box_list: list of (original_index, w, h, d)
        Returns list of (bin, x, y, z, w, h, d) for all original boxes.
        """
        n = len(box_list)
        placements = [(-1, 0, 0, 0, 0, 0, 0)] * n

        # Shelf state:
        # current_x: x position for next box in current row
        # current_y: y offset for next box in current row (row stacking)
        # row_height: max height in current row
        # current_z: z offset for next row
        # layer_depth: max depth in current layer (z-slice)
        # Actually for 3D shelf: we fill a "layer" at fixed z, within which we have rows in x,
        # and within a row we stack in y.
        
        cur_x = 0
        cur_y = 0
        row_h = 0
        cur_z = 0
        layer_d = 0

        placed_boxes = []  # (x, y, z, w, h, d) for overlap check

        for idx, (orig_idx, w, h, d) in box_list:
            if time.time() > deadline:
                break
            orientations = get_orientations(w, h, d)
            best = None
            best_vol = -1
            for (bw, bh, bd) in orientations:
                # Try to place at (cur_x, cur_y, cur_z)
                if cur_x + bw > W:
                    # Move to next row (up in y)
                    new_y = cur_y + row_h
                    if new_y + bh > H:
                        # Move to next layer (z)
                        new_z = cur_z + layer_d
                        if new_z + bd > D:
                            continue
                        cur_z = new_z
                        cur_x = 0
                        cur_y = 0
                        row_h = 0
                        layer_d = 0
                        # Now try again at new position
                        if cur_x + bw > W or cur_y + bh > H or cur_z + bd > D:
                            continue
                    else:
                        cur_y = new_y
                        cur_x = 0
                        row_h = 0
                        if cur_x + bw > W:
                            continue
                # Check overlap with placed boxes
                pos = (cur_x, cur_y, cur_z)
                ok = True
                for (ox, oy, oz, ow, oh, od) in placed_boxes:
                    if (pos[0] < ox + ow and pos[0] + bw > ox and
                        pos[1] < oy + oh and pos[1] + bh > oy and
                        pos[2] < oz + od and pos[2] + bd > oz):
                        ok = False
                        break
                if ok:
                    vol = bw * bh * bd
                    if vol > best_vol:
                        best_vol = vol
                        best = (pos[0], pos[1], pos[2], bw, bh, bd)
            
            if best is not None:
                x, y, z, bw, bh, bd = best
                placements[orig_idx] = (0, x, y, z, bw, bh, bd)
                placed_boxes.append((x, y, z, bw, bh, bd))
                cur_x = x + bw
                row_h = max(row_h, bh)
                layer_d = max(layer_d, bd)
            # else: box left out
        
        return placements

    # Strategy 1: Sort by volume descending
    def strategy_volume_desc(boxes, W, H, D, deadline):
        indexed = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
        indexed.sort(key=lambda t: t[1]*t[2]*t[3], reverse=True)
        return shelf_pack(indexed, W, H, D, deadline)

    # Strategy 2: Sort by largest dimension descending
    def strategy_max_dim_desc(boxes, W, H, D, deadline):
        indexed = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
        indexed.sort(key=lambda t: max(t[1], t[2], t[3]), reverse=True)
        return shelf_pack(indexed, W, H, D, deadline)

    # Strategy 3: Sort by minimum dimension descending
    def strategy_min_dim_desc(boxes, W, H, D, deadline):
        indexed = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
        indexed.sort(key=lambda t: min(t[1], t[2], t[3]), reverse=True)
        return shelf_pack(indexed, W, H, D, deadline)

    # Strategy 4: Sort by surface area descending
    def strategy_surface_desc(boxes, W, H, D, deadline):
        indexed = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
        indexed.sort(key=lambda t: 2*(t[1]*t[2] + t[2]*t[3] + t[1]*t[3]), reverse=True)
        return shelf_pack(indexed, W, H, D, deadline)

    # Evaluate placements: compute filled volume
    def evaluate(placements, W, H, D):
        total_vol = 0
        for (bin, x, y, z, w, h, d) in placements:
            if bin == 0:
                # Verify within bounds and no overlap
                if x + w > W or y + h > H or z + d > D:
                    return 0  # invalid
                total_vol += w * h * d
        # Check overlaps
        loaded = [(x, y, z, w, h, d) for (bin, x, y, z, w, h, d) in placements if bin == 0]
        for i in range(len(loaded)):
            for j in range(i+1, len(loaded)):
                xi, yi, zi, wi, hi, di = loaded[i]
                xj, yj, zj, wj, hj, dj = loaded[j]
                if (xi < xj + wj and xi + wi > xj and
                    yi < yj + hj and yi + hi > yj and
                    zi < zj + dj and zi + di > zj):
                    return 0  # overlap
        container_vol = W * H * D
        if container_vol == 0:
            return 0
        return total_vol / container_vol

    # Try multiple strategies, keep best
    strategies = [
        strategy_volume_desc,
        strategy_max_dim_desc,
        strategy_min_dim_desc,
        strategy_surface_desc,
    ]

    best_placements = None
    best_score = 0

    for strat in strategies:
        if time.time() > deadline:
            break
        try:
            placements = strat(boxes, W, H, D, deadline)
            score = evaluate(placements, W, H, D)
            if score > best_score:
                best_score = score
                best_placements = placements
        except Exception:
            continue

    if best_placements is None:
        # Fallback: all left out
        best_placements = [(-1, 0, 0, 0, w, h, d) for (w, h, d) in boxes]

    return best_placements
# EVOLVE-BLOCK-END
