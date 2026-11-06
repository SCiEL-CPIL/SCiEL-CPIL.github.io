"""3D Bin Packing with multi-strategy FFD and local search."""
import time
import random
import numpy as np

def pack(boxes, container):
    """
    Place boxes into bins. Returns list of (bin_id, x, y, z, w, h, d) for each box.
    """
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []

    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill
    container_volume = W * H * D

    # Box dimensions
    dims = [(w, h, d) for (w, h, d) in boxes]
    vols = [w * h * d for (w, h, d) in boxes]

    # Generate candidate orderings: volume desc, surface area desc, max dim desc, etc.
    orderings = []
    # Primary: volume descending, then longest side descending
    orderings.append(sorted(range(n), key=lambda i: (-vols[i], -max(dims[i]))))
    # Surface area
    orderings.append(sorted(range(n), key=lambda i: (-(2*(dims[i][0]*dims[i][1] + dims[i][1]*dims[i][2] + dims[i][0]*dims[i][2])), -vols[i])))
    # Max dimension
    orderings.append(sorted(range(n), key=lambda i: (-max(dims[i]), -vols[i])))
    # Min dimension
    orderings.append(sorted(range(n), key=lambda i: (-min(dims[i]), -vols[i])))

    def make_grid():
        return np.zeros((H, W, D), dtype=np.uint8)

    def can_place(grid, x, y, z, w, h, d):
        if x + w > W or y + h > H or z + d > D:
            return False
        if grid[y:y+h, x:x+w, z:z+d].any():
            return False
        return True

    def place(grid, x, y, z, w, h, d):
        grid[y:y+h, x:x+w, z:z+d] = 1

    def unplace(grid, x, y, z, w, h, d):
        grid[y:y+h, x:x+w, z:z+d] = 0

    def find_position_corner(bin_placements, grid, w, h, d):
        """Find best position using corner-based candidates."""
        candidates = set()
        candidates.add((0, 0, 0))
        for (px, py, pz, pw, ph, pd) in bin_placements:
            candidates.add((px + pw, py, pz))
            candidates.add((px, py + ph, pz))
            candidates.add((px, py, pz + pd))
            candidates.add((px + pw, py + ph, pz))
            candidates.add((px + pw, py, pz + pd))
            candidates.add((px, py + ph, pz + pd))
            candidates.add((px + pw, py + ph, pz + pd))

        best_pos = None
        best_score = float('inf')

        for (xc, yc, zc) in candidates:
            if xc + w > W or yc + h > H or zc + d > D:
                continue
            if can_place(grid, xc, yc, zc, w, h, d):
                # Score: prefer compact placement (minimize max extents)
                max_x = max(xc + w, *(px + pw for (px, py, pz, pw, ph, pd) in bin_placements)) if bin_placements else xc + w
                max_y = max(yc + h, *(py + ph for (px, py, pz, pw, ph, pd) in bin_placements)) if bin_placements else yc + h
                max_z = max(zc + d, *(pz + pd for (px, py, pz, pw, ph, pd) in bin_placements)) if bin_placements else zc + d
                score = max_x * max_y * max_z + xc + yc + zc
                if score < best_score:
                    best_score = score
                    best_pos = (xc, yc, zc)

        return best_pos

    def find_position_shelf(bin_placements, grid, w, h, d):
        """Find best position using shelf-based placement."""
        shelf_levels = set([0])
        for (px, py, pz, pw, ph, pd) in bin_placements:
            shelf_levels.add(py + ph)

        best_pos = None
        best_score = float('inf')

        for y_level in sorted(shelf_levels):
            if y_level + h > H:
                continue
            x_candidates = set([0])
            for (px, py, pz, pw, ph, pd) in bin_placements:
                if py == y_level:
                    x_candidates.add(px + pw)

            for x in sorted(x_candidates):
                if x + w > W:
                    continue
                z_candidates = set([0])
                for (px, py, pz, pw, ph, pd) in bin_placements:
                    if py == y_level and px <= x and px + pw >= x + w:
                        z_candidates.add(pz + pd)

                for z in sorted(z_candidates):
                    if z + d > D:
                        continue
                    if can_place(grid, x, y_level, z, w, h, d):
                        score = x + y_level + z
                        if score < best_score:
                            best_score = score
                            best_pos = (x, y_level, z)
                        break

        return best_pos

    def find_position(bin_placements, grid, w, h, d):
        """Try multiple placement strategies."""
        pos = find_position_corner(bin_placements, grid, w, h, d)
        if pos is not None:
            return pos
        pos = find_position_shelf(bin_placements, grid, w, h, d)
        if pos is not None:
            return pos
        return None

    def do_ffd(ordering):
        """Run FFD with given ordering. Returns (result, bins, grids)."""
        result = [None] * n
        bins = []
        grids = []

        for idx in ordering:
            if time.time() > deadline:
                break
            w, h, d = dims[idx]

            best_bin = -1
            best_pos = None
            best_fit = None

            for b_idx, bin_placements in enumerate(bins):
                grid = grids[b_idx]
                pos = find_position(bin_placements, grid, w, h, d)
                if pos is not None:
                    x, y, z = pos
                    new_max_x = max(x + w, *(px + pw for (px, py, pz, pw, ph, pd) in bin_placements)) if bin_placements else x + w
                    new_max_y = max(y + h, *(py + ph for (px, py, pz, pw, ph, pd) in bin_placements)) if bin_placements else y + h
                    new_max_z = max(z + d, *(pz + pd for (px, py, pz, pw, ph, pd) in bin_placements)) if bin_placements else z + d
                    fit = new_max_x + new_max_y + new_max_z

                    if best_fit is None or fit < best_fit:
                        best_fit = fit
                        best_bin = b_idx
                        best_pos = pos

            if best_bin >= 0:
                x, y, z = best_pos
                bins[best_bin].append((x, y, z, w, h, d))
                place(grids[best_bin], x, y, z, w, h, d)
                result[idx] = (best_bin, x, y, z, w, h, d)
            else:
                new_bin = [(0, 0, 0, w, h, d)]
                new_grid = make_grid()
                place(new_grid, 0, 0, 0, w, h, d)
                bins.append(new_bin)
                grids.append(new_grid)
                result[idx] = (len(bins) - 1, 0, 0, 0, w, h, d)

        # Fallback for any unplaced
        for idx in ordering:
            if result[idx] is not None:
                continue
            w, h, d = dims[idx]
            placed = False
            for b_idx in range(len(bins)):
                pos = find_position(bins[b_idx], grids[b_idx], w, h, d)
                if pos is not None:
                    x, y, z = pos
                    bins[b_idx].append((x, y, z, w, h, d))
                    place(grids[b_idx], x, y, z, w, h, d)
                    result[idx] = (b_idx, x, y, z, w, h, d)
                    placed = True
                    break
            if not placed:
                new_bin = [(0, 0, 0, w, h, d)]
                new_grid = make_grid()
                place(new_grid, 0, 0, 0, w, h, d)
                bins.append(new_bin)
                grids.append(new_grid)
                result[idx] = (len(bins) - 1, 0, 0, 0, w, h, d)

        return result, bins, grids

    def compute_score(result, bins):
        """Compute B + 0.1*(1 - last_bin_occupied/container_volume)."""
        if not bins:
            return float('inf')
        B = len(bins)
        # Find last bin (max label)
        last_bin_idx = max(r[0] for r in result) if result else 0
        last_vol = sum(w * h * d for (x, y, z, w, h, d) in bins[last_bin_idx])
        frac = last_vol / container_volume if container_volume > 0 else 1.0
        return B + 0.1 * (1 - frac)

    def local_search(result, bins, grids):
        """Local search to reduce bins and improve last-bin occupancy."""
        if time.time() > deadline - 5:
            return

        # Phase 1: Try to eliminate bins by moving boxes to earlier bins
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            for b_idx in range(len(bins) - 1, -1, -1):
                if time.time() > deadline - 5:
                    break
                if not bins[b_idx]:
                    continue

                boxes_to_try = list(bins[b_idx])
                for box in boxes_to_try:
                    x, y, z, w, h, d = box
                    moved = False
                    for target_bin in range(b_idx):
                        if time.time() > deadline - 5:
                            break
                        pos = find_position(bins[target_bin], grids[target_bin], w, h, d)
                        if pos is not None:
                            # Move box
                            unplace(grids[b_idx], x, y, z, w, h, d)
                            bins[b_idx].remove(box)
                            nx, ny, nz = pos
                            bins[target_bin].append((nx, ny, nz, w, h, d))
                            place(grids[target_bin], nx, ny, nz, w, h, d)
                            # Update result
                            for i in range(len(result)):
                                r = result[i]
                                if r is not None and r[0] == b_idx and r[1:] == (x, y, z, w, h, d):
                                    result[i] = (target_bin, nx, ny, nz, w, h, d)
                                    break
                            moved = True
                            changed = True
                            break
                    if not bins[b_idx]:
                        # Remove empty bin and renumber
                        bins.pop(b_idx)
                        grids.pop(b_idx)
                        for i in range(len(result)):
                            if result[i] is not None:
                                bid = result[i][0]
                                if bid > b_idx:
                                    result[i] = (bid - 1,) + result[i][1:]
                                elif bid == b_idx:
                                    result[i] = (bid - 1,) + result[i][1:]
                        break

        # Phase 2: Try to increase last-bin occupancy by moving boxes in
        if time.time() < deadline - 3 and len(bins) > 1:
            last_bin_idx = len(bins) - 1
            for b_idx in range(last_bin_idx):
                if time.time() > deadline - 3:
                    break
                boxes_to_try = list(bins[b_idx])
                for box in boxes_to_try:
                    x, y, z, w, h, d = box
                    pos = find_position(bins[last_bin_idx], grids[last_bin_idx], w, h, d)
                    if pos is not None:
                        unplace(grids[b_idx], x, y, z, w, h, d)
                        bins[b_idx].remove(box)
                        nx, ny, nz = pos
                        bins[last_bin_idx].append((nx, ny, nz, w, h, d))
                        place(grids[last_bin_idx], nx, ny, nz, w, h, d)
                        for i in range(len(result)):
                            r = result[i]
                            if r is not None and r[0] == b_idx and r[1:] == (x, y, z, w, h, d):
                                result[i] = (last_bin_idx, nx, ny, nz, w, h, d)
                                break

    # Try multiple orderings and keep the best
    best_result = None
    best_score = float('inf')

    for ordering in orderings:
        if time.time() > deadline - 20:
            break
        result, bins, grids = do_ffd(ordering)
        local_search(result, bins, grids)
        score = compute_score(result, bins)
        if score < best_score:
            best_score = score
            best_result = result

    # If we still have time, try randomized orderings
    if time.time() < deadline - 15:
        for restart in range(5):
            if time.time() > deadline - 10:
                break
            # Perturb the volume ordering
            base = orderings[0][:]
            for _ in range(min(5, n // 4)):
                i = random.randint(0, n - 2)
                j = random.randint(i + 1, n - 1)
                base[i], base[j] = base[j], base[i]
            result, bins, grids = do_ffd(base)
            local_search(result, bins, grids)
            score = compute_score(result, bins)
            if score < best_score:
                best_score = score
                best_result = result

    if best_result is None:
        # Shouldn't happen, but fallback
        best_result, _, _ = do_ffd(orderings[0])

    return best_result
