import time
import math

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []

    # Precompute box dimensions as tuples
    box_dims = [(int(w), int(h), int(d)) for w, h, d in boxes]

    # Sort indices by volume descending (largest first)
    vol_sorted = sorted(range(n), key=lambda i: box_dims[i][0]*box_dims[i][1]*box_dims[i][2], reverse=True)

    def create_full_space():
        return [(0, 0, 0, W, H, D)]

    def fits_in_space(box, space):
        """Check if box fits in the given space (x,y,z,x2,y2,z2)."""
        w, h, d = box
        sx, sy, sz, sx2, sy2, sz2 = space
        return (w <= sx2 - sx and h <= sy2 - sy and d <= sz2 - sz)

    def split_space(space, box_pos, box):
        """
        Split a space by placing a box at box_pos inside it.
        Returns list of new non-overlapping sub-spaces.
        space = (sx, sy, sz, sx2, sy2, sz2)
        box_pos = (px, py, pz)
        box = (w, h, d)
        """
        sx, sy, sz, sx2, sy2, sz2 = space
        px, py, pz = box_pos
        w, h, d = box

        new_spaces = []

        # Split along x-axis: left and right
        if px > sx:
            new_spaces.append((sx, sy, sz, px, sy2, sz2))
        if px + w < sx2:
            new_spaces.append((px + w, sy, sz, sx2, sy2, sz2))

        # Split along y-axis: bottom and top (within the x-range of the box)
        if py > sy:
            new_spaces.append((px, sy, sz, px + w, py, sz2))
        if py + h < sy2:
            new_spaces.append((px, py + h, sz, px + w, sy2, sz2))

        # Split along z-axis: back and front (within the x,y range of the box)
        if pz > sz:
            new_spaces.append((px, py, sz, px + w, py + h, pz))
        if pz + d < sz2:
            new_spaces.append((px, py, pz + d, px + w, py + h, sz2))

        return new_spaces

    def find_position(box, spaces):
        """
        Find a position for box in the list of spaces.
        Returns (position, new_spaces) or (None, spaces).
        Uses bottom-left-farthest heuristic: minimize (x, y, z) lexicographically.
        """
        w, h, d = box
        best_pos = None
        best_key = None

        for space in spaces:
            sx, sy, sz, sx2, sy2, sz2 = space
            if not fits_in_space(box, space):
                continue
            # Try placing at the "bottom-left-farthest" corner of this space
            px, py, pz = sx, sy, sz
            if w <= sx2 - sx and h <= sy2 - sy and d <= sz2 - sz:
                key = (px, py, pz)
                if best_key is None or key < best_key:
                    best_key = key
                    best_pos = (px, py, pz)

        if best_pos is None:
            return None, spaces

        # Now split all spaces that contain best_pos
        new_spaces = []
        for space in spaces:
            sx, sy, sz, sx2, sy2, sz2 = space
            # Check if best_pos is within this space
            if (best_pos[0] >= sx and best_pos[0] + w <= sx2 and
                best_pos[1] >= sy and best_pos[1] + h <= sy2 and
                best_pos[2] >= sz and best_pos[2] + d <= sz2):
                # This space contains the box, so split it
                new_spaces.extend(split_space(space, best_pos, box))
            else:
                # Check if this space overlaps with the placed box
                # If it does, we need to handle it, but in our approach
                # we only split the space that fully contains the box position
                new_spaces.append(space)

        return best_pos, new_spaces

    def try_pack_order(order):
        """Try to pack boxes in the given order. Returns placements or None if failed."""
        bins_spaces = [create_full_space()]
        placements = {}  # index -> (bin, x, y, z)

        for idx in order:
            box = box_dims[idx]
            placed = False

            # Try existing bins first (prefer bins with more filled space / lower index)
            for b in range(len(bins_spaces)):
                if time.time() > deadline:
                    return None
                pos, new_spaces = find_position(box, bins_spaces[b])
                if pos is not None:
                    placements[idx] = (b, pos[0], pos[1], pos[2])
                    bins_spaces[b] = new_spaces
                    placed = True
                    break

            if not placed:
                # Try a new bin
                if time.time() > deadline:
                    return None
                b = len(bins_spaces)
                pos = (0, 0, 0)
                box = box_dims[idx]
                if box[0] <= W and box[1] <= H and box[2] <= D:
                    new_spaces = split_space(create_full_space(), pos, box)
                    bins_spaces.append(new_spaces)
                    placements[idx] = (b, pos[0], pos[1], pos[2])
                else:
                    return None

        # Build result in original order
        result = []
        for i in range(n):
            b, x, y, z = placements[i]
            w, h, d = box_dims[i]
            result.append((b, x, y, z, w, h, d))
        return result

    # Try different packing orders
    best_result = None
    best_score = float('inf')

    # Order 1: Volume descending
    result = try_pack_order(vol_sorted)
    if result is not None:
        best_result = result
        # Compute score
        max_bin = max(p[0] for p in result) if result else 0
        B = max_bin + 1
        last_bin_vol = sum(p[4]*p[5]*p[6] for p in result if p[0] == max_bin)
        container_vol = W * H * D
        score = B + 0.1 * (1 - last_bin_vol / container_vol)
        best_score = score

    # Order 2: Volume ascending
    vol_asc = sorted(range(n), key=lambda i: box_dims[i][0]*box_dims[i][1]*box_dims[i][2])
    if time.time() < deadline:
        result = try_pack_order(vol_asc)
        if result is not None:
            max_bin = max(p[0] for p in result) if result else 0
            B = max_bin + 1
            last_bin_vol = sum(p[4]*p[5]*p[6] for p in result if p[0] == max_bin)
            container_vol = W * H * D
            score = B + 0.1 * (1 - last_bin_vol / container_vol)
            if score < best_score:
                best_result = result
                best_score = score

    # Order 3: Original order
    if time.time() < deadline:
        result = try_pack_order(list(range(n)))
        if result is not None:
            max_bin = max(p[0] for p in result) if result else 0
            B = max_bin + 1
            last_bin_vol = sum(p[4]*p[5]*p[6] for p in result if p[0] == max_bin)
            container_vol = W * H * D
            score = B + 0.1 * (1 - last_bin_vol / container_vol)
            if score < best_score:
                best_result = result
                best_score = score

    # Order 4: Sort by max dimension descending
    max_dim_sorted = sorted(range(n), key=lambda i: max(box_dims[i]), reverse=True)
    if time.time() < deadline:
        result = try_pack_order(max_dim_sorted)
        if result is not None:
            max_bin = max(p[0] for p in result) if result else 0
            B = max_bin + 1
            last_bin_vol = sum(p[4]*p[5]*p[6] for p in result if p[0] == max_bin)
            container_vol = W * H * D
            score = B + 0.1 * (1 - last_bin_vol / container_vol)
            if score < best_score:
                best_result = result
                best_score = score

    # Order 5: Sort by sum of dimensions descending
    sum_dim_sorted = sorted(range(n), key=lambda i: sum(box_dims[i]), reverse=True)
    if time.time() < deadline:
        result = try_pack_order(sum_dim_sorted)
        if result is not None:
            max_bin = max(p[0] for p in result) if result else 0
            B = max_bin + 1
            last_bin_vol = sum(p[4]*p[5]*p[6] for p in result if p[0] == max_bin)
            container_vol = W * H * D
            score = B + 0.1 * (1 - last_bin_vol / container_vol)
            if score < best_score:
                best_result = result
                best_score = score

    # Fallback: use the simple shelf heuristic if MSP fails or deadline exceeded
    if best_result is None or time.time() > deadline - 5:
        W, H, D = container
        placements, b, x, y, z, row_h, layer_d = [], 0, 0, 0, 0, 0, 0
        for i in range(n):
            w, h, d = box_dims[i]
            if x + w > W:
                x, y, row_h = 0, y + row_h, 0
            if y + h > H:
                y, z, layer_d = 0, z + layer_d, 0
                x, row_h = 0, 0
            if z + d > D:
                b, x, y, z, row_h, layer_d = b + 1, 0, 0, 0, 0, 0
            placements.append((b, x, y, z, w, h, d))
            x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)
        best_result = placements

    return best_result
