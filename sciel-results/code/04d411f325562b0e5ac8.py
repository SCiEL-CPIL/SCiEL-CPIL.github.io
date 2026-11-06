"""3D Container Loading: hierarchical free-space (L-shaped decomposition) packer."""
import time
import itertools

# EVOLVE-BLOCK-START
def load(boxes, container):
    start = time.time()
    deadline = start + 108.0  # safety margin vs 150s kill
    W, H, D = int(container[0]), int(container[1]), int(container[2])
    n = len(boxes)

    # Fallback: simple row algorithm (always valid, fast)
    def fallback():
        placements, x, y, z, row_h, layer_d = [], 0, 0, 0, 0, 0
        for w, h, d in boxes:
            if x + w > W:
                x, y, row_h = 0, y + row_h, 0
            if y + h > H:
                y, z, layer_d, x, row_h = 0, z + layer_d, 0, 0, 0
            if z + d > D:
                placements.append((-1, 0, 0, 0, w, h, d))
                continue
            placements.append((0, x, y, z, w, h, d))
            x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)
        return placements

    if W * H * D > 12_000_000:
        return fallback()

    # Free-space list: each entry is (x, y, z, w, h, d) of an empty rectangular prism.
    # Placing a box in one free-space recursively splits it into up to 3 sub-prisms
    # (L-shaped decomposition), which inherently guarantees non-overlap.
    free_spaces = [(0, 0, 0, W, H, D)]

    def split_free_space(fs, bx, by, bz, bw, bh, bd):
        """Given free-space fs and a box placed at (bx,by,bz) with size (bw,bh,bd),
        return the list of remaining free sub-spaces (up to 3)."""
        fx, fy, fz, fw, fh, fd = fs
        result = []
        # Left part: [fx, bx) x [fy, fy+fh) x [fz, fz+fd)
        lw = bx - fx
        if lw > 0:
            result.append((fx, fy, fz, lw, fh, fd))
        # Right part: [bx+bw, fx+fw) x [fy, fy+fh) x [fz, fz+fd)
        rw = fx + fw - (bx + bw)
        if rw > 0:
            result.append((bx + bw, fy, fz, rw, fh, fd))
        # Bottom part: [bx, bx+bw) x [fy, by) x [fz, fz+fd)
        bh_ = by - fy
        if bh_ > 0:
            result.append((bx, fy, fz, bw, bh_, fd))
        # Top part: [bx, bx+bw) x [by+bh, fy+fh) x [fz, fz+fd)
        th = fy + fh - (by + bh)
        if th > 0:
            result.append((bx, by + bh, fz, bw, th, fd))
        # Front part: [bx, bx+bw) x [by, by+bh) x [fz, bz)
        fd_ = bz - fz
        if fd_ > 0:
            result.append((bx, by, fz, bw, bh, fd_))
        # Back part: [bx, bx+bw) x [by, by+bh) x [bz+bd, fz+fd)
        bd_ = fz + fd - (bz + bd)
        if bd_ > 0:
            result.append((bx, by, bz + bd, bw, bh, bd_))
        return result

    # Orientation generator: all 6 permutations (deduplicated)
    def orientations(w, h, d):
        seen = set()
        for perm in itertools.permutations((w, h, d)):
            if perm not in seen:
                seen.add(perm)
                yield perm

    # Score a placement: prefer low z, then low y, then low x, then smaller footprint
    def placement_score(px, py, pz, pw, ph, pd):
        # Primary: minimize z (fill from bottom), then y, then x
        # Secondary: prefer tighter fit (smaller max dimension)
        return (pz, py, px, max(pw, ph, pd), pw * ph * pd)

    placements = [None] * n
    placed_count = 0
    unplaced = []

    # Order boxes: largest volume first
    order = sorted(range(n), key=lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2], reverse=True)

    # First pass: place boxes in volume order using free-space search
    for bi in order:
        if time.time() > deadline:
            unplaced.append(bi)
            continue
        w, h, d = boxes[bi]
        best = None  # (score_tuple, fs_index, ox, oy, oz, ow, oh, od)

        # Search each free-space for a valid placement
        # For each free-space, the box can be placed at the origin of that space
        # (bottom-left-front) since we want compact packing
        for fi, fs in enumerate(list(free_spaces)):
            if time.time() > deadline:
                unplaced.append(bi)
                break
            fx, fy, fz, fw, fh, fd = fs
            for (ow, oh, od) in orientations(w, h, d):
                if ow > fw or oh > fh or od > fd:
                    continue
                # Place at the bottom-left-front of this free-space
                px, py, pz = fx, fy, fz
                sc = placement_score(px, py, pz, ow, oh, od)
                if best is None or sc < best[0]:
                    best = (sc, fi, px, py, pz, ow, oh, od)
            else:
                continue
            # Found a placement in this free-space; check if it's the best so far
            # but continue searching other free-spaces for potentially better positions
        if best is None:
            unplaced.append(bi)
        else:
            _, fi, px, py, pz, pw, ph, pd = best
            # Replace the chosen free-space with its split sub-spaces
            new_subspaces = split_free_space(free_spaces[fi], px, py, pz, pw, ph, pd)
            free_spaces[fi] = new_subspaces[0] if new_subspaces else (0, 0, 0, 0, 0, 0)
            for s in new_subspaces[1:]:
                free_spaces.append(s)
            # Remove zero-volume spaces
            free_spaces = [s for s in free_spaces if s[3] > 0 and s[4] > 0 and s[5] > 0]
            placements[bi] = (0, px, py, pz, pw, ph, pd)
            placed_count += 1

    # Second pass: try to place unplaced boxes
    for bi in unplaced:
        if time.time() > deadline:
            break
        w, h, d = boxes[bi]
        best = None
        for fi, fs in enumerate(list(free_spaces)):
            if time.time() > deadline:
                break
            fx, fy, fz, fw, fh, fd = fs
            for (ow, oh, od) in orientations(w, h, d):
                if ow > fw or oh > fh or od > fd:
                    continue
                px, py, pz = fx, fy, fz
                sc = placement_score(px, py, pz, ow, oh, od)
                if best is None or sc < best[0]:
                    best = (sc, fi, px, py, pz, ow, oh, od)
        if best is not None:
            _, fi, px, py, pz, pw, ph, pd = best
            new_subspaces = split_free_space(free_spaces[fi], px, py, pz, pw, ph, pd)
            free_spaces[fi] = new_subspaces[0] if new_subspaces else (0, 0, 0, 0, 0, 0)
            for s in new_subspaces[1:]:
                free_spaces.append(s)
            free_spaces = [s for s in free_spaces if s[3] > 0 and s[4] > 0 and s[5] > 0]
            placements[bi] = (0, px, py, pz, pw, ph, pd)
            placed_count += 1

    # Fill any unplaced with -1
    for i in range(n):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)

    return placements
# EVOLVE-BLOCK-END
