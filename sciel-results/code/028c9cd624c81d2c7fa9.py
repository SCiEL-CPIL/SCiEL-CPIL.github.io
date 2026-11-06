"""3D Container Loading: skyline-anchor + occupancy-grid packer with deadline guard."""
import time
import itertools

# EVOLVE-BLOCK-START
def load(boxes, container):
    start = time.time()
    deadline = start + 105.0  # safety margin vs 120s (evaluator kills at 150s)
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

    # If container is too large for a unit grid, fall back
    if W * H * D > 8_000_000:
        return fallback()

    # Order boxes: largest volume first, but keep original index
    order = sorted(range(n), key=lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2], reverse=True)

    # Occupancy grid (0 = free, 1 = occupied)
    occ = bytearray(W * H * D)

    def idx(x, y, z):
        return x + W * (y + H * z)

    def fits(x, y, z, w, h, d):
        if x + w > W or y + h > H or z + d > D:
            return False
        base = x + W * (y + H * z)
        for dz in range(d):
            row = base + W * H * dz
            for dy in range(h):
                r = row + W * dy
                for dx in range(w):
                    if occ[r + dx]:
                        return False
        return True

    def place(x, y, z, w, h, d):
        base = x + W * (y + H * z)
        for dz in range(d):
            row = base + W * H * dz
            for dy in range(h):
                r = row + W * dy
                for dx in range(w):
                    occ[r + dx] = 1

    # Skyline anchors: set of (x,y,z) candidate positions
    anchors = [(0, 0, 0)]

    def add_anchors(x, y, z, w, h, d):
        # New skyline points: top faces and side faces of placed box
        for nx, ny, nz in (
            (x + w, y, z), (x, y + h, z), (x, y, z + d),
            (x + w, y + h, z), (x + w, y, z + d), (x, y + h, z + d),
            (x + w, y + h, z + d),
        ):
            if nx <= W and ny <= H and nz <= D:
                anchors.append((nx, ny, nz))

    # Orientation generator: all 6 permutations
    def orientations(w, h, d):
        seen = set()
        for perm in itertools.permutations((w, h, d)):
            if perm not in seen:
                seen.add(perm)
                yield perm

    placements = [None] * n
    placed_count = 0

    for bi in order:
        if time.time() > deadline:
            break
        w, h, d = boxes[bi]
        best = None  # (score, ox, oy, oz, ow, oh, od)
        # Dedup anchors for speed
        for (ax, ay, az) in set(anchors):
            if time.time() > deadline:
                break
            for (ow, oh, od) in orientations(w, h, d):
                if not fits(ax, ay, az, ow, oh, od):
                    continue
                # Score: prefer low z, then low y, then low x
                score = (az, ay, ax)
                if best is None or score < best[0]:
                    best = (score, ax, ay, az, ow, oh, od)
        if best is None:
            placements[bi] = (-1, 0, 0, 0, w, h, d)
        else:
            _, px, py, pz, pw, ph, pd = best
            place(px, py, pz, pw, ph, pd)
            placements[bi] = (0, px, py, pz, pw, ph, pd)
            add_anchors(px, py, pz, pw, ph, pd)
            placed_count += 1

    # Fill any unplaced (deadline hit) with -1
    for i in range(n):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)

    return placements
# EVOLVE-BLOCK-END
