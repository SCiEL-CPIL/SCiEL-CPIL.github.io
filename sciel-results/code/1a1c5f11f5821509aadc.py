import time

# EVOLVE-BLOCK-START
def pack(boxes, container):
    W, H, D = container
    if not boxes:
        return []
    n = len(boxes)
    start = time.time()
    deadline = start + 140.0

    # FFD: sort by volume descending
    order = sorted(range(n), key=lambda i: (boxes[i][0]*boxes[i][1]*boxes[i][2],
                                            max(boxes[i])), reverse=True)

    bins = []  # each bin: list of (x, y, z, w, h, d, orig_idx)

    def find_position(placed, w, h, d):
        xs = [0]
        for (px, py, pz, pw, ph, pd, oi) in placed:
            xs.append(px + pw)
        ys = [0]
        for (px, py, pz, pw, ph, pd, oi) in placed:
            ys.append(py + ph)
        zs = [0]
        for (px, py, pz, pw, ph, pd, oi) in placed:
            zs.append(pz + pd)
        best = None
        best_key = None
        for x in xs:
            if x + w > W:
                continue
            for y in ys:
                if y + h > H:
                    continue
                for z in zs:
                    if z + d > D:
                        continue
                    ok = True
                    for (px, py, pz, pw, ph, pd, oi) in placed:
                        if (x < px + pw and x + w > px and
                            y < py + ph and y + h > py and
                            z < pz + pd and z + d > pz):
                            ok = False
                            break
                    if ok:
                        key = (x + y + z, x, y, z)
                        if best_key is None or key < best_key:
                            best = (x, y, z)
                            best_key = key
        return best

    placements = [None] * n
    for idx in order:
        if time.time() > deadline:
            break
        w, h, d = boxes[idx]
        if w > W or h > H or d > D:
            continue
        best_bin = -1
        best_pos = None
        best_fit = None
        for b in range(len(bins)):
            pos = find_position(bins[b], w, h, d)
            if pos is not None:
                key = (pos[0] + pos[1] + pos[2], pos[0], pos[1], pos[2])
                if best_fit is None or key < best_fit:
                    best_fit = key
                    best_bin = b
                    best_pos = pos
        if best_bin >= 0:
            bins[best_bin].append((best_pos[0], best_pos[1], best_pos[2], w, h, d, idx))
            placements[idx] = (best_bin, best_pos[0], best_pos[1], best_pos[2], w, h, d)
        else:
            b = len(bins)
            bins.append([(0, 0, 0, w, h, d, idx)])
            placements[idx] = (b, 0, 0, 0, w, h, d)

    # Handle any boxes that weren't placed (deadline or too big)
    for idx in range(n):
        if placements[idx] is None:
            # Put in a new bin at origin (fallback)
            b = len(bins)
            w, h, d = boxes[idx]
            bins.append([(0, 0, 0, w, h, d, idx)])
            placements[idx] = (b, 0, 0, 0, w, h, d)

    return placements
# EVOLVE-BLOCK-END
