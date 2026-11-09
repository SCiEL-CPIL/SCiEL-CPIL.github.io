"""Best-fit 3D bin packing with candidate-position search and time-guarded polish."""
import time
import heapq

DEADLINE = 110.0  # wall-clock seconds; evaluator kills at 150s, keep margin


def _candidates(boxes, W, H, D, w, h, d):
    """Yield (x,y,z) candidate positions for a box of size (w,h,d).

    Candidate coords per axis = {0} union {p + size for each existing box}.
    Only positions fully inside the container and not overlapping any box
    are yielded.
    """
    xs = {0}
    ys = {0}
    zs = {0}
    for (bx, by, bz, bw, bh, bd) in boxes:
        if bx + bw <= W:
            xs.add(bx + bw)
        if by + bh <= H:
            ys.add(by + bh)
        if bz + bd <= D:
            zs.add(bz + bd)
    xs = sorted(x for x in xs if x + w <= W)
    ys = sorted(y for y in ys if y + h <= H)
    zs = sorted(z for z in zs if z + d <= D)
    if not xs or not ys or not zs:
        return
    for x in xs:
        for y in ys:
            for z in zs:
                ok = True
                for (bx, by, bz, bw, bh, bd) in boxes:
                    if (x < bx + bw and bx < x + w and
                            y < by + bh and by < y + h and
                            z < bz + bd and bz < z + d):
                        ok = False
                        break
                if ok:
                    yield (x, y, z)


def _best_position(boxes, W, H, D, w, h, d):
    """Return the best (x,y,z) for the box, minimizing wasted volume.

    Wasted volume = min over the 8 corners of the gap left in each direction.
    Returns None if no position fits.
    """
    best = None
    best_waste = None
    for (x, y, z) in _candidates(boxes, W, H, D, w, h, d):
        # wasted volume: for each of the 8 corner gaps, the product of the
        # three distances to the nearest boundary/box on each side.
        # Cheap surrogate: use the gap to the far wall on each axis from the
        # box's far face.
        waste = (W - (x + w)) * (H - (y + h)) * (D - (z + d))
        if best_waste is None or waste < best_waste:
            best_waste = waste
            best = (x, y, z)
    return best


def _fits_anywhere(boxes, W, H, D, w, h, d):
    """Return a valid (x,y,z) if the box fits anywhere, else None."""
    for (x, y, z) in _candidates(boxes, W, H, D, w, h, d):
        return (x, y, z)
    return None


def _occupied_volume(boxes):
    return sum(bw * bh * bd for (_, _, _, bw, bh, bd) in boxes)


def pack(boxes, container):
    W, H, D = container
    container_vol = W * H * D
    start = time.time()

    # Sort boxes by volume descending (larger first) for better packing,
    # but remember original index to return in original order.
    order = sorted(range(len(boxes)), key=lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2], reverse=True)

    # bins: list of lists of (x, y, z, w, h, d)
    bins = []
    # placement[i] = (bin_idx, x, y, z)
    placement = [None] * len(boxes)

    for idx in order:
        w, h, d = boxes[idx]
        best_bin = -1
        best_pos = None
        best_waste = None
        # Try existing bins (best fit)
        for bi, bboxes in enumerate(bins):
            pos = _best_position(bboxes, W, H, D, w, h, d)
            if pos is not None:
                x, y, z = pos
                waste = (W - (x + w)) * (H - (y + h)) * (D - (z + d))
                if best_waste is None or waste < best_waste:
                    best_waste = waste
                    best_bin = bi
                    best_pos = pos
        if best_bin >= 0:
            bins[best_bin].append((best_pos[0], best_pos[1], best_pos[2], w, h, d))
            placement[idx] = (best_bin, best_pos[0], best_pos[1], best_pos[2])
        else:
            # Open a new bin
            pos = _best_position([], W, H, D, w, h, d)
            if pos is None:
                # Should not happen if box fits container; fallback to 0,0,0
                pos = (0, 0, 0)
            bins.append([(pos[0], pos[1], pos[2], w, h, d)])
            placement[idx] = (len(bins) - 1, pos[0], pos[1], pos[2])

    def build_result():
        res = []
        for i in range(len(boxes)):
            w, h, d = boxes[i]
            bi, x, y, z = placement[i]
            res.append((bi, x, y, z, w, h, d))
        return res

    def score_of(placement):
        occ = set()
        last = -1
        for i in range(len(boxes)):
            bi, x, y, z = placement[i]
            occ.add(bi)
            if bi > last:
                last = bi
        B = len(occ)
        last_vol = 0
        for i in range(len(boxes)):
            bi, x, y, z = placement[i]
            if bi == last:
                last_vol += boxes[i][0] * boxes[i][1] * boxes[i][2]
        frac = last_vol / container_vol if container_vol else 0.0
        return B + 0.1 * (1 - frac)

    # Time-guarded polish: try to move misplaced boxes into better positions.
    cur_score = score_of(placement)
    while time.time() < DEADLINE - 5:
        improved = False
        for i in range(len(boxes)):
            if time.time() >= DEADLINE - 5:
                break
            bi, x, y, z = placement[i]
            w, h, d = boxes[i]
            # current wasted volume at current pos
            cur_waste = (W - (x + w)) * (H - (y + h)) * (D - (z + d))
            # try to find a better position in same bin
            bboxes = [(bx, by, bz, bw, bh, bd)
                      for j in range(len(boxes)) if placement[j][0] == bi and j != i]
            pos = _best_position(bboxes, W, H, D, w, h, d)
            if pos is not None:
                nw, nh, nz = pos
                new_waste = (W - (nw + w)) * (H - (nh + h)) * (D - (nz + d))
                if new_waste < cur_waste - 1e-9:
                    placement[i] = (bi, nw, nh, nz)
                    improved = True
        if not improved:
            break

    return build_result()
