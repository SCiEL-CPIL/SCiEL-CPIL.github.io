"""3D bin packing: best-fit placement into numbered bins, time-bounded local repair."""
import time

# EVOLVE-BLOCK-START
def pack(boxes, container):
    start = time.time()
    DEADLINE = 145.0  # safety margin under the 150s kill (task says 120s per eval)

    W, H, D = container
    box_vol = [w * h * d for (w, h, d) in boxes]
    n = len(boxes)

    # Sort boxes by volume descending (best-fit likes to place big items first)
    order = sorted(range(n), key=lambda i: box_vol[i], reverse=True)

    def overlaps(bi, bj, placed):
        # placed: list of (x,y,z,w,h,d) per box index
        ax, ay, az, aw, ah, ad = placed[bi]
        bx, by, bz, bw, bh, bd = placed[bj]
        return (ax < bx + bw and bx < ax + aw and
                ay < by + bh and by < ay + ah and
                az < bz + bd and bz < az + ad)

    def fits_and_free(bin_items, i, x, y, z):
        w, h, d = boxes[i]
        if x < 0 or y < 0 or z < 0:
            return False
        if x + w > W or y + h > H or z + d > D:
            return False
        for j in bin_items:
            if overlaps(i, j, _placed_ref):
                return False
        return True

    # We'll keep a per-bin list of placed item indices and a global placed array.
    bins = []          # list of lists of item indices
    placed = [None] * n  # (x,y,z,w,h,d) or None

    _placed_ref = placed

    def candidate_positions(bin_items):
        # Generate candidate (x,y,z) anchor points.
        # Start with origin; then for each placed box, three faces.
        cands = [(0, 0, 0)]
        for j in bin_items:
            px, py, pz, pw, ph, pd = placed[j]
            cands.append((px + pw, py, pz))
            cands.append((px, py + ph, pz))
            cands.append((px, py, pz + pd))
        return cands

    # Greedy best-fit: for each box (big first), try each bin, find a free spot
    # minimizing the "height" (z), then y, then x (pack toward origin).
    for i in order:
        if time.time() > DEADLINE:
            break
        w, h, d = boxes[i]
        best = None  # (score, bin_idx, x, y, z)
        for bi in range(len(bins)):
            if time.time() > DEADLINE:
                break
            for (cx, cy, cz) in candidate_positions(bins[bi]):
                if fits_and_free(bins[bi], i, cx, cy, cz):
                    score = (cz, cy, cx)
                    if best is None or score < best[0]:
                        best = (score, bi, cx, cy, cz)
        if best is not None:
            _, bi, cx, cy, cz = best
            bins[bi].append(i)
            placed[i] = (cx, cy, cz, w, h, d)
        else:
            # open a new bin
            bins.append([i])
            placed[i] = (0, 0, 0, w, h, d)

    # Local improvement: try to move a box to an earlier bin to reduce last-bin usage,
    # or to a lower position. Time-bounded.
    def try_move(i, target_bin, x, y, z):
        # temporarily remove i, check fit in target_bin
        w, h, d = boxes[i]
        if x + w > W or y + h > H or z + d > D:
            return False
        for j in target_bin:
            if j == i:
                continue
            if overlaps(i, j, placed):
                return False
        return True

    # Attempt to reduce number of bins by pulling boxes from the highest bin into lower bins.
    def compute_bins_used():
        used = set()
        for i in range(n):
            if placed[i] is not None:
                # find its bin
                pass
        return used

    # Rebuild bin membership from placed (we need bin index per item)
    item_bin = [None] * n
    for bi, lst in enumerate(bins):
        for i in lst:
            item_bin[i] = bi

    # Try to compact: for each bin from highest down, try to move its items to lower bins.
    improved = True
    while improved and time.time() < DEADLINE - 5:
        improved = False
        # find max bin index that has items
        max_bi = max((item_bin[i] for i in range(n) if item_bin[i] is not None), default=0)
        for bi in range(max_bi, -1, -1):
            if time.time() > DEADLINE - 5:
                break
            members = [i for i in range(n) if item_bin[i] == bi]
            for i in members:
                if time.time() > DEADLINE - 5:
                    break
                # try to move i to a lower bin
                for tb in range(bi):
                    if time.time() > DEADLINE - 5:
                        break
                    target = [j for j in range(n) if item_bin[j] == tb]
                    found = False
                    for (cx, cy, cz) in candidate_positions(target):
                        if try_move(i, target, cx, cy, cz):
                            # commit move
                            oldpos = placed[i]
                            placed[i] = (cx, cy, cz, boxes[i][0], boxes[i][1], boxes[i][2])
                            item_bin[i] = tb
                            improved = True
                            found = True
                            break
                    if found:
                        break
                if found:
                    break

    # Build final placements in original box order
    result = []
    for i in range(n):
        x, y, z, w, h, d = placed[i]
        result.append((item_bin[i], x, y, z, w, h, d))
    return result
# EVOLVE-BLOCK-END
