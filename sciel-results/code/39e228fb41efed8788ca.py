"""3D Bin Packing solver: best-fit over corner candidates with multi-order local search."""
import time

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    start = time.time()
    DEADLINE = 110.0  # wall-clock safety margin (evaluator kills at 150s)

    # A box is identified by its index; dims are fixed (no rotation).
    dims = [tuple(b) for b in boxes]
    vol = [w * h * d for (w, h, d) in dims]

    # ---------- geometry helpers ----------
    def overlaps(a, b):
        # a,b = (x,y,z,w,h,d)
        ax, ay, az, aw, ah, ad = a
        bx, by, bz, bw, bh, bd = b
        return (ax < bx + bw and bx < ax + aw and
                ay < by + bh and by < ay + ah and
                az < bz + bd and bz < az + ad)

    def fits(pos, occ):
        x, y, z, w, h, d = pos
        if x + w > W or y + h > H or z + d > D:
            return False
        for o in occ:
            if overlaps(pos, o):
                return False
        return True

    def corners(occ):
        # candidate anchor positions from existing box faces
        xs = {0}; ys = {0}; zs = {0}
        for (x, y, z, w, h, d) in occ:
            xs.add(x); xs.add(x + w)
            ys.add(y); ys.add(y + h)
            zs.add(z); zs.add(z + d)
        cands = []
        for x in xs:
            if x >= W:
                continue
            for y in ys:
                if y >= H:
                    continue
                for z in zs:
                    if z >= D:
                        continue
                    cands.append((x, y, z))
        return cands

    # ---------- single-bin packer (returns list of positions or None) ----------
    def pack_bin(order, occ, used, start_idx=0):
        # occ: list of placed positions; used: set of placed box indices
        # place boxes in order (skipping already-used), best-fit corner
        for i in order:
            if i in used:
                continue
            w, h, d = dims[i]
            best = None
            best_key = None
            for (cx, cy, cz) in corners(occ):
                pos = (cx, cy, cz, w, h, d)
                if not fits(pos, occ):
                    continue
                key = (cx, cy, cz)  # prefer low x,y,z
                if best_key is None or key < best_key:
                    best_key = key
                    best = pos
            if best is None:
                return None
            occ.append(best)
            used.add(i)
        return occ

    # ---------- scoring ----------
    def score_of(assignment):
        # assignment: dict box_index -> (bin, x, y, z)
        bins = {}
        for i, (b, x, y, z) in assignment.items():
            bins.setdefault(b, []).append((x, y, z, dims[i][0], dims[i][1], dims[i][2]))
        B = len(bins)
        if B == 0:
            return (0.0, 0, 0.0)
        last = max(bins.keys())
        lv = sum(w * h * d for (_, _, _, w, h, d) in bins[last])
        frac = lv / (W * H * D)
        val = B + 0.1 * (1.0 - frac)
        return (val, B, frac)

    def build_assignment(per_bin_order):
        # per_bin_order: list of lists of box indices (one per bin)
        assignment = {}
        for b, order in enumerate(per_bin_order):
            occ = []
            used = set()
            res = pack_bin(order, occ, used)
            if res is None:
                return None
            for pos in res:
                # need to know which box this position belongs to
                pass
            # rebuild mapping: re-run to record index per position
            occ2 = []
            used2 = set()
            idx_map = []
            for i in order:
                if i in used2:
                    continue
                w, h, d = dims[i]
                best = None; best_key = None
                for (cx, cy, cz) in corners(occ2):
                    pos = (cx, cy, cz, w, h, d)
                    if not fits(pos, occ2):
                        continue
                    key = (cx, cy, cz)
                    if best_key is None or key < best_key:
                        best_key = key; best = pos
                if best is None:
                    return None
                occ2.append(best)
                used2.add(i)
                idx_map.append(i)
            for i, pos in zip(idx_map, occ2):
                assignment[i] = (b, pos[0], pos[1], pos[2])
        return assignment

    # ---------- multi-bin construction ----------
    def construct(order):
        # greedy: fill bins one at a time using best-fit, next bin when a box doesn't fit
        per_bin = []
        remaining = list(order)
        while remaining:
            # try to place as many as possible in a fresh bin
            occ = []
            used = set()
            new_remaining = []
            for i in remaining:
                w, h, d = dims[i]
                best = None; best_key = None
                for (cx, cy, cz) in corners(occ):
                    pos = (cx, cy, cz, w, h, d)
                    if not fits(pos, occ):
                        continue
                    key = (cx, cy, cz)
                    if best_key is None or key < best_key:
                        best_key = key; best = pos
                if best is None:
                    new_remaining.append(i)
                    continue
                occ.append(best)
                used.add(i)
            per_bin.append(remaining)  # placeholder, rebuild below
            # actually record which went in
            placed = [i for i in remaining if i in used]
            per_bin[-1] = placed
            remaining = new_remaining
            if time.time() > DEADLINE:
                break
        return per_bin

    def construct_and_score(order):
        per_bin = construct(order)
        if not per_bin:
            return None
        assignment = build_assignment(per_bin)
        if assignment is None:
            return None
        val, B, frac = score_of(assignment)
        return (val, assignment)

    # ---------- local search: move a box to another bin ----------
    def try_move(assignment, per_bin):
        # attempt to reduce B or improve last-bin fraction by moving boxes
        # simple: try moving one box from last bin to an earlier bin if it fits
        bins = {}
        for i, (b, x, y, z) in assignment.items():
            bins.setdefault(b, []).append(i)
        if not bins:
            return assignment, per_bin
        last = max(bins.keys())
        improved = False
        # try to relocate each box in last bin to an earlier bin
        for i in list(bins[last]):
            w, h, d = dims[i]
            for b in sorted(bins.keys()):
                if b == last:
                    continue
                # test fit in bin b (rebuild occ for bin b)
                occ = [(assignment[j][1], assignment[j][2], assignment[j][3],
                        dims[j][0], dims[j][1], dims[j][2]) for j in bins[b]]
                best = None; best_key = None
                for (cx, cy, cz) in corners(occ):
                    pos = (cx, cy, cz, w, h, d)
                    if not fits(pos, occ):
                        continue
                    key = (cx, cy, cz)
                    if best_key is None or key < best_key:
                        best_key = key; best = pos
                if best is not None:
                    assignment[i] = (b, best[0], best[1], best[2])
                    bins[b].append(i)
                    bins[last].remove(i)
                    improved = True
                    break
        # rebuild per_bin from bins
        per_bin = []
        for b in sorted(bins.keys()):
            per_bin.append(bins[b])
        # drop empty bins at the end
        while per_bin and not per_bin[-1]:
            per_bin.pop()
        return assignment, per_bin

    # ---------- candidate orders ----------
    def make_orders():
        idxs = list(range(n))
        orders = []
        # volume descending
        orders.append(sorted(idxs, key=lambda i: -vol[i]))
        # max-dim descending
        orders.append(sorted(idxs, key=lambda i: -max(dims[i])))
        # volume ascending (fill small first sometimes helps last bin)
        orders.append(sorted(idxs, key=lambda i: vol[i]))
        # original
        orders.append(idxs)
        # longest side
        orders.append(sorted(idxs, key=lambda i: -max(dims[i])))
        return orders

    best_val = None
    best_assignment = None
    best_per_bin = None

    for order in make_orders():
        if time.time() > DEADLINE:
            break
        res = construct_and_score(order)
        if res is None:
            continue
        val, assignment = res
        per_bin = construct(order)
        # local improvement
        assignment, per_bin = try_move(assignment, per_bin)
        val2, _, _ = score_of(assignment)
        if best_val is None or val2 < best_val:
            best_val = val2
            best_assignment = assignment
            best_per_bin = per_bin

    # If nothing found (shouldn't happen), fall back to simple shelf
    if best_assignment is None:
        placements = []
        b = x = y = z = row_h = layer_d = 0
        for (w, h, d) in boxes:
            if x + w > W:
                x, y, row_h = 0, y + row_h, 0
            if y + h > H:
                y, z, layer_d = 0, z + layer_d, 0
                x, row_h = 0, 0
            if z + d > D:
                b, x, y, z, row_h, layer_d = b + 1, 0, 0, 0, 0, 0
            placements.append((b, x, y, z, w, h, d))
            x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)
        return placements

    # emit placements in original box order
    placements = []
    for i in range(n):
        b, x, y, z = best_assignment[i]
        w, h, d = dims[i]
        placements.append((b, x, y, z, w, h, d))
    return placements
