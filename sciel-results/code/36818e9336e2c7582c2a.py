"""3D Bin Packing - Best-Fit Decreasing with empty-space management + consolidation."""
import time
from itertools import product

# EVOLVE-BLOCK-START

def _can_place(box, pos, placed, W, H, D):
    """Check if box (w,h,d) at pos (x,y,z) fits and doesn't overlap placed boxes."""
    bx, by, bz = pos
    w, h, d = box
    if bx + w > W or by + h > H or bz + d > D:
        return False
    for (px, py, pz, pw, ph, pd) in placed:
        # overlap if intervals intersect on all 3 axes
        if (bx < px + pw and px < bx + w and
            by < py + ph and py < by + h and
            bz < pz + pd and pz < bz + d):
            return False
    return True

def _gen_candidates(pos, w, h, d, W, H, D, placed):
    """Generate candidate positions to try, prioritizing corners (min x+y+z)."""
    cands = [pos]
    # Also try snapping to boundaries of placed boxes
    xs = [pos[0]]
    ys = [pos[1]]
    zs = [pos[2]]
    for (px, py, pz, pw, ph, pd) in placed:
        if px + pw <= W:
            xs.append(px + pw)
        if py + ph <= H:
            ys.append(py + ph)
        if pz + pd <= D:
            zs.append(pz + pd)
    xs = sorted(set(x for x in xs if x + w <= W))
    ys = sorted(set(y for y in ys if y + h <= H))
    zs = sorted(set(z for z in zs if z + d <= D))
    for x in xs:
        for y in ys:
            for z in zs:
                c = (x, y, z)
                if c not in cands:
                    cands.append(c)
    # Sort by compactness (minimize x+y+z), then by x, y, z
    cands.sort(key=lambda c: (c[0] + c[1] + c[2], c[0], c[1], c[2]))
    return cands

def _place_box(bins, box, W, H, D, strategy="best_fit"):
    """Place one box into the best bin. Returns (bin_idx, pos) or None."""
    w, h, d = box
    best = None  # (score, bin_idx, pos)
    for bi, (placed, corners) in enumerate(bins):
        # try each corner as a starting candidate
        tried = set()
        for corner in sorted(corners, key=lambda c: (c[0]+c[1]+c[2], c[0], c[1], c[2])):
            cands = _gen_candidates(corner, w, h, d, W, H, D, placed)
            for pos in cands:
                if pos in tried:
                    continue
                tried.add(pos)
                if _can_place(box, pos, placed, W, H, D):
                    # score: prefer lower x+y+z (compact), then fewer bins
                    s = pos[0] + pos[1] + pos[2]
                    if best is None or s < best[0]:
                        best = (s, bi, pos)
                    break  # first valid per corner is the most compact for that corner
        if best is not None:
            # We found a placement; but check if another bin is better
            # Actually best_fit: pick the bin+pos with min score. Continue checking bins.
            pass
    if best is None:
        return None
    return best[1], best[2]

def _try_order(boxes, container, order_key, deadline):
    """Run BFD with a given ordering. Returns (placements, num_bins)."""
    W, H, D = container
    idx = list(range(len(boxes)))
    idx.sort(key=lambda i: order_key(boxes[i]), reverse=True)
    bins = []  # list of (placed_list, corners_set)
    placements = [None] * len(boxes)
    for i in idx:
        if time.time() > deadline:
            return None, None
        box = boxes[i]
        res = _place_box(bins, box, W, H, D)
        if res is None:
            # open a new bin
            bi = len(bins)
            bins.append(([], {(0, 0, 0)}))
            res = (bi, (0, 0, 0))
        bi, pos = res
        placed, corners = bins[bi]
        # verify (safety)
        if not _can_place(box, pos, placed, W, H, D):
            return None, None
        placed.append((pos[0], pos[1], pos[2], box[0], box[1], box[2]))
        placements[i] = (bi, pos[0], pos[1], pos[2], box[0], box[1], box[2])
        # update corners: add new free corners from this placement
        bx, by, bz = pos
        w, h, d = box
        new_corners = set()
        if bx + w <= W:
            new_corners.add((bx + w, by, bz))
        if by + h <= H:
            new_corners.add((bx, by + h, bz))
        if bz + d <= D:
            new_corners.add((bx, by, bz + d))
        # also the box's own top-right-back corner combos
        for cx in (bx, bx + w) if bx + w <= W else (bx,):
            for cy in (by, by + h) if by + h <= H else (by,):
                for cz in (bz, bz + d) if bz + d <= D else (bz,):
                    if (cx, cy, cz) != (bx, by, bz):
                        new_corners.add((cx, cy, cz))
        corners |= new_corners
        # prune dominated corners: remove corner a if corner b has b<=a on all axes and b != a
        cl = list(corners)
        keep = set()
        for a in cl:
            dominated = False
            for b in cl:
                if a != b and b[0] <= a[0] and b[1] <= a[1] and b[2] <= a[2]:
                    dominated = True
                    break
            if not dominated:
                keep.add(a)
        bins[bi] = (placed, keep)
    num_bins = len(bins)
    return placements, num_bins

def _consolidate(placements, boxes, container, deadline):
    """Try to move boxes from the last bin into earlier bins. Returns new placements or original."""
    W, H, D = container
    if time.time() > deadline:
        return placements
    # find max bin
    if not placements:
        return placements
    max_bin = max(p[0] for p in placements)
    # boxes in last bin
    last_boxes = [i for i in range(len(placements)) if placements[i][0] == max_bin]
    if not last_boxes:
        return placements
    # build per-bin placed lists
    from collections import defaultdict
    bin_placed = defaultdict(list)
    for i, p in enumerate(placements):
        b = p[0]
        bin_placed[b].append((p[1], p[2], p[3], boxes[i][0], boxes[i][1], boxes[i][2]))
    new_placements = [list(p) for p in placements]
    # try to move each last-bin box (largest first) into an earlier bin
    last_boxes.sort(key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    for i in last_boxes:
        if time.time() > deadline:
            break
        box = boxes[i]
        moved = False
        for target_bin in sorted([b for b in bin_placed if b < max_bin], reverse=True):
            placed = bin_placed[target_bin]
            # generate candidate positions
            corners = {(0,0,0)}
            for (px, py, pz, pw, ph, pd) in placed:
                for cx in (px, px+pw) if px+pw <= W else (px,):
                    for cy in (py, py+ph) if py+ph <= H else (py,):
                        for cz in (pz, pz+pd) if pz+pd <= D else (pz,):
                            corners.add((cx, cy, cz))
            # prune
            cl = list(corners)
            keep = set()
            for a in cl:
                dom = False
                for b in cl:
                    if a != b and b[0] <= a[0] and b[1] <= a[1] and b[2] <= a[2]:
                        dom = True
                        break
                if not dom:
                    keep.add(a)
            corners = keep
            for corner in sorted(corners, key=lambda c: (c[0]+c[1]+c[2], c[0], c[1], c[2])):
                cands = _gen_candidates(corner, box[0], box[1], box[2], W, H, D, placed)
                for pos in cands:
                    if _can_place(box, pos, placed, W, H, D):
                        # move it
                        bin_placed[target_bin].append((pos[0], pos[1], pos[2], box[0], box[1], box[2]))
                        new_placements[i] = [target_bin, pos[0], pos[1], pos[2], box[0], box[1], box[2]]
                        moved = True
                        break
                if moved:
                    break
            if moved:
                break
        if not moved:
            # leave in last bin
            pass
    # if no box in last bin now, max_bin effectively reduced
    if all(p[0] < max_bin for p in new_placements):
        # last bin empty; relabel to be consecutive is not required, but reduce max label
        # keep labels as is; B counts occupied bins
        pass
    return [tuple(p) for p in new_placements]

def _score(placements, boxes, container):
    """Compute score: mean B + 0.1*(1 - last-bin occupied volume / container volume).
    Here we return B and last-bin fraction for a single instance."""
    W, H, D = container
    cont_vol = W * H * D
    if not placements:
        return 0, 0.0
    occupied = set(p[0] for p in placements)
    B = len(occupied)
    max_bin = max(occupied)
    last_vol = sum(boxes[i][0]*boxes[i][1]*boxes[i][2] for i in range(len(placements)) if placements[i][0] == max_bin)
    frac = last_vol / cont_vol if cont_vol > 0 else 1.0
    return B, frac

def pack(boxes, container):
    """Pack boxes into numbered bins. Returns list of (bin, x, y, z, w, h, d)."""
    start = time.time()
    deadline = start + 140.0  # safety margin before 150s kill
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    # ordering strategies
    def key_vol(b):
        return b[0]*b[1]*b[2]
    def key_max(b):
        return max(b)
    def key_aspect(b):
        # prefer boxes that are more "blocky" / fill well; use product of dims
        return b[0]*b[1]*b[2]
    def key_sum(b):
        return b[0]+b[1]+b[2]

    strategies = [key_vol, key_max, key_sum]
    best_placements = None
    best_score = float('inf')
    for key in strategies:
        if time.time() > deadline:
            break
        placements, num_bins = _try_order(boxes, container, key, deadline)
        if placements is None:
            continue
        # consolidate
        placements = _consolidate(placements, boxes, container, deadline)
        B, frac = _score(placements, boxes, container)
        score = B + 0.1 * (1 - frac)
        if score < best_score:
            best_score = score
            best_placements = [tuple(p) for p in placements]
    if best_placements is None:
        # fallback: simple shelf
        W, H, D = container
        placements, b, x, y, z, row_h, layer_d = [], 0, 0, 0, 0, 0, 0
        for w, h, d in boxes:
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
    return best_placements
# EVOLVE-BLOCK-END
