"""Combined program: BLF coordinate-scan placement core + multi-strategy diversification
+ time-budgeted local-search polish (re-insertion)."""
import time
import math
import random
import itertools

def _get_rotations(w, h, d):
    """All unique rotations, sorted to prefer low z-extent (stacking), then low y, then low x."""
    rots = set()
    for perm in itertools.permutations((w, h, d)):
        rots.add(perm)
    rots = sorted(rots, key=lambda r: (r[2], r[1], r[0]))
    return list(rots)


def _check_overlap(placed, x, y, z, w, h, d):
    """Return True if the box overlaps any placed box."""
    for (px, py, pz, pw, ph, pd) in placed:
        if (x < px + pw and x + w > px and
                y < py + ph and y + h > py and
                z < pz + pd and z + d > pz):
            return True
    return False


def _place_blf(boxes, container, order, box_rotations, deadline, W, H, D):
    """
    Place boxes in the given order using Bottom-Left-Farthest (BLF) coordinate scan.
    For each box, scan z (asc), y (asc), x (asc); the first valid (z,y,x) across
    rotations is the BLF position. Early termination on first valid per position.
    Returns (result, placed) where result[i] = (bin,x,y,z,w,h,d) and
    placed is a list of (x,y,z,w,h,d) for placed boxes (in placement order).
    """
    n = len(boxes)
    placed = []
    result = [None] * n

    for i in order:
        if time.time() > deadline:
            break
        bw, bh, bd = boxes[i]
        rots = box_rotations[i]

        # Candidate coordinates: 0 plus right/back/top faces of placed boxes.
        xs = [0]
        ys = [0]
        zs = [0]
        for (px, py, pz, pw, ph, pd) in placed:
            if px + pw <= W:
                xs.append(px + pw)
            if py + ph <= H:
                ys.append(py + ph)
            if pz + pd <= D:
                zs.append(pz + pd)

        xs = sorted(set(x for x in xs if x <= W))
        ys = sorted(set(y for y in ys if y <= H))
        zs = sorted(set(z for z in zs if z <= D))

        best_key = None
        best_pos = None

        # BLF: scan z asc, y asc, x asc. First valid position wins.
        for z in zs:
            for y in ys:
                for x in xs:
                    found = False
                    for (rw, rh, rd) in rots:
                        if x + rw > W or y + rh > H or z + rd > D:
                            continue
                        if _check_overlap(placed, x, y, z, rw, rh, rd):
                            continue
                        # Valid position found.
                        best_key = (z, y, x)
                        best_pos = (x, y, z, rw, rh, rd)
                        found = True
                        break
                    if found:
                        break
                if found:
                    break
            if found:
                break

        if best_pos is not None:
            x, y, z, rw, rh, rd = best_pos
            placed.append((x, y, z, rw, rh, rd))
            result[i] = (0, x, y, z, rw, rh, rd)
        else:
            result[i] = (-1, 0, 0, 0, bw, bh, bd)

    # Fill unprocessed
    for i in range(n):
        if result[i] is None:
            result[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])

    return result, placed


def _filled_volume(result):
    return sum(r[4] * r[5] * r[6] for r in result if r[0] == 0)


def _local_search(boxes, container, result, box_rotations, deadline, W, H, D):
    """
    Time-budgeted re-insertion local search.
    Repeatedly remove a placed box and re-place it via BLF; keep the move only
    if it allows a previously-failed box to fit (i.e., increases filled volume),
    or accept moves that keep the same volume but improve packing density order.
    Returns improved result.
    """
    n = len(boxes)
    # Build placed list consistent with result
    placed = []
    for i in range(n):
        r = result[i]
        if r[0] == 0:
            placed.append((r[1], r[2], r[3], r[4], r[5], r[6]))

    cur_vol = _filled_volume(result)
    # Order to try removing: largest first (they create the most space)
    placed_order = sorted(range(n),
                          key=lambda i: (-(result[i][4]*result[i][5]*result[i][6]) if result[i][0] == 0 else 0))

    improved = True
    passes = 0
    while improved and time.time() < deadline and passes < 30:
        improved = False
        passes += 1
        for i in placed_order:
            if time.time() > deadline:
                break
            if result[i][0] != 0:
                continue
            # Remove box i from placed
            bx, by, bz, bw, bh, bd = result[i][1], result[i][2], result[i][3], result[i][4], result[i][5], result[i][6]
            # find and remove from placed
            target = (bx, by, bz, bw, bh, bd)
            try:
                pidx = placed.index(target)
                placed.pop(pidx)
            except ValueError:
                continue

            # Re-place box i via BLF with current placed
            rots = box_rotations[i]
            xs = [0]
            ys = [0]
            zs = [0]
            for (px, py, pz, pw, ph, pd) in placed:
                if px + pw <= W:
                    xs.append(px + pw)
                if py + ph <= H:
                    ys.append(py + ph)
                if pz + pd <= D:
                    zs.append(pz + pd)
            xs = sorted(set(x for x in xs if x <= W))
            ys = sorted(set(y for y in ys if y <= H))
            zs = sorted(set(z for z in zs if z <= D))

            best_key = None
            best_pos = None
            for z in zs:
                for y in ys:
                    for x in xs:
                        found = False
                        for (rw, rh, rd) in rots:
                            if x + rw > W or y + rh > H or z + rd > D:
                                continue
                            if _check_overlap(placed, x, y, z, rw, rh, rd):
                                continue
                            best_key = (z, y, x)
                            best_pos = (x, y, z, rw, rh, rd)
                            found = True
                            break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break

            if best_pos is None:
                # Can't re-place; restore original
                placed.append(target)
                continue

            # Accept the new placement (it may enable others to fit later)
            x, y, z, rw, rh, rd = best_pos
            placed.append((x, y, z, rw, rh, rd))
            result[i] = (0, x, y, z, rw, rh, rd)
            improved = True

            # Now try to place any currently-failed box into the freed space
            for j in range(n):
                if result[j][0] != -1:
                    continue
                jw, jh, jd = boxes[j]
                jrots = box_rotations[j]
                # rebuild coords
                xs2 = [0]
                ys2 = [0]
                zs2 = [0]
                for (px, py, pz, pw, ph, pd) in placed:
                    if px + pw <= W:
                        xs2.append(px + pw)
                    if py + ph <= H:
                        ys2.append(py + ph)
                    if pz + pd <= D:
                        zs2.append(pz + pd)
                xs2 = sorted(set(x for x in xs2 if x <= W))
                ys2 = sorted(set(y for y in ys2 if y <= H))
                zs2 = sorted(set(z for z in zs2 if z <= D))
                for z2 in zs2:
                    done = False
                    for y2 in ys2:
                        if done:
                            break
                        for x2 in xs2:
                            if done:
                                break
                            for (rw2, rh2, rd2) in jrots:
                                if x2 + rw2 > W or y2 + rh2 > H or z2 + rd2 > D:
                                    continue
                                if _check_overlap(placed, x2, y2, z2, rw2, rh2, rd2):
                                    continue
                                placed.append((x2, y2, z2, rw2, rh2, rd2))
                                result[j] = (0, x2, y2, z2, rw2, rh2, rd2)
                                cur_vol += rw2 * rh2 * rd2
                                improved = True
                                done = True
                                break
                        if done:
                            break
                    if done:
                        break
                if time.time() > deadline:
                    break

    return result


def load(boxes, container):
    """
    Place boxes into container. Returns list of (bin, x, y, z, w, h, d) for each box.
    bin=0 for loaded, bin=-1 for left out.
    """
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    n = len(boxes)
    if n == 0:
        return []

    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s kill

    box_rotations = [_get_rotations(w, h, d) for w, h, d in boxes]

    # Sorting strategies (index orders)
    def order_volume_desc():
        return sorted(range(n), key=lambda i: -(boxes[i][0]*boxes[i][1]*boxes[i][2]))

    def order_max_dim_desc():
        return sorted(range(n), key=lambda i: -max(boxes[i]))

    def order_sum_dim_desc():
        return sorted(range(n), key=lambda i: -(boxes[i][0]+boxes[i][1]+boxes[i][2]))

    def order_min_dim_desc():
        return sorted(range(n), key=lambda i: -min(boxes[i]))

    strategies = [order_volume_desc, order_max_dim_desc, order_sum_dim_desc, order_min_dim_desc]

    best_result = None
    best_vol = -1

    for strat in strategies:
        if time.time() > deadline:
            break
        order = strat()
        result, placed = _place_blf(boxes, container, order, box_rotations, deadline, W, H, D)
        vol = _filled_volume(result)
        if vol > best_vol:
            best_vol = vol
            best_result = result

    if best_result is None:
        best_result = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    # Local search polish on the best result if time remains
    if time.time() < deadline - 5.0:
        polished = _local_search(boxes, container, best_result, box_rotations, deadline - 3.0, W, H, D)
        if _filled_volume(polished) >= _filled_volume(best_result):
            best_result = polished

    return best_result
