"""3D Container Loading: largest-empty-space (distance-transform) greedy
with a 3D sweep-line collision filter, multiple orders + gap local search."""
import time
import itertools
import numpy as np

def _orientations(w, h, d):
    seen = set()
    result = []
    for perm in itertools.permutations((w, h, d)):
        if perm not in seen:
            seen.add(perm)
            result.append(perm)
    return result

def _sweep_line_check(intervals, pos, length, axis, max_coord):
    """3D sweep-line: O(K log K) check that a box (start at 'pos', extent 'length'
    along 'axis', fixed 'intervals' on the other two axes) is empty & in bounds.
    intervals = list of (lo,hi,boxid) for the two perpendicular axes."""
    if pos < 0 or pos + length > max_coord:
        return False
    ev = []
    for (lo, hi, bid) in intervals:
        ev.append((lo, 1, bid))
        ev.append((hi, -1, bid))
    ev.sort(key=lambda t: (t[0], t[1]))
    active = set()
    ok = True
    for (t, typ, bid) in ev:
        if typ == 1:
            if bid in active:
                continue
            active.add(bid)
            if pos < t and pos + length > t:
                ok = False
                break
        else:
            active.discard(bid)
    return ok

def _empty_and_inbounds(x, y, z, ow, oh, od, placed_ids, placed_data, W, H, D):
    # Build interval lists for the two non-moving axes and sweep along z.
    # Intervals in Y from boxes overlapping X and Z; sweep along Z.
    yint = []
    zint = []
    for bid in placed_ids:
        (bx, by, bz, bw, bh, bd) = placed_data[bid]
        # overlap in X?
        if x + ow <= bx or bx + bw <= x:
            continue
        # overlap in Z?
        if z + od <= bz or bz + bd <= z:
            continue
        yint.append((by, by + bh, bid))
    if yint:
        if not _sweep_line_check(yint, y, oh, 1, H):
            return False
    # Now need Z emptiness: boxes overlapping X and Y, sweep along Z.
    for bid in placed_ids:
        (bx, by, bz, bw, bh, bd) = placed_data[bid]
        if x + ow <= bx or bx + bw <= x:
            continue
        if y + oh <= by or by + bh <= y:
            continue
        zint.append((bz, bz + bd, bid))
    if zint:
        if not _sweep_line_check(zint, z, od, 2, D):
            return False
    return True

def _build_distance_grid(placed_data, placed_ids, W, H, D, step):
    """Return (grid[ny][nz][nx], xs, ys, zs, step). grid value = max L_inf clearance
    (distance to nearest obstacle or boundary) at each cell center."""
    nx = max(2, int(np.ceil(W / step)))
    ny = max(2, int(np.ceil(H / step)))
    nz = max(2, int(np.ceil(D / step)))
    xs = (np.arange(nx) + 0.5) * step
    ys = (np.arange(ny) + 0.5) * step
    zs = (np.arange(nz) + 0.5) * step

    dist = np.full((ny, nz, nx), np.inf, dtype=np.float64)
    for bid in placed_ids:
        (bx, by, bz, bw, bh, bd) = placed_data[bid]
        ix0 = int(np.clip(bx / step, 0, nx - 1))
        ix1 = int(np.clip((bx + bw) / step, 0, nx - 1))
        iy0 = int(np.clip(by / step, 0, ny - 1))
        iy1 = int(np.clip((by + bh) / step, 0, ny - 1))
        iz0 = int(np.clip(bz / step, 0, nz - 1))
        iz1 = int(np.clip((bz + bd) / step, 0, nz - 1))
        for ix in range(ix0, ix1 + 1):
            cx = xs[ix]
            dx = max(bx - cx, cx - (bx + bw), 0.0)
            for iy in range(iy0, iy1 + 1):
                cy = ys[iy]
                dy = max(by - cy, cy - (by + bh), 0.0)
                for iz in range(iz0, iz1 + 1):
                    cz = zs[iz]
                    dz = max(bz - cz, cz - (bz + bd), 0.0)
                    v = max(dx, dy, dz)
                    if v < dist[iy, iz, ix]:
                        dist[iy, iz, ix] = v

    # Boundary: distance to any container face.
    for iy in range(ny):
        for iz in range(nz):
            for ix in range(nx):
                b = min(xs[ix], W - xs[ix], ys[iy], H - ys[iy], zs[iz], D - zs[iz])
                if b < dist[iy, iz, ix]:
                    dist[iy, iz, ix] = b
    return dist, xs, ys, zs, step

def _grid_max_clearance(grid, xs, ys, zs, x, y, z, ow, oh, od, nx, ny, nz):
    """Max over the box footprint of min clearance (approx: sample the 8 corners
    plus interior grid points within the box)."""
    ix0 = int(np.clip(x / (xs[1] - xs[0]), 0, nx - 1))
    ix1 = int(np.clip((x + ow) / (xs[1] - xs[0]), 0, nx - 1))
    iy0 = int(np.clip(y / (ys[1] - ys[0]), 0, ny - 1))
    iy1 = int(np.clip((y + oh) / (ys[1] - ys[0]), 0, ny - 1))
    iz0 = int(np.clip(z / (zs[1] - zs[0]), 0, nz - 1))
    iz1 = int(np.clip((z + od) / (zs[1] - zs[0]), 0, nz - 1))
    if ix0 > ix1:
        ix0, ix1 = ix1, ix0
    if iy0 > iy1:
        iy0, iy1 = iy1, iy0
    if iz0 > iz1:
        iz0, iz1 = iz1, iz0
    sub = grid[iy0:iy1 + 1, iz0:iz1 + 1, ix0:ix1 + 1]
    return float(sub.min()) if sub.size else 0.0

def _place_one(boxes, container, placed_ids, placed_data, box_idx,
              use_grid, grid_ctx, deadline):
    """Find best position/orientation for box_idx. Returns (x,y,z,ow,oh,od) or None.
    Best = maximize min-clearance (largest empty space), tie-break minimize BB volume."""
    W, H, D = container
    w, h, d = boxes[box_idx]

    # Candidate coordinate sets (from placed box edges + 0)
    xs = {0}
    ys = {0}
    zs = {0}
    for bid in placed_ids:
        (px, py, pz, pw, ph, pd) = placed_data[bid]
        xs.add(px)
        xs.add(px + pw)
        ys.add(py)
        ys.add(py + ph)
        zs.add(pz)
        zs.add(pz + pd)

    # Grid-guided candidates: top cells -> add their coords
    extra_xs, extra_ys, extra_zs = [], [], []
    if use_grid and grid_ctx is not None:
        grid, gxs, gys, gzs, step = grid_ctx
        nx, ny, nz = len(gxs), len(gys), len(gzs)
        # sample a coarse set of the highest-clearance cells
        flat = grid.reshape(-1)
        k = min(40, flat.size)
        top = np.argpartition(flat, -k)[-k:]
        for f in top:
            iy, iz, ix = np.unravel_index(f, grid.shape)
            extra_xs.append(gxs[ix])
            extra_ys.append(gys[iy])
            extra_zs.append(gzs[iz])
    xs = sorted(set(xs) | set(extra_xs))
    ys = sorted(set(ys) | set(extra_ys))
    zs = sorted(set(zs) | set(extra_zs))
    xs = [c for c in xs if 0 <= c <= W]
    ys = [c for c in ys if 0 <= c <= H]
    zs = [c for c in zs if 0 <= c <= D]

    best_key = None
    best = None
    for ow, oh, od in _orientations(w, h, d):
        for cx in xs:
            if cx + ow > W:
                continue
            for cy in ys:
                if cy + oh > H:
                    continue
                for cz in zs:
                    if cz + od > D:
                        continue
                    if not _empty_and_inbounds(cx, cy, cz, ow, oh, od,
                                               placed_ids, placed_data, W, H, D):
                        continue
                    if use_grid and grid_ctx is not None:
                        grid, gxs, gys, gzs, step = grid_ctx
                        nx, ny, nz = len(gxs), len(gys), len(gzs)
                        clear = _grid_max_clearance(grid, gxs, gys, gzs,
                                                    cx, cy, cz, ow, oh, od,
                                                    nx, ny, nz)
                        bb = (max(cx + ow, *( [0] )))
                        # bounding-box tie-break: minimize overall extent product
                        mx = max(cx + ow, max([placed_data[b][0] + placed_data[b][3] for b in placed_ids], default=0))
                        my = max(cy + oh, max([placed_data[b][1] + placed_data[b][4] for b in placed_ids], default=0))
                        mz = max(cz + od, max([placed_data[b][2] + placed_data[b][5] for b in placed_ids], default=0))
                        key = (clear, -(mx * my * mz))
                    else:
                        mx = max(cx + ow, max([placed_data[b][0] + placed_data[b][3] for b in placed_ids], default=0))
                        my = max(cy + oh, max([placed_data[b][1] + placed_data[b][4] for b in placed_ids], default=0))
                        mz = max(cz + od, max([placed_data[b][2] + placed_data[b][5] for b in placed_ids], default=0))
                        key = (0.0, -(mx * my * mz))
                    if best_key is None or key > best_key:
                        best_key = key
                        best = (cx, cy, cz, ow, oh, od)
                    if time.time() > deadline:
                        return best
    return best

def _pack(boxes, container, order, deadline, use_grid, step):
    W, H, D = container
    placed_ids = []
    placed_data = {}
    placements = {}
    filled = 0.0
    grid_ctx = None
    for idx in order:
        if time.time() > deadline:
            break
        if use_grid and len(placed_ids) >= 6:
            grid_ctx = _build_distance_grid(placed_data, placed_ids, W, H, D, step)
        res = _place_one(boxes, container, placed_ids, placed_data, idx,
                         use_grid, grid_ctx, deadline)
        if res is None:
            continue
        x, y, z, ow, oh, od = res
        bid = len(placed_ids)
        placed_data[bid] = (x, y, z, ow, oh, od)
        placed_ids.append(bid)
        placements[idx] = (x, y, z, ow, oh, od)
        filled += boxes[idx][0] * boxes[idx][1] * boxes[idx][2]
        if time.time() > deadline:
            break
    return placements, filled

def _local_search(boxes, container, placements, filled, deadline):
    W, H, D = container
    best_pl = dict(placements)
    for _ in range(1):
        improved = False
        for idx in list(best_pl.keys()):
            if time.time() > deadline:
                break
            (bx, by, bz, bw, bh, bd) = best_pl[idx]
            del best_pl[idx]
            # rebuild placed structures without idx
            placed_ids = []
            placed_data = {}
            for j, p in best_pl.items():
                bid = len(placed_ids)
                placed_data[bid] = p
                placed_ids.append(bid)
            res = _place_one(boxes, container, placed_ids, placed_data, idx,
                             False, None, deadline)
            if res is not None and res != (bx, by, bz, bw, bh, bd):
                best_pl[idx] = res
                improved = True
            elif res is not None:
                best_pl[idx] = (bx, by, bz, bw, bh, bd)
            else:
                best_pl[idx] = (bx, by, bz, bw, bh, bd)
        if not improved:
            break
    return best_pl, filled

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    start_time = time.time()
    deadline = start_time + 130.0

    orders = []
    orders.append(sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True))
    orders.append(sorted(range(n), key=lambda i: max(boxes[i]), reverse=True))
    orders.append(sorted(range(n), key=lambda i: sum(boxes[i]), reverse=True))
    orders.append(sorted(range(n), key=lambda i: (max(boxes[i]) / max(min(boxes[i]), 1e-9)), reverse=True))
    orders.append(sorted(range(n), key=lambda i: min(boxes[i]), reverse=True))

    # grid step: ~1/24 of min container dim, capped
    step = max(1.0, min(W, H, D) / 24.0)

    best_result = None
    best_filled = -1.0

    # Run: best-fit (no grid) for all orders, and grid-guided for top 3 orders.
    configs = [(o, False) for o in orders]
    configs = configs + [(orders[i], True) for i in range(min(3, len(orders)))]

    for order, use_grid in configs:
        if time.time() > deadline:
            break
        placements, filled = _pack(boxes, container, order, deadline, use_grid, step)
        if time.time() < deadline - 5.0:
            placements, filled = _local_search(boxes, container, placements, filled, deadline)
        if filled > best_filled:
            best_filled = filled
            best_result = placements

    result = []
    for i in range(n):
        w, h, d = boxes[i]
        if best_result is not None and i in best_result:
            x, y, z, ow, oh, od = best_result[i]
            result.append((0, x, y, z, ow, oh, od))
        else:
            result.append((-1, 0, 0, 0, w, h, d))
    return result
