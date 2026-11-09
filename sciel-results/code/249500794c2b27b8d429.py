import time
import itertools
import random

def load(boxes, container):
    start_time = time.time()
    # Explicit deadline guard: return before the 150s hard kill with a safety margin.
    deadline = start_time + 140.0

    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []

    # Precompute unique rotations per box (list of (rw, rh, rd))
    rotations = [list(set(itertools.permutations(b))) for b in boxes]

    # Precompute volume per box
    vols = [b[0] * b[1] * b[2] for b in boxes]

    # Spatial hash grid for fast overlap queries.
    # Cell size chosen to be at least as large as the largest box dimension
    # so a query only needs to inspect the box's own cell (and neighbors).
    max_dim = max(max(b) for b in boxes)
    cell = max(1, max_dim)

    def grid_key(x, y, z):
        return (x // cell, y // cell, z // cell)

    def overlaps(grid, cx, cy, cz, rw, rh, rd, placed):
        # Quick bounds already checked by caller.
        # Check grid cells overlapping the box footprint.
        x0, x1 = cx // cell, (cx + rw - 1) // cell
        y0, y1 = cy // cell, (cy + rh - 1) // cell
        z0, z1 = cz // cell, (cz + rd - 1) // cell
        for gx in range(x0, x1 + 1):
            for gy in range(y0, y1 + 1):
                for gz in range(z0, z1 + 1):
                    for (px, py, pz, pw, ph, pd) in grid.get((gx, gy, gz), ()):
                        if (cx < px + pw and px < cx + rw and
                                cy < py + ph and py < cy + rh and
                                cz < pz + pd and pz < cz + rd):
                            return True
        return False

    def insert_box(grid, placed, x, y, z, w, h, d):
        placed.append((x, y, z, w, h, d))
        x0, x1 = x // cell, (x + w - 1) // cell
        y0, y1 = y // cell, (y + h - 1) // cell
        z0, z1 = z // cell, (z + d - 1) // cell
        for gx in range(x0, x1 + 1):
            for gy in range(y0, y1 + 1):
                for gz in range(z0, z1 + 1):
                    grid.setdefault((gx, gy, gz), []).append((x, y, z, w, h, d))

    def candidate_positions(placed, dense=False):
        """Generate candidate (x, y, z) points from placed boxes."""
        cands = {(0, 0, 0)}
        for (x, y, z, w, h, d) in placed:
            cands.add((x, y, z + d))            # top
            cands.add((x + w, y, z))            # right
            cands.add((x, y + h, z))            # front
            cands.add((x + w, y + h, z))        # right-front
            cands.add((x + w, y, z + d))        # top-right
            cands.add((x, y + h, z + d))        # top-front
            cands.add((x + w, y + h, z + d))    # top-right-front
            if dense:
                # Shelf-edge midpoints to fill gaps
                cands.add((x + w // 2, y, z))
                cands.add((x, y + h // 2, z))
                cands.add((x + w // 2, y + h // 2, z))
                cands.add((x, y, z + d // 2))
                cands.add((x + w, y, z + d // 2))
        return list(cands)

    def try_place(placed, grid, idx, cands, dense):
        """Try to place box idx at best candidate position. Returns (x,y,z,rw,rh,rd) or None."""
        w, h, d = boxes[idx]
        best_pos = None
        best_rot = None
        best_score = float('inf')
        for rw, rh, rd in rotations[idx]:
            if rw > W or rh > H or rd > D:
                continue
            for cx, cy, cz in cands:
                if cx + rw > W or cy + rh > H or cz + rd > D:
                    continue
                if overlaps(grid, cx, cy, cz, rw, rh, rd, placed):
                    continue
                # BLB lexicographic score: minimize z, then y, then x
                score = cz * 1000000 + cy * 1000 + cx
                if score < best_score:
                    best_score = score
                    best_pos = (cx, cy, cz)
                    best_rot = (rw, rh, rd)
        if best_pos is None:
            return None
        x, y, z = best_pos
        rw, rh, rd = best_rot
        insert_box(grid, placed, x, y, z, rw, rh, rd)
        return (x, y, z, rw, rh, rd)

    def greedy_solve(ordered_indices):
        """Run BLB greedy on a fixed ordering. Returns (results, placed, placed_set)."""
        placed = []
        grid = {}
        results = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
        placed_set = set()
        for idx in ordered_indices:
            if time.time() > deadline:
                return None  # timeout
            cands = candidate_positions(placed, dense=False)
            res = try_place(placed, grid, idx, cands, dense=False)
            if res is not None:
                x, y, z, rw, rh, rd = res
                results[idx] = (0, x, y, z, rw, rh, rd)
                placed_set.add(idx)
        return results, placed, placed_set

    def final_insertion(results, placed, placed_set):
        """Attempt to insert unplaced boxes into the current solution using dense candidates."""
        grid = {}
        for (x, y, z, w, h, d) in placed:
            insert_box(grid, placed, x, y, z, w, h, d)  # rebuild grid (idempotent-ish; rebuild fresh)
        # Rebuild grid cleanly
        grid = {}
        for (x, y, z, w, h, d) in placed:
            insert_box(grid, placed, x, y, z, w, h, d)
        # Unplaced boxes in decreasing volume order
        unplaced = [i for i in range(n) if i not in placed_set]
        unplaced.sort(key=lambda i: vols[i], reverse=True)
        for idx in unplaced:
            if time.time() > deadline:
                break
            cands = candidate_positions(placed, dense=True)
            res = try_place(placed, grid, idx, cands, dense=True)
            if res is not None:
                x, y, z, rw, rh, rd = res
                results[idx] = (0, x, y, z, rw, rh, rd)
                placed_set.add(idx)
        return results, placed, placed_set

    def vol_of(results):
        return sum(r[4] * r[5] * r[6] for r in results if r[0] == 0)

    indices = list(range(n))

    # Deterministic sorting strategies
    strategies = [
        lambda i: vols[i],
        lambda i: max(boxes[i]),
        lambda i: (boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2]),
        lambda i: boxes[i][0] * boxes[i][1],
        lambda i: -min(boxes[i]),
    ]

    best_results = None
    best_vol = 0

    for sort_func in strategies:
        if time.time() > deadline - 5.0:
            break
        ordered = sorted(indices, key=sort_func, reverse=True)
        out = greedy_solve(ordered)
        if out is None:
            continue
        results, placed, placed_set = out
        # Final insertion pass to recover volume
        results, placed, placed_set = final_insertion(results, placed, placed_set)
        v = vol_of(results)
        if v > best_vol:
            best_vol = v
            best_results = results

    # Random restarts
    if best_results is not None and time.time() < deadline - 15.0:
        attempts = 0
        while time.time() < deadline - 10.0 and attempts < 40:
            random.shuffle(indices)
            out = greedy_solve(indices)
            attempts += 1
            if out is None:
                continue
            results, placed, placed_set = out
            results, placed, placed_set = final_insertion(results, placed, placed_set)
            v = vol_of(results)
            if v > best_vol:
                best_vol = v
                best_results = results

    # Fallback
    if best_results is None:
        best_results = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    return best_results
