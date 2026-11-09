"""3D Container Loading: dense candidate positions + insertion pass + multiple strategies."""
import time
import itertools
from collections import defaultdict

# EVOLVE-BLOCK-START
def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s wall-clock

    W, H, D = container
    container_vol = W * H * D
    if container_vol <= 0:
        return [(-1, 0, 0, 0, w, h, d) for (w, h, d) in boxes]

    n = len(boxes)
    dims = [(w, h, d) for (w, h, d) in boxes]
    box_vols = [w * h * d for (w, h, d) in dims]

    # Precompute all unique rotations for each box
    def get_rotations(w, h, d):
        perms = set(itertools.permutations((w, h, d)))
        return list(perms)

    box_rots = [get_rotations(*dims[i]) for i in range(n)]

    # Sorting strategies
    def sort_key_volume(i):
        return -box_vols[i]

    def sort_key_maxdim(i):
        return -max(dims[i])

    def sort_key_surfarea(i):
        w, h, d = dims[i]
        return -(2 * (w * h + h * d + w * d))

    def sort_key_vol_over_maxdim(i):
        return -(box_vols[i] / max(dims[i]))

    def sort_key_mindim(i):
        return -min(dims[i])

    def sort_key_vol_then_maxdim(i):
        return (-box_vols[i], -max(dims[i]))

    def sort_key_surfarea_over_maxdim(i):
        w, h, d = dims[i]
        sa = 2 * (w * h + h * d + w * d)
        return -(sa / max(dims[i]))

    sort_strategies = [
        sort_key_volume,
        sort_key_maxdim,
        sort_key_surfarea,
        sort_key_vol_over_maxdim,
        sort_key_mindim,
        sort_key_vol_then_maxdim,
        sort_key_surfarea_over_maxdim,
    ]

    def is_overlapping(x, y, z, w, h, d, placements):
        """Check if box at (x,y,z) with dims (w,h,d) overlaps any placement."""
        for (px, py, pz, pw, ph, pd) in placements:
            if not (x + w <= px or px + pw <= x or
                    y + h <= py or py + ph <= y or
                    z + d <= pz or pz + pd <= z):
                return True
        return False

    def generate_candidate_positions(placements):
        """Generate dense candidate positions from all unique coordinate combinations."""
        xs = [0]
        ys = [0]
        zs = [0]
        for (px, py, pz, pw, ph, pd) in placements:
            xs.append(px)
            xs.append(px + pw)
            ys.append(py)
            ys.append(py + ph)
            zs.append(pz)
            zs.append(pz + pd)
        # Deduplicate
        xs = list(set(xs))
        ys = list(set(ys))
        zs = list(set(zs))
        # Only consider positions within container
        xs = [x for x in xs if x < W]
        ys = [y for y in ys if y < H]
        zs = [z for z in zs if z < D]
        candidates = []
        for x in xs:
            for y in ys:
                for z in zs:
                    candidates.append((x, y, z))
        return candidates

    def place_box(box_idx, placements, candidate_cache):
        """Try to place box_idx using cached candidate positions.
        Returns (x, y, z, rw, rh, rd) or None."""
        candidates = candidate_cache
        best = None
        best_score = -1

        for (cx, cy, cz) in candidates:
            for (rw, rh, rd) in box_rots[box_idx]:
                if cx + rw > W or cy + rh > H or cz + rd > D:
                    continue
                if is_overlapping(cx, cy, cz, rw, rh, rd, placements):
                    continue
                # Valid placement - use heuristic score to prefer "tight" placements
                # Prefer lower z, then lower y, then lower x to pack bottom-front
                score = -(cz * 1000 + cy * 100 + cx)
                if best is None or score > best_score:
                    best = (cx, cy, cz, rw, rh, rd)
                    best_score = score
                    break
            if best is not None:
                break

        return best

    def rebuild_candidate_cache(placements):
        """Rebuild the candidate position cache after adding a placement."""
        return generate_candidate_positions(placements)

    def solve_with_order(order):
        """Solve using a specific box ordering. Returns (filled_vol, result_list)."""
        placements = []
        result = [None] * n
        filled_vol = 0
        unplaced = []

        # Build initial candidate cache
        candidate_cache = generate_candidate_positions(placements)

        for idx in order:
            if time.time() > deadline:
                break
            pos = place_box(idx, placements, candidate_cache)
            if pos is not None:
                x, y, z, rw, rh, rd = pos
                placements.append((x, y, z, rw, rh, rd))
                result[idx] = (0, x, y, z, rw, rh, rd)
                filled_vol += box_vols[idx]
                # Rebuild candidate cache (can be expensive but necessary for accuracy)
                candidate_cache = rebuild_candidate_cache(placements)
            else:
                unplaced.append(idx)

        # Final insertion pass: try to fit remaining unplaced boxes
        if time.time() < deadline - 5:
            # Sort unplaced by volume descending for better chance
            unplaced.sort(key=lambda i: -box_vols[i])
            candidate_cache = rebuild_candidate_cache(placements)
            for idx in unplaced:
                if time.time() > deadline - 2:
                    break
                pos = place_box(idx, placements, candidate_cache)
                if pos is not None:
                    x, y, z, rw, rh, rd = pos
                    placements.append((x, y, z, rw, rh, rd))
                    result[idx] = (0, x, y, z, rw, rh, rd)
                    filled_vol += box_vols[idx]
                    candidate_cache = rebuild_candidate_cache(placements)

        # Fill in any None entries
        for i in range(n):
            if result[i] is None:
                w, h, d = dims[i]
                result[i] = (-1, 0, 0, 0, w, h, d)

        return filled_vol, result

    best_filled = 0
    best_result = [(-1, 0, 0, 0, w, h, d) for (w, h, d) in dims]

    for strat_idx, key_func in enumerate(sort_strategies):
        if time.time() > deadline:
            break
        order = sorted(range(n), key=key_func)
        filled, result = solve_with_order(order)
        if filled > best_filled:
            best_filled = filled
            best_result = result

    return best_result
# EVOLVE-BLOCK-END
