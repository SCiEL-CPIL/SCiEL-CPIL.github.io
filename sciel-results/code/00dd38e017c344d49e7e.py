"""3D Bin Packing with corner-point heuristic and local search."""
import time
import random
import math
from typing import List, Tuple
from collections import defaultdict


def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []

    deadline = time.time() + 115.0
    container_vol = W * H * D

    # Precompute box volumes
    box_vols = [w * h * d for w, h, d in boxes]

    # Sort indices by volume descending
    indices = list(range(n))
    indices_sorted = sorted(indices, key=lambda i: box_vols[i], reverse=True)

    def find_best_position(items, w, h, d, W, H, D):
        """
        Find the best position (x, y, z) where box (w, h, d) fits without overlap.
        Uses corner-point heuristic: generate candidate positions from box faces,
        and select the one that minimizes distance to origin (compactness).
        """
        if not items:
            if w <= W and h <= H and d <= D:
                return (0, 0, 0)
            return None

        # Collect unique coordinates from box faces
        x_coords = {0}
        y_coords = {0}
        z_coords = {0}
        for (ex, ey, ez, ew, eh, ed) in items:
            x_coords.add(ex)
            x_coords.add(ex + ew)
            y_coords.add(ey)
            y_coords.add(ey + eh)
            z_coords.add(ez)
            z_coords.add(ez + ed)

        # Filter to valid positions
        x_coords = sorted(x for x in x_coords if x + w <= W)
        y_coords = sorted(y for y in y_coords if y + h <= H)
        z_coords = sorted(z for z in z_coords if z + d <= D)

        if not x_coords or not y_coords or not z_coords:
            return None

        # Generate candidate positions
        candidates = []
        for cx in x_coords:
            for cy in y_coords:
                for cz in z_coords:
                    candidates.append((cx, cy, cz))

        # Sort by distance to origin (compactness)
        candidates.sort(key=lambda c: c[0] + c[1] + c[2])

        for (cx, cy, cz) in candidates:
            # Check overlap with all existing items
            overlap = False
            for (ex, ey, ez, ew, eh, ed) in items:
                if (cx < ex + ew and ex < cx + w and
                    cy < ey + eh and ey < cy + h and
                    cz < ez + ed and ez < cz + d):
                    overlap = True
                    break
            if not overlap:
                return (cx, cy, cz)

        return None

    def pack_with_order(ordering):
        """Pack boxes in given order. Returns placements list or None."""
        state = {}
        placements = [None] * n
        all_placed = True

        for box_idx in ordering:
            if time.time() > deadline:
                all_placed = False
                break

            w, h, d = boxes[box_idx]

            # Try existing bins
            best_bin = None
            best_pos = None
            best_score = float('inf')

            # Sort bins by current fill volume descending
            bin_ids = list(state.keys())
            bin_vols = {bid: sum(ew * eh * ed for (_, _, _, ew, eh, ed) in state[bid]) for bid in bin_ids}
            bin_ids_sorted = sorted(bin_ids, key=lambda bid: bin_vols[bid], reverse=True)

            for bin_id in bin_ids_sorted:
                pos = find_best_position(state[bin_id], w, h, d, W, H, D)
                if pos is not None:
                    # Score: prefer positions closer to origin
                    score = pos[0] + pos[1] + pos[2]
                    if score < best_score:
                        best_score = score
                        best_bin = bin_id
                        best_pos = pos

            if best_bin is not None:
                state[best_bin] = state[best_bin] + [(best_pos[0], best_pos[1], best_pos[2], w, h, d)]
                placements[box_idx] = (best_bin, best_pos[0], best_pos[1], best_pos[2], w, h, d)
            else:
                # Open new bin
                new_bin_id = max(state.keys()) + 1 if state else 0
                state[new_bin_id] = [(0, 0, 0, w, h, d)]
                placements[box_idx] = (new_bin_id, 0, 0, 0, w, h, d)

        if all_placed and all(p is not None for p in placements):
            return placements
        return None

    def compute_score(placements, W, H, D):
        if not placements:
            return 0.0
        bins_used = set(p[0] for p in placements)
        B = len(bins_used)
        last_bin = max(bins_used)
        last_bin_vol = sum(p[4] * p[5] * p[6] for p in placements if p[0] == last_bin)
        return B + 0.1 * (1 - last_bin_vol / container_vol)

    def local_search_polish(placements, W, H, D, max_iterations=50):
        """
        Local search: try to move boxes to other bins to reduce total bins used.
        """
        if not placements:
            return placements

        bin_boxes = defaultdict(list)
        for i, (b, x, y, z, w, h, d) in enumerate(placements):
            bin_boxes[b].append(i)

        current_score = compute_score(placements, W, H, D)
        improved = True
        iterations = 0

        while improved and iterations < max_iterations and time.time() < deadline - 5:
            improved = False
            iterations += 1

            for box_idx in list(range(n)):
                if time.time() > deadline - 5:
                    break
                b, x, y, z, w, h, d = placements[box_idx]
                other_bins = [bid for bid in bin_boxes if bid != b]
                for target_bin in other_bins:
                    target_items = [(placements[j][1], placements[j][2], placements[j][3],
                                     placements[j][4], placements[j][5], placements[j][6])
                                    for j in bin_boxes[target_bin]]
                    pos = find_best_position(target_items, w, h, d, W, H, D)
                    if pos is not None:
                        new_placements = list(placements)
                        new_placements[box_idx] = (target_bin, pos[0], pos[1], pos[2], w, h, d)
                        new_score = compute_score(new_placements, W, H, D)
                        if new_score < current_score:
                            placements = new_placements
                            current_score = new_score
                            bin_boxes[b].remove(box_idx)
                            if not bin_boxes[b]:
                                del bin_boxes[b]
                            bin_boxes[target_bin].append(box_idx)
                            improved = True
                            break

        return placements

    # Generate diverse orderings
    orderings = []
    orderings.append(indices_sorted)
    orderings.append(list(reversed(indices_sorted)))
    orderings.append(indices)
    orderings.append(sorted(indices, key=lambda i: max(boxes[i]), reverse=True))
    orderings.append(sorted(indices, key=lambda i: boxes[i][0] + boxes[i][1] + boxes[i][2], reverse=True))
    orderings.append(sorted(indices, key=lambda i: min(boxes[i]), reverse=True))
    orderings.append(sorted(indices, key=lambda i: math.sqrt(box_vols[i]), reverse=True))

    # Run all orderings and keep the best
    best_result = None
    best_score = float('inf')

    for ordering in orderings:
        if time.time() > deadline:
            break
        result = pack_with_order(ordering)
        if result is not None:
            score = compute_score(result, W, H, D)
            if score < best_score:
                best_score = score
                best_result = result

    # Fallback
    if best_result is None:
        placements = []
        b, x, y, z, row_h, layer_d = 0, 0, 0, 0, 0, 0
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
        best_result = placements

    # Local search polish
    if best_result and time.time() < deadline - 10:
        best_result = local_search_polish(best_result, W, H, D)

    # Relabel bins to be consecutive
    if best_result:
        bin_labels = sorted(set(p[0] for p in best_result))
        label_map = {old: new for new, old in enumerate(bin_labels)}
        best_result = [(label_map[p[0]], p[1], p[2], p[3], p[4], p[5], p[6]) for p in best_result]

    return best_result
