"""3D Bin Packing: Best-Fit Decreasing with Enhanced Local Search and Swap."""
import time
import random
import math

# EVOLVE-BLOCK-START
def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []

    container_vol = W * H * D

    # Sort boxes by volume descending, with tie-breaking by max dimension descending
    indexed_boxes = [(w * h * d, max(w, h, d), i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: (-x[0], -x[1]))
    indexed_boxes = [(vol, i, w, h, d) for (vol, _, i, w, h, d) in indexed_boxes]

    # Bins: list of lists of (x, y, z, w, h, d, orig_idx)
    bins = []
    placement_map = [None] * n  # (bin_idx, x, y, z)

    def overlaps(x1, y1, z1, w1, h1, d1, x2, y2, z2, w2, h2, d2):
        return (x1 < x2 + w2 and x2 < x1 + w1 and
                y1 < y2 + h2 and y2 < y1 + h1 and
                z1 < z2 + d2 and z2 < z1 + d1)

    def find_best_position(bin_boxes, w, h, d, mode="min"):
        """
        Find the best valid position for a box in a bin.
        mode='min': minimize (z, y, x) - pack to origin
        mode='max': maximize (z, y, x) - pack to far walls (for last bin filling)
        """
        if not bin_boxes:
            return (0, 0, 0)

        # Generate candidate positions from corners of existing boxes and (0,0,0)
        candidates = set()
        candidates.add((0, 0, 0))

        for (px, py, pz, pw, ph, pd, _) in bin_boxes:
            # Right face
            if px + pw <= W:
                candidates.add((px + pw, py, pz))
                candidates.add((px + pw, py + ph, pz))
                candidates.add((px + pw, py, pz + pd))
                candidates.add((px + pw, py + ph, pz + pd))
            # Top face
            if py + ph <= H:
                candidates.add((px, py + ph, pz))
                candidates.add((px + pw, py + ph, pz))
                candidates.add((px, py + ph, pz + pd))
                candidates.add((px + pw, py + ph, pz + pd))
            # Front face
            if pz + pd <= D:
                candidates.add((px, py, pz + pd))
                candidates.add((px + pw, py, pz + pd))
                candidates.add((px, py + ph, pz + pd))
                candidates.add((px + pw, py + ph, pz + pd))

        best_pos = None
        best_score = None

        for (cx, cy, cz) in candidates:
            if cx + w > W or cy + h > H or cz + d > D:
                continue

            # Check overlap
            valid = True
            for (px, py, pz, pw, ph, pd, _) in bin_boxes:
                if overlaps(cx, cy, cz, w, h, d, px, py, pz, pw, ph, pd):
                    valid = False
                    break

            if valid:
                if mode == "min":
                    score = (cz, cy, cx)
                    if best_score is None or score < best_score:
                        best_score = score
                        best_pos = (cx, cy, cz)
                else:
                    score = (cz, cy, cx)
                    if best_score is None or score > best_score:
                        best_score = score
                        best_pos = (cx, cy, cz)

        return best_pos

    # --- Phase 1: Initial Placement (Best-Fit Decreasing) ---
    for vol, orig_idx, w, h, d in indexed_boxes:
        if time.time() > deadline:
            bin_idx = len(bins)
            bins.append([(0, 0, 0, w, h, d, orig_idx)])
            placement_map[orig_idx] = (bin_idx, 0, 0, 0)
            continue

        placed = False
        best_bin = -1
        best_pos = None
        best_score = None

        for b_idx in range(len(bins)):
            if time.time() > deadline:
                break
            pos = find_best_position(bins[b_idx], w, h, d, mode="min")
            if pos is not None:
                score = (pos[2], pos[1], pos[0])
                if best_score is None or score < best_score:
                    best_score = score
                    best_bin = b_idx
                    best_pos = pos

        if best_bin >= 0:
            bins[best_bin].append((best_pos[0], best_pos[1], best_pos[2], w, h, d, orig_idx))
            placement_map[orig_idx] = (best_bin, best_pos[0], best_pos[1], best_pos[2])
            placed = True

        if not placed:
            bin_idx = len(bins)
            bins.append([(0, 0, 0, w, h, d, orig_idx)])
            placement_map[orig_idx] = (bin_idx, 0, 0, 0)

    # --- Phase 2: Local Search Refinement ---
    def bin_volume(b):
        return sum(bw * bh * bd for (_, _, _, bw, bh, bd, _) in b)

    def try_move_box(bin_from, box_idx_in_bin, bin_to, mode="min"):
        if bin_from == bin_to or bin_from < 0 or bin_to < 0:
            return False
        if box_idx_in_bin < 0 or box_idx_in_bin >= len(bins[bin_from]):
            return False

        box = bins[bin_from][box_idx_in_bin]
        bx, by, bz, bw, bh, bd, bidx = box

        bins[bin_from].pop(box_idx_in_bin)

        pos = find_best_position(bins[bin_to], bw, bh, bd, mode=mode)
        if pos is not None:
            bins[bin_to].append((pos[0], pos[1], pos[2], bw, bh, bd, bidx))
            placement_map[bidx] = (bin_to, pos[0], pos[1], pos[2])
            return True
        else:
            bins[bin_from].insert(box_idx_in_bin, box)
            return False

    def try_swap_boxes(bin_a, idx_a, bin_b, idx_b):
        """Try to swap two boxes between bins."""
        if bin_a == bin_b:
            return False
        if idx_a < 0 or idx_a >= len(bins[bin_a]) or idx_b < 0 or idx_b >= len(bins[bin_b]):
            return False

        box_a = bins[bin_a][idx_a]
        box_b = bins[bin_b][idx_b]

        # Remove both
        bins[bin_a].pop(idx_a)
        bins[bin_b].pop(idx_b)

        # Try to place box_b in bin_a and box_a in bin_b
        pos_b_in_a = find_best_position(bins[bin_a], box_b[3], box_b[4], box_b[5], mode="min")
        if pos_b_in_a is None:
            # Restore
            bins[bin_a].insert(idx_a, box_a)
            bins[bin_b].insert(idx_b, box_b)
            return False

        bins[bin_a].append((pos_b_in_a[0], pos_b_in_a[1], pos_b_in_a[2], box_b[3], box_b[4], box_b[5], box_b[6]))
        placement_map[box_b[6]] = (bin_a, pos_b_in_a[0], pos_b_in_a[1], pos_b_in_a[2])

        pos_a_in_b = find_best_position(bins[bin_b], box_a[3], box_a[4], box_a[5], mode="min")
        if pos_a_in_b is None:
            # Restore
            # Remove box_b from bin_a
            for i in range(len(bins[bin_a]) - 1, -1, -1):
                if bins[bin_a][i][6] == box_b[6]:
                    bins[bin_a].pop(i)
                    break
            bins[bin_a].insert(idx_a, box_a)
            bins[bin_b].insert(idx_b, box_b)
            # Restore placement map
            for i in range(n):
                b_idx, x, y, z = placement_map[i]
                # Re-find
                for nb in bins:
                    for (bx, by, bz, bw, bh, bd, bidx) in nb:
                        if bidx == i:
                            placement_map[i] = (bins.index(nb), bx, by, bz)
                            break
                    else:
                        continue
                    break
            return False

        bins[bin_b].append((pos_a_in_b[0], pos_a_in_b[1], pos_a_in_b[2], box_a[3], box_a[4], box_a[5], box_a[6]))
        placement_map[box_a[6]] = (bin_b, pos_a_in_b[0], pos_a_in_b[1], pos_a_in_b[2])

        return True

    # Aggressive consolidation: repeatedly try to empty the last bin
    max_rounds = 80
    for _ in range(max_rounds):
        if time.time() > deadline - 15:
            break
        if len(bins) <= 1:
            break

        # Clean up any empty bins
        new_bins = []
        for b in bins:
            if len(b) > 0:
                new_bins.append(b)
        if len(new_bins) != len(bins):
            bins.clear()
            bins.extend(new_bins)
            for i in range(n):
                for nb_idx, nb in enumerate(bins):
                    for (bx, by, bz, bw, bh, bd, bidx) in nb:
                        if bidx == i:
                            placement_map[i] = (nb_idx, bx, by, bz)
                            break
                    else:
                        continue
                    break
            continue

        last_bin_idx = len(bins) - 1
        if len(bins[last_bin_idx]) == 0:
            bins.pop()
            continue

        moved_any_this_round = False
        box_order = list(range(len(bins[last_bin_idx])))
        box_order.sort(key=lambda idx: bins[last_bin_idx][idx][3] * bins[last_bin_idx][idx][4] * bins[last_bin_idx][idx][5])

        for box_idx in box_order:
            if time.time() > deadline - 15:
                break
            if box_idx >= len(bins[last_bin_idx]):
                continue
            for target_bin in range(last_bin_idx):
                if try_move_box(last_bin_idx, box_idx, target_bin, mode="min"):
                    moved_any_this_round = True
                    break
            if moved_any_this_round:
                break

        if not moved_any_this_round:
            # Try swaps to improve packing in the last bin
            if len(bins[last_bin_idx]) >= 2:
                improved = False
                for i in range(len(bins[last_bin_idx])):
                    if time.time() > deadline - 15:
                        break
                    for j in range(i + 1, len(bins[last_bin_idx])):
                        if time.time() > deadline - 15:
                            break
                        for other_bin in range(last_bin_idx):
                            for k in range(len(bins[other_bin])):
                                if try_swap_boxes(last_bin_idx, i, other_bin, k):
                                    improved = True
                                    break
                                if time.time() > deadline - 15:
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    continue

            break

    # Final cleanup: Remove any empty bins from the end
    while len(bins) > 0 and len(bins[-1]) == 0:
        bins.pop()

    # Renumber bins to be consecutive
    bin_remap = {}
    new_bins = []
    for i, b in enumerate(bins):
        if len(b) > 0:
            bin_remap[i] = len(new_bins)
            new_bins.append(b)
    bins = new_bins

    # Rebuild placement_map with remapped bin indices
    for i in range(n):
        old_b, x, y, z = placement_map[i]
        if old_b in bin_remap:
            placement_map[i] = (bin_remap[old_b], x, y, z)
        else:
            # Should not happen, but handle gracefully
            placement_map[i] = (0, x, y, z)

    # Build result
    placements = []
    for i in range(n):
        w, h, d = boxes[i]
        b_idx, x, y, z = placement_map[i]
        placements.append((b_idx, x, y, z, w, h, d))

    return placements
# EVOLVE-BLOCK-END
