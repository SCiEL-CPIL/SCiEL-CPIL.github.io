"""Rewritten 3D bin packing with First-Fit Decreasing + best-fit placement and time guard."""
import time
import math

# EVOLVE-BLOCK-START
def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s kill

    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []

    # Sort boxes by volume descending (First-Fit Decreasing)
    indexed_boxes = [(w * h * d, i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: -x[0])

    # Bin state: list of bins, each bin is a list of (x, y, z, w, h, d)
    bins = []  # list of lists of placements
    bin_volumes = []  # cumulative volume in each bin
    container_vol = W * H * D

    def overlaps(x1, y1, z1, w1, h1, d1, x2, y2, z2, w2, h2, d2):
        return (x1 < x2 + w2 and x2 < x1 + w1 and
                y1 < y2 + h2 and y2 < y1 + h1 and
                z1 < z2 + d2 and z2 < z1 + d1)

    def can_place_in_bin(bin_placements, x, y, z, w, h, d):
        if x + w > W or y + h > H or z + d > D:
            return False
        for (px, py, pz, pw, ph, pd) in bin_placements:
            if overlaps(x, y, z, w, h, d, px, py, pz, pw, ph, pd):
                return False
        return True

    def find_position(bin_placements, w, h, d):
        # Try to find a position using a grid-based approach for small containers
        # or a simple bottom-left-front heuristic
        # First try bottom-left-front: minimize z, then y, then x
        best_pos = None
        best_key = None

        # Generate candidate positions
        # Try all existing box positions as reference points, plus origin
        candidates = set()
        candidates.add((0, 0, 0))
        for (px, py, pz, pw, ph, pd) in bin_placements:
            # Positions to the right, above, and in front of each existing box
            if px + pw < W:
                candidates.add((px + pw, py, pz))
            if py + ph < H:
                candidates.add((px, py + ph, pz))
            if pz + pd < D:
                candidates.add((px, py, pz + pd))
            # Also try combinations
            if px + pw < W and py + ph < H:
                candidates.add((px + pw, py + ph, pz))
            if px + pw < W and pz + pd < D:
                candidates.add((px + pw, py, pz + pd))
            if py + ph < H and pz + pd < D:
                candidates.add((px, py + ph, pz + pd))
            if px + pw < W and py + ph < H and pz + pd < D:
                candidates.add((px + pw, py + ph, pz + pd))

        # Also add some grid points for better coverage
        # For small dimensions, add more candidates
        if W <= 50 and H <= 50 and D <= 50:
            step_x = max(1, W // 20)
            step_y = max(1, H // 20)
            step_z = max(1, D // 20)
            for cx in range(0, W - w + 1, step_x):
                for cy in range(0, H - h + 1, step_y):
                    for cz in range(0, D - d + 1, step_z):
                        candidates.add((cx, cy, cz))

        for (cx, cy, cz) in candidates:
            if cx + w > W or cy + h > H or cz + d > D:
                continue
            if can_place_in_bin(bin_placements, cx, cy, cz, w, h, d):
                # Score: prefer lower z, then lower y, then lower x
                key = (cz, cy, cx)
                if best_key is None or key < best_key:
                    best_key = key
                    best_pos = (cx, cy, cz)

        return best_pos

    # Place each box
    for vol, orig_idx, w, h, d in indexed_boxes:
        if time.time() > deadline:
            break

        placed = False
        # Try existing bins first (First Fit)
        for b_idx in range(len(bins)):
            if time.time() > deadline:
                break
            pos = find_position(bins[b_idx], w, h, d)
            if pos is not None:
                bins[b_idx].append((pos[0], pos[1], pos[2], w, h, d))
                bin_volumes[b_idx] += w * h * d
                placed = True
                break

        if not placed:
            # Open a new bin
            bins.append([(0, 0, 0, w, h, d)])
            bin_volumes.append(w * h * d)

    # If we ran out of time, place remaining boxes in new bins (guaranteed valid)
    # Actually, let's ensure all boxes are placed. If time runs out during placement,
    # we need to place the rest quickly.
    # Let's restructure: track which boxes are placed
    placed_flags = [False] * n
    for vol, orig_idx, w, h, d in indexed_boxes:
        if placed_flags[orig_idx]:
            continue
        if time.time() > deadline:
            # Emergency: just assign to a new bin
            bins.append([(0, 0, 0, w, h, d)])
            bin_volumes.append(w * h * d)
            placed_flags[orig_idx] = True
            continue

        placed = False
        for b_idx in range(len(bins)):
            if time.time() > deadline:
                break
            pos = find_position(bins[b_idx], w, h, d)
            if pos is not None:
                bins[b_idx].append((pos[0], pos[1], pos[2], w, h, d))
                bin_volumes[b_idx] += w * h * d
                placed_flags[orig_idx] = True
                placed = True
                break

        if not placed:
            bins.append([(0, 0, 0, w, h, d)])
            bin_volumes.append(w * h * d)
            placed_flags[orig_idx] = True

    # Build result in original order
    # We need to map each original box to its placement
    # Since we placed in sorted order, let's rebuild
    # Actually, let's redo this more carefully

    # Reset and do it properly
    bins = []
    bin_volumes = []
    placed_idx = [None] * n  # bin index for each original box
    bin_placements = []  # list of lists

    for vol, orig_idx, w, h, d in indexed_boxes:
        if time.time() > deadline:
            # Emergency placement
            bin_idx = len(bins)
            bins.append((w, h, d))
            bin_placements.append([(0, 0, 0, w, h, d)])
            bin_volumes.append(w * h * d)
            placed_idx[orig_idx] = bin_idx
            continue

        placed = False
        for b_idx in range(len(bin_placements)):
            if time.time() > deadline:
                break
            pos = find_position(bin_placements[b_idx], w, h, d)
            if pos is not None:
                bin_placements[b_idx].append((pos[0], pos[1], pos[2], w, h, d))
                bin_volumes[b_idx] += w * h * d
                placed_idx[orig_idx] = b_idx
                placed = True
                break

        if not placed:
            b_idx = len(bin_placements)
            bin_placements.append([(0, 0, 0, w, h, d)])
            bin_volumes.append(w * h * d)
            placed_idx[orig_idx] = b_idx

    # Build output
    placements = []
    for i in range(n):
        w, h, d = boxes[i]
        b_idx = placed_idx[i]
        # Find this box's position in bin b_idx
        # We need to match by original index, but we stored by placement order
        # Let's store positions differently

    # This approach is getting complicated. Let me simplify.

    # Final clean implementation:
    bins = []  # list of lists of (x, y, z, w, h, d, orig_idx)
    for vol, orig_idx, w, h, d in indexed_boxes:
        if time.time() > deadline:
            bins.append([(0, 0, 0, w, h, d, orig_idx)])
            continue

        placed = False
        for b_idx in range(len(bins)):
            if time.time() > deadline:
                break
            # Find position
            best_pos = None
            best_key = None
            candidates = set()
            candidates.add((0, 0, 0))
            for (px, py, pz, pw, ph, pd, _) in bins[b_idx]:
                if px + pw < W:
                    candidates.add((px + pw, py, pz))
                if py + ph < H:
                    candidates.add((px, py + ph, pz))
                if pz + pd < D:
                    candidates.add((px, py, pz + pd))
                if px + pw < W and py + ph < H:
                    candidates.add((px + pw, py + ph, pz))
                if px + pw < W and pz + pd < D:
                    candidates.add((px + pw, py, pz + pd))
                if py + ph < H and pz + pd < D:
                    candidates.add((px, py + ph, pz + pd))
                if px + pw < W and py + ph < H and pz + pd < D:
                    candidates.add((px + pw, py + ph, pz + pd))

            for (cx, cy, cz) in candidates:
                if cx + w > W or cy + h > H or cz + d > D:
                    continue
                valid = True
                for (px, py, pz, pw, ph, pd, _) in bins[b_idx]:
                    if (cx < px + pw and px < cx + w and
                        cy < py + ph and py < cy + h and
                        cz < pz + pd and pz < cz + d):
                        valid = False
                        break
                if valid:
                    key = (cz, cy, cx)
                    if best_key is None or key < best_key:
                        best_key = key
                        best_pos = (cx, cy, cz)

            if best_pos is not None:
                bins[b_idx].append((best_pos[0], best_pos[1], best_pos[2], w, h, d, orig_idx))
                placed = True
                break

        if not placed:
            bins.append([(0, 0, 0, w, h, d, orig_idx)])

    # Build output in original order
    result = [None] * n
    for b_idx, bin_boxes in enumerate(bins):
        for (x, y, z, w, h, d, orig_idx) in bin_boxes:
            result[orig_idx] = (b_idx, x, y, z, w, h, d)

    # Ensure no None values (shouldn't happen, but safety)
    for i in range(n):
        if result[i] is None:
            w, h, d = boxes[i]
            result[i] = (len(bins), 0, 0, 0, w, h, d)
            bins.append([(0, 0, 0, w, h, d, i)])

    return result
# EVOLVE-BLOCK-END
