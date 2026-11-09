"""3D Bin Packing: FFD with multiple sorting strategies + local search refinement."""
import time

# EVOLVE-BLOCK-START
def pack(boxes, container):
    start_time = time.time()
    # Use a safety margin. The evaluator kills at 150s.
    # We reserve 5s for cleanup/return, so deadline is 145s.
    deadline = start_time + 145.0

    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []

    container_vol = W * H * D

    def overlaps(x1, y1, z1, w1, h1, d1, x2, y2, z2, w2, h2, d2):
        return (x1 < x2 + w2 and x2 < x1 + w1 and
                y1 < y2 + h2 and y2 < y1 + h1 and
                z1 < z2 + d2 and z2 < z1 + d1)

    def find_best_position(bin_boxes, w, h, d):
        """
        Find the best valid position for a box in a bin.
        Heuristic: Minimize (z, y, x) to pack tightly to the origin.
        """
        if not bin_boxes:
            return (0, 0, 0)

        # Generate candidate positions
        candidates = set()
        candidates.add((0, 0, 0))

        # Add corners of existing boxes
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
                # Score: Minimize z, then y, then x
                score = (cz, cy, cx)
                if best_score is None or score < best_score:
                    best_score = score
                    best_pos = (cx, cy, cz)

        return best_pos

    def run_ffd(order_key):
        """
        Run First Fit Decreasing with a specific sorting key.
        Returns (bins, placement_map, num_bins).
        """
        # Create indexed boxes with the specific key
        indexed_boxes = []
        for i, (w, h, d) in enumerate(boxes):
            key = order_key(w, h, d)
            indexed_boxes.append((key, i, w, h, d))
        
        # Sort by key descending
        indexed_boxes.sort(key=lambda x: x[0], reverse=True)

        bins = []
        placement_map = [None] * n

        for key, orig_idx, w, h, d in indexed_boxes:
            if time.time() > deadline:
                # Emergency placement to ensure validity
                bin_idx = len(bins)
                bins.append([(0, 0, 0, w, h, d, orig_idx)])
                placement_map[orig_idx] = (bin_idx, 0, 0, 0)
                continue

            placed = False
            best_bin = -1
            best_pos = None
            best_score = None

            # Try existing bins
            for b_idx in range(len(bins)):
                if time.time() > deadline:
                    break
                pos = find_best_position(bins[b_idx], w, h, d)
                if pos is not None:
                    score = (pos[2], pos[1], pos[0])  # (z, y, x)
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

        return bins, placement_map, len(bins)

    def local_search(bins, placement_map):
        """
        Refine the packing by trying to move boxes from later bins to earlier ones.
        Modifies bins and placement_map in place.
        Returns the new number of bins.
        """
        def try_move_box(bin_from, box_idx_in_bin, bin_to):
            if bin_from == bin_to or bin_from < 0 or bin_to < 0:
                return False
            if box_idx_in_bin < 0 or box_idx_in_bin >= len(bins[bin_from]):
                return False

            box = bins[bin_from][box_idx_in_bin]
            bx, by, bz, bw, bh, bd, bidx = box

            # Remove from source
            bins[bin_from].pop(box_idx_in_bin)

            # Try to place in target
            pos = find_best_position(bins[bin_to], bw, bh, bd)
            if pos is not None:
                bins[bin_to].append((pos[0], pos[1], pos[2], bw, bh, bd, bidx))
                placement_map[bidx] = (bin_to, pos[0], pos[1], pos[2])
                return True
            else:
                bins[bin_from].insert(box_idx_in_bin, box)
                return False

        def try_swap_boxes(bin1, box_idx1, bin2, box_idx2):
            if bin1 == bin2:
                return False
            if box_idx1 < 0 or box_idx1 >= len(bins[bin1]):
                return False
            if box_idx2 < 0 or box_idx2 >= len(bins[bin2]):
                return False

            box1 = bins[bin1][box_idx1]
            box2 = bins[bin2][box_idx2]

            # Remove both
            bins[bin1].pop(box_idx1)
            bins[bin2].pop(box_idx2)

            # Try to place box1 in bin2 and box2 in bin1
            pos1 = find_best_position(bins[bin2], box1[3], box1[4], box1[5])
            pos2 = find_best_position(bins[bin1], box2[3], box2[4], box2[5])

            if pos1 is not None and pos2 is not None:
                bins[bin2].append((pos1[0], pos1[1], pos1[2], box1[3], box1[4], box1[5], box1[6]))
                bins[bin1].append((pos2[0], pos2[1], pos2[2], box2[3], box2[4], box2[5], box2[6]))
                placement_map[box1[6]] = (bin2, pos1[0], pos1[1], pos1[2])
                placement_map[box2[6]] = (bin1, pos2[0], pos2[1], pos2[2])
                return True
            else:
                # Put back
                bins[bin1].insert(box_idx1, box1)
                bins[bin2].insert(box_idx2, box2)
                return False

        # Aggressive consolidation: repeatedly try to empty the last bin
        max_rounds = 150
        for _ in range(max_rounds):
            if time.time() > deadline - 10:
                break
            if len(bins) <= 1:
                break

            # Clean up any empty bins first
            new_bins = []
            for b in bins:
                if len(b) > 0:
                    new_bins.append(b)
            if len(new_bins) != len(bins):
                bins.clear()
                bins.extend(new_bins)
                # Rebuild placement_map with correct indices
                for i in range(n):
                    old_b, x, y, z = placement_map[i]
                    new_idx = -1
                    for nb in bins:
                        for (bx, by, bz, bw, bh, bd, bidx) in nb:
                            if bidx == i:
                                new_idx = bins.index(nb)
                                break
                        if new_idx >= 0:
                            break
                    if new_idx >= 0:
                        placement_map[i] = (new_idx, x, y, z)
                continue

            last_bin_idx = len(bins) - 1
            if len(bins[last_bin_idx]) == 0:
                bins.pop()
                continue

            moved_any_this_round = False

            # Strategy 1: Try to move each box from the last bin to any earlier bin
            # Sort boxes in last bin by volume ascending to move small ones first (easier to fit)
            box_order = list(range(len(bins[last_bin_idx])))
            box_order.sort(key=lambda idx: bins[last_bin_idx][idx][3] * bins[last_bin_idx][idx][4] * bins[last_bin_idx][idx][5])

            for box_idx in box_order:
                if time.time() > deadline - 10:
                    break
                moved = False
                for target_bin in range(last_bin_idx):
                    if try_move_box(last_bin_idx, box_idx, target_bin):
                        moved = True
                        moved_any_this_round = True
                        break
                if moved:
                    # Box index changed after removal, re-sort
                    box_order = list(range(len(bins[last_bin_idx])))
                    box_order.sort(key=lambda idx: bins[last_bin_idx][idx][3] * bins[last_bin_idx][idx][4] * bins[last_bin_idx][idx][5])

            # Strategy 2: Try swaps between last bin and other bins
            if not moved_any_this_round:
                for box_idx in range(len(bins[last_bin_idx])):
                    if time.time() > deadline - 10:
                        break
                    for other_bin in range(last_bin_idx):
                        for other_box_idx in range(len(bins[other_bin])):
                            if try_swap_boxes(last_bin_idx, box_idx, other_bin, other_box_idx):
                                moved_any_this_round = True
                                break
                        if moved_any_this_round:
                            break
                    if moved_any_this_round:
                        break

            if not moved_any_this_round:
                break

        # Final cleanup of empty bins
        final_bins = [b for b in bins if len(b) > 0]
        if len(final_bins) != len(bins):
            bins.clear()
            bins.extend(final_bins)
            for i in range(n):
                old_b, x, y, z = placement_map[i]
                new_idx = -1
                for nb in bins:
                    for (bx, by, bz, bw, bh, bd, bidx) in nb:
                        if bidx == i:
                            new_idx = bins.index(nb)
                            break
                    if new_idx >= 0:
                        break
                if new_idx >= 0:
                    placement_map[i] = (new_idx, x, y, z)

        return len(bins)

    # Define different sorting strategies
    strategies = [
        lambda w, h, d: w * h * d,  # Volume
        lambda w, h, d: max(w, h, d),  # Max dimension
        lambda w, h, d: 2 * (w * h + h * d + w * d),  # Surface area
        lambda w, h, d: w * h * d + 0.1 * max(w, h, d)  # Mixed
    ]

    best_bins = None
    best_placement = None
    best_bin_count = float('inf')

    # Run FFD with each strategy
    for strategy in strategies:
        if time.time() > deadline - 30:
            break # Reserve time for local search on the best one found so far
            
        bins, placement, num_bins = run_ffd(strategy)
        
        if num_bins < best_bin_count:
            best_bin_count = num_bins
            best_bins = bins
            best_placement = placement

    if best_bins is None:
        # Fallback: just use the first strategy
        best_bins, best_placement, best_bin_count = run_ffd(strategies[0])

    # Apply local search to the best FFD result
    if time.time() < deadline - 5:
        local_search(best_bins, best_placement)
        
        # Recalculate bin count after local search
        # (local_search returns count, but we can also just count non-empty bins)
        # The local_search function modifies bins in place and cleans up empties.
        
    # Build result in original order
    placements = []
    for i in range(n):
        w, h, d = boxes[i]
        b_idx, x, y, z = best_placement[i]
        placements.append((b_idx, x, y, z, w, h, d))

    return placements
# EVOLVE-BLOCK-END
