"""3D Bin Packing: Best-Fit Decreasing with Local Search Refinement."""
import time

# EVOLVE-BLOCK-START
def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []

    container_vol = W * H * D

    # Sort boxes by volume descending
    indexed_boxes = [(w * h * d, i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: -x[0])

    # Bins: list of lists of (x, y, z, w, h, d, orig_idx)
    bins = []
    placement_map = [None] * n  # (bin_idx, x, y, z)

    def overlaps(x1, y1, z1, w1, h1, d1, x2, y2, z2, w2, h2, d2):
        return (x1 < x2 + w2 and x2 < x1 + w1 and
                y1 < y2 + h2 and y2 < y1 + h1 and
                z1 < z2 + d2 and z2 < z1 + d1)

    def find_best_position(bin_boxes, w, h, d):
        """
        Find the best valid position for a box in a bin.
        Heuristic: Minimize (z + y + x) first? Or minimize distance to far walls?
        Let's use a composite score: prioritize low z, then low y, then low x.
        Actually, a better heuristic for packing is to minimize the "waste" or maximize support.
        Let's stick to a strong geometric heuristic:
        1. Generate candidates from corners of existing boxes and (0,0,0).
        2. Filter valid non-overlapping positions.
        3. Score each position: 
           - Primary: Minimize z (stack as low as possible)
           - Secondary: Minimize y
           - Tertiary: Minimize x
           This is effectively "First Fit" in z, then y, then x.
           
           Alternative: Minimize (W - (x+w)) + (H - (y+h)) + (D - (z+d)) 
           This pushes boxes to the origin.
           
           Let's try: Minimize (z, y, x) is standard and works well.
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
            
            # Also add the box's own corners just in case (though they are usually occupied or invalid for new box unless touching)
            # Actually, just the "outer" corners of the union of boxes are usually sufficient.
            # The above adds the "positive" corners of each box.

        # Limit candidates if too many? 40 boxes * 8 corners = 320 candidates. Manageable.
        
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
                # This packs tightly to the origin
                score = (cz, cy, cx)
                
                # Alternative Score: Minimize distance to far walls (push to origin)
                # score = (W - (cx + w)) + (H - (cy + h)) + (D - (cz + d))
                # This is equivalent to maximizing (cx+cy+cz) roughly? No.
                # Let's stick to (cz, cy, cx) as it is stable and effective.
                
                if best_score is None or score < best_score:
                    best_score = score
                    best_pos = (cx, cy, cz)

        return best_pos

    # --- Phase 1: Initial Placement (Best-Fit Decreasing) ---
    for vol, orig_idx, w, h, d in indexed_boxes:
        if time.time() > deadline:
            # Emergency: place in new bin
            bin_idx = len(bins)
            bins.append([(0, 0, 0, w, h, d, orig_idx)])
            placement_map[orig_idx] = (bin_idx, 0, 0, 0)
            continue

        placed = False
        # Try existing bins
        for b_idx in range(len(bins)):
            if time.time() > deadline:
                break
            pos = find_best_position(bins[b_idx], w, h, d)
            if pos is not None:
                bins[b_idx].append((pos[0], pos[1], pos[2], w, h, d, orig_idx))
                placement_map[orig_idx] = (b_idx, pos[0], pos[1], pos[2])
                placed = True
                break

        if not placed:
            # Open new bin
            bin_idx = len(bins)
            bins.append([(0, 0, 0, w, h, d, orig_idx)])
            placement_map[orig_idx] = (bin_idx, 0, 0, 0)

    # --- Phase 2: Local Search Refinement ---
    # Goal: Reduce number of bins (B) by moving boxes from later bins to earlier ones.
    # Also try to improve fill of the last bin if B is fixed (less priority than reducing B).

    def bin_volume(b_idx):
        return sum(bw * bh * bd for (_, _, _, bw, bh, bd, _) in bins[b_idx])

    def try_move_box(bin_from, box_idx_in_bin, bin_to):
        """Try to move a specific box from bin_from to bin_to."""
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
            # Put back
            bins[bin_from].insert(box_idx_in_bin, box)
            return False

    # Iterate to allow cascading moves
    max_iterations = 20
    for _ in range(max_iterations):
        if time.time() > deadline - 5:
            break
        if len(bins) <= 1:
            break

        moved_any = False
        # Try to move boxes from the last bin to any earlier bin
        last_bin_idx = len(bins) - 1
        
        # Iterate in reverse to handle index changes safely
        for box_idx in range(len(bins[last_bin_idx]) - 1, -1, -1):
            if time.time() > deadline - 5:
                break
            # Try all earlier bins
            for target_bin in range(last_bin_idx):
                if try_move_box(last_bin_idx, box_idx, target_bin):
                    moved_any = True
                    break
            if moved_any:
                break
        
        # If we emptied the last bin, it will be removed in the next logic step or just remain empty?
        # The problem says "B is the number of occupied bins". So empty bins don't count.
        # But our `bins` list might have empty bins. We should clean them up or just ignore them in count.
        # For the local search, if a bin becomes empty, we can remove it to simplify.
        if not moved_any:
            # Try to move boxes from second to last bin to last bin? No, that increases B if last was empty?
            # No, we want to minimize B.
            # Let's try to move boxes from *any* bin to *any* earlier bin to consolidate.
            # This is more expensive. Let's stick to last bin reduction for now.
            # If no box could be moved from the last bin, we might be stuck with this B.
            break
        
        # Clean up empty bins (optional, but good for logic)
        # Note: Removing bins changes indices. We must be careful.
        # Since we only moved boxes *out* of the last bin, if it becomes empty, we can pop it.
        if len(bins) > 0 and len(bins[-1]) == 0:
            bins.pop()
            # Update placement_map? No, placement_map points to bin indices.
            # If we remove a bin, all indices after it shift?
            # No, we only remove the LAST bin. So indices of remaining bins are unchanged.
            # However, if we had boxes in bin 5, and we removed bin 5, that's fine.
            # But what if we moved a box from bin 5 to bin 2, and bin 5 becomes empty?
            # Then bin 5 is gone. Any other boxes in bin 5? No, it was empty.
            # So this is safe.

    # Final cleanup: Remove any empty bins from the end
    while len(bins) > 0 and len(bins[-1]) == 0:
        bins.pop()

    # Build result
    placements = []
    for i in range(n):
        w, h, d = boxes[i]
        b_idx, x, y, z = placement_map[i]
        placements.append((b_idx, x, y, z, w, h, d))

    return placements
# EVOLVE-BLOCK-END
